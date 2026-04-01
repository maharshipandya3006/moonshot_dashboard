
import os
import sys
import time
import random
import asyncio
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    BRANDS, PRODUCTS_PER_BRAND, REVIEWS_PER_PRODUCT,
    DATA_DIR, PRODUCTS_CSV, REVIEWS_CSV, SCRAPE_DELAY, AMAZON_BASE_URL
)

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


class AmazonScraper:
    def __init__(self):
        self.products = []
        self.reviews = []

    # ── Helpers ──────────────────────────────────────────────────────────────

    async def _setup_page(self, browser):
        context = await browser.new_context(
            user_agent=random.choice(USER_AGENTS),
            viewport={"width": 1280, "height": 800},
            locale="en-IN",
            timezone_id="Asia/Kolkata",
            extra_http_headers={
                "Accept-Language": "en-IN,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
        )
        page = await context.new_page()
        # Block images/fonts to speed up scraping
        await page.route("**/*.{png,jpg,jpeg,gif,webp,svg,woff,woff2,ttf}", lambda r: r.abort())
        return page

    async def _safe_text(self, element, selector: str, default="") -> str:
        try:
            el = await element.query_selector(selector)
            if el:
                return (await el.inner_text()).strip()
        except Exception:
            pass
        return default

    async def _safe_attr(self, element, selector: str, attr: str, default="") -> str:
        try:
            el = await element.query_selector(selector)
            if el:
                val = await el.get_attribute(attr)
                return val.strip() if val else default
        except Exception:
            pass
        return default

    def _parse_price(self, price_str: str) -> float:
        try:
            cleaned = price_str.replace("₹", "").replace(",", "").strip()
            return float(cleaned)
        except (ValueError, AttributeError):
            return 0.0

    def _parse_rating(self, rating_str: str) -> float:
        try:
            return float(rating_str.split()[0])
        except (ValueError, AttributeError, IndexError):
            return 0.0

    def _parse_review_count(self, count_str: str) -> int:
        try:
            cleaned = count_str.replace(",", "").replace("ratings", "").replace("reviews", "").strip()
            return int(cleaned.split()[0])
        except (ValueError, AttributeError, IndexError):
            return 0

    # ── Search results page ──────────────────────────────────────────────────

    async def scrape_search_page(self, page, brand: str) -> list[dict]:
        """Scrape product listings from Amazon search results."""
        query = f"{brand} luggage trolley bag"
        url = f"{AMAZON_BASE_URL}/s?k={query.replace(' ', '+')}&rh=n%3A1350380031"

        print(f"   🔍 Searching: {query}")
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(SCRAPE_DELAY + random.uniform(0.5, 1.5))

            # Check for CAPTCHA
            if await page.query_selector("[action='/errors/validateCaptcha']"):
                print(f"   ⚠️  CAPTCHA detected for {brand}. Skipping.")
                return []

            results = await page.query_selector_all('[data-component-type="s-search-result"]')
            products = []

            for result in results[:PRODUCTS_PER_BRAND]:
                try:
                    asin = await result.get_attribute("data-asin") or ""
                    title = await self._safe_text(result, "h2 a span")
                    price_str = await self._safe_text(result, ".a-price-whole")
                    mrp_str = await self._safe_text(result, ".a-text-price .a-offscreen")
                    rating_str = await self._safe_text(result, ".a-icon-alt")
                    review_count_str = await self._safe_text(result, ".s-underline-text")
                    product_url = await self._safe_attr(result, "h2 a", "href")

                    selling_price = self._parse_price(price_str)
                    mrp = self._parse_price(mrp_str)
                    if mrp == 0:
                        mrp = selling_price
                    discount_pct = round((mrp - selling_price) / mrp * 100) if mrp > 0 else 0
                    rating = self._parse_rating(rating_str)
                    review_count = self._parse_review_count(review_count_str)

                    if not title or selling_price == 0:
                        continue

                    size = "Unknown"
                    for sz in ["Cabin", "cabin", "55", "65", "68", "75", "77", "80"]:
                        if sz.lower() in title.lower():
                            size = "Cabin (55 cm)" if "55" in sz or "cabin" in sz.lower() else f"Large ({sz} cm)"
                            break

                    products.append({
                        "asin": asin,
                        "brand": brand,
                        "title": title,
                        "product_line": title.split()[1] if len(title.split()) > 1 else brand,
                        "size": size,
                        "mrp": mrp,
                        "selling_price": selling_price,
                        "discount_pct": discount_pct,
                        "rating": rating,
                        "review_count": review_count,
                        "segment": "unknown",
                        "url": AMAZON_BASE_URL + product_url if product_url else "",
                    })
                except Exception as e:
                    print(f"     ⚠️  Error parsing product: {e}")
                    continue

            print(f"   ✅ Found {len(products)} products for {brand}")
            return products

        except PlaywrightTimeout:
            print(f"   ⚠️  Timeout for {brand}.")
            return []
        except Exception as e:
            print(f"   ❌ Error scraping {brand}: {e}")
            return []

    # ── Review page ──────────────────────────────────────────────────────────

    async def scrape_reviews(self, page, asin: str, brand: str, product_title: str) -> list[dict]:
        """Scrape reviews for a given ASIN."""
        url = f"{AMAZON_BASE_URL}/product-reviews/{asin}?sortBy=recent&pageSize=50"
        reviews = []

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(SCRAPE_DELAY + random.uniform(0.5, 2.0))

            if await page.query_selector("[action='/errors/validateCaptcha']"):
                print(f"     ⚠️  CAPTCHA on reviews for {asin}")
                return []

            review_elements = await page.query_selector_all('[data-hook="review"]')

            for el in review_elements[:REVIEWS_PER_PRODUCT]:
                try:
                    review_id = await el.get_attribute("id") or f"R{asin[:5]}"
                    stars_text = await self._safe_text(el, '[data-hook="review-star-rating"] .a-icon-alt')
                    stars = int(float(stars_text.split()[0])) if stars_text else 3
                    review_text = await self._safe_text(el, '[data-hook="review-body"] span')
                    date_text = await self._safe_text(el, '[data-hook="review-date"]')
                    verified = bool(await el.query_selector('[data-hook="avp-badge"]'))

                    if stars >= 4:
                        sentiment = "positive"
                    elif stars == 3:
                        sentiment = "neutral"
                    else:
                        sentiment = "negative"

                    reviews.append({
                        "review_id": review_id,
                        "asin": asin,
                        "brand": brand,
                        "product_title": product_title,
                        "stars": stars,
                        "sentiment_label": sentiment,
                        "review_text": review_text,
                        "review_date": date_text,
                        "verified_purchase": verified,
                        "helpful_votes": 0,
                    })
                except Exception:
                    continue

        except Exception as e:
            print(f"     ⚠️  Review error for {asin}: {e}")

        return reviews

    # ── Main runner ──────────────────────────────────────────────────────────

    async def run(self):
        if not PLAYWRIGHT_AVAILABLE:
            print("❌ Playwright not installed. Run: pip install playwright && playwright install chromium")
            return False

        os.makedirs(DATA_DIR, exist_ok=True)
        print("Starting Amazon India scraper\n")

        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-setuid-sandbox",
                      "--disable-blink-features=AutomationControlled"]
            )

            page = await self._setup_page(browser)

            # ── Scrape product listings ────────────────────────────────────
            for brand in BRANDS:
                print(f"\n Scraping products for: {brand}")
                brand_products = await self.scrape_search_page(page, brand)
                self.products.extend(brand_products)
                await asyncio.sleep(SCRAPE_DELAY)

            if not self.products:
                print("\n❌ No products scraped. Amazon may be blocking. Using mock data instead.")
                await browser.close()
                return False

            products_df = pd.DataFrame(self.products)
            products_df.to_csv(PRODUCTS_CSV, index=False)
            print(f"\n✅ Saved {len(products_df)} products → {PRODUCTS_CSV}")

            # ── Scrape reviews ────────────────────────────────────────────
            print("\n Scraping reviews")
            for _, product in products_df.iterrows():
                if not product["asin"]:
                    continue
                print(f"   Reviews for: {product['title'][:50]}")
                product_reviews = await self.scrape_reviews(
                    page, product["asin"], product["brand"], product["title"]
                )
                self.reviews.extend(product_reviews)
                await asyncio.sleep(SCRAPE_DELAY + random.uniform(1, 3))

            await browser.close()

        reviews_df = pd.DataFrame(self.reviews)
        reviews_df.to_csv(REVIEWS_CSV, index=False)
        print(f" Saved {len(reviews_df)} reviews → {REVIEWS_CSV}")
        return True


def run():
    success = asyncio.run(AmazonScraper().run())
    if not success:
        print("\n Falling back to mock data generator")
        from data.mock_data_generator import run as mock_run
        mock_run()


if __name__ == "__main__":
    run()
