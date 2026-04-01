
import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    BRANDS, PRODUCTS_PER_BRAND, REVIEWS_PER_PRODUCT,
    DATA_DIR, PRODUCTS_CSV, REVIEWS_CSV
)

random.seed(42)
np.random.seed(42)

# Brand profiles (researched from real Amazon India data)
BRAND_PROFILES = {
    "Safari": {
        "price_range": (2500, 7500),
        "avg_discount": 38,
        "base_rating": 4.1,
        "review_volume": "high",
        "segment": "mid-range",
        "strengths": ["durable", "spacious", "good wheels", "value for money",
                      "strong zippers", "lightweight"],
        "weaknesses": ["average looks", "basic design", "handle issues after extended use"],
        "product_lines": ["Thorium", "Moonbeam", "Polaris", "Zenith", "Superlite",
                          "Fabric Series", "Optima", "Prestige", "Glide", "Trek",
                          "Flash", "Scope"],
        "sizes": ["Cabin (55 cm)", "Medium (65 cm)", "Large (75 cm)", "Set of 3"],
    },
    "Skybags": {
        "price_range": (1800, 6000),
        "avg_discount": 45,
        "base_rating": 3.9,
        "review_volume": "very_high",
        "segment": "value",
        "strengths": ["trendy design", "affordable", "colorful", "lightweight",
                      "good for students"],
        "weaknesses": ["zipper quality", "wheel durability", "handle wobbles",
                       "material thin", "not for heavy use"],
        "product_lines": ["Swag", "Strio", "Backpacker", "Neon", "Fab", "Lazer",
                          "Milford", "Reverb", "Centuple", "Hexa", "Cosmo", "Rapid"],
        "sizes": ["Cabin (55 cm)", "Medium (65 cm)", "Large (75 cm)", "XL (80 cm)"],
    },
    "American Tourister": {
        "price_range": (3500, 12000),
        "avg_discount": 30,
        "base_rating": 4.3,
        "review_volume": "high",
        "segment": "premium-mid",
        "strengths": ["brand trust", "excellent quality", "smooth wheels",
                      "TSA lock", "great warranty", "sturdy build"],
        "weaknesses": ["slightly heavy", "premium pricing", "limited colors"],
        "product_lines": ["Linex", "Starvibe", "Spinner", "Wavetw", "Seaventure",
                          "Beau", "Curio", "Trigard", "Moonbeam", "Alltrail",
                          "Backfire", "Fieldwork"],
        "sizes": ["Cabin (55 cm)", "Medium (68 cm)", "Large (77 cm)", "Set of 2"],
    },
    "VIP": {
        "price_range": (2000, 8000),
        "avg_discount": 40,
        "base_rating": 4.0,
        "review_volume": "medium",
        "segment": "mid-range",
        "strengths": ["classic brand", "reliable", "good capacity",
                      "sturdy frame", "decent quality"],
        "weaknesses": ["outdated design", "heavier than competitors",
                       "customer service issues", "limited warranty support"],
        "product_lines": ["Odyssey", "Vectra", "Vantage", "Elanza", "Spectra",
                          "Rochester", "Turbo", "Sapphire", "Premier", "Classic",
                          "Magnus", "Crest"],
        "sizes": ["Cabin (55 cm)", "Medium (65 cm)", "Large (75 cm)", "XL (85 cm)"],
    },
    "Aristocrat": {
        "price_range": (1500, 5000),
        "avg_discount": 50,
        "base_rating": 3.7,
        "review_volume": "medium",
        "segment": "budget",
        "strengths": ["very affordable", "decent for occasional travel",
                      "lightweight", "easy on budget"],
        "weaknesses": ["durability concerns", "wheels break easily",
                       "zipper quality poor", "material not premium",
                       "handle issues", "not for frequent flyers"],
        "product_lines": ["Opal", "Crystal", "Poise", "Elan", "Sprint",
                          "Glide", "Ultra", "Style", "Edge", "Smart",
                          "Pro", "Lite"],
        "sizes": ["Cabin (55 cm)", "Medium (65 cm)", "Large (75 cm)"],
    },
    "Nasher Miles": {
        "price_range": (3000, 10000),
        "avg_discount": 35,
        "base_rating": 4.2,
        "review_volume": "medium",
        "segment": "premium-mid",
        "strengths": ["premium look", "excellent wheels", "TSA lock",
                      "hard shell", "smooth spinner", "modern design",
                      "great for frequent travelers"],
        "weaknesses": ["higher price point", "limited availability",
                       "fewer review samples", "less brand recognition"],
        "product_lines": ["Derby", "Bruges", "Liverpool", "Milan", "Almaty",
                          "Boston", "London", "Dubai", "Colombo", "Bali",
                          "Geneva", "Tokyo"],
        "sizes": ["Cabin (55 cm)", "Medium (65 cm)", "Large (75 cm)", "Set of 3"],
    },
}

# Review templates
POSITIVE_TEMPLATES = [
    "Really happy with this {brand} luggage. The {strength1} is impressive and {strength2}.",
    "Great buy! {strength1} and {strength2}. Would recommend to friends and family.",
    "Bought this for my trip to {city}. Excellent {strength1}. The {strength2} is a big plus.",
    "Superb quality from {brand}. {strength1} is exactly what I expected. {strength2} too.",
    "Value for money! Got this at {discount}% discount. {strength1} is great.",
    "Using this for {months} months now. Still {strength1}. {strength2} holds up well.",
    "Perfect for {travel_type} travel. {strength1} makes it easy to carry. {strength2} is solid.",
    "{strength1} is top notch. Very satisfied with the purchase. {strength2} also good.",
    "Excellent product. {strength1} and {strength2}. Packaging was also very good.",
    "Highly recommend. {strength1}. The design looks premium. {strength2} as well.",
]

NEGATIVE_TEMPLATES = [
    "Disappointed with {brand}. {weakness1} is a major issue. {weakness2} too.",
    "Not worth the price. {weakness1}. Regret buying this. {weakness2} also.",
    "The {weakness1} started after just {months} months. {weakness2} as well. Not durable.",
    "Returned this product. {weakness1} is unacceptable. {weakness2} was also bad.",
    "Looks good in photos but {weakness1} in reality. {weakness2} too. Misleading.",
    "Bought for {travel_type} travel. {weakness1}. Was expecting better from {brand}.",
    "Quality issues from day one. {weakness1}. {weakness2}. Would not buy again.",
    "Average product. {weakness1}. For this price {weakness2}. Not recommended.",
    "Had high hopes but {weakness1}. The {weakness2} is disappointing.",
    "My {months} year old bag was better. {weakness1}. {weakness2}.",
]

NEUTRAL_TEMPLATES = [
    "Decent product overall. {strength1} is good but {weakness1} could be better.",
    "Average experience. {strength1} is fine. However, {weakness1}.",
    "OK product. {strength1} but {weakness1}. Works for occasional travel.",
    "Mixed feelings. {strength1} is impressive. But {weakness1} is concerning.",
    "Gets the job done. {strength1}. Just watch out for {weakness1}.",
    "Good for the price. {strength1}. But {weakness1} over time.",
]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune",
          "Kolkata", "Ahmedabad", "Jaipur", "Goa", "Kerala", "Manali"]
TRAVEL_TYPES = ["business", "leisure", "frequent", "international", "domestic",
                "weekend", "family", "solo"]


def _pick_review_text(brand: str, sentiment: str) -> str:
    profile = BRAND_PROFILES[brand]
    strengths = profile["strengths"]
    weaknesses = profile["weaknesses"]

    s1 = random.choice(strengths)
    s2 = random.choice([x for x in strengths if x != s1])
    w1 = random.choice(weaknesses)
    w2 = random.choice([x for x in weaknesses if x != w1] or weaknesses)

    if sentiment == "positive":
        tmpl = random.choice(POSITIVE_TEMPLATES)
    elif sentiment == "negative":
        tmpl = random.choice(NEGATIVE_TEMPLATES)
    else:
        tmpl = random.choice(NEUTRAL_TEMPLATES)

    return tmpl.format(
        brand=brand,
        strength1=s1,
        strength2=s2,
        weakness1=w1,
        weakness2=w2,
        city=random.choice(CITIES),
        discount=random.randint(25, 55),
        months=random.randint(1, 18),
        travel_type=random.choice(TRAVEL_TYPES),
    )


def _star_to_sentiment(star: int) -> str:
    if star >= 4:
        return "positive"
    elif star == 3:
        return "neutral"
    else:
        return "negative"


def generate_products() -> pd.DataFrame:
    records = []
    for brand in BRANDS:
        profile = BRAND_PROFILES[brand]
        lo, hi = profile["price_range"]
        lines = profile["product_lines"]
        sizes = profile["sizes"]

        for i in range(PRODUCTS_PER_BRAND):
            line = lines[i % len(lines)]
            size = sizes[i % len(sizes)]
            mrp = random.randint(lo, hi)
            discount_pct = max(10, int(np.random.normal(profile["avg_discount"], 8)))
            discount_pct = min(discount_pct, 70)
            selling_price = int(mrp * (1 - discount_pct / 100))

            base_r = profile["base_rating"]
            rating = round(np.clip(np.random.normal(base_r, 0.25), 1.0, 5.0), 1)

            vol_map = {"very_high": (800, 5000), "high": (300, 2000),
                       "medium": (100, 800), "low": (20, 200)}
            review_count = random.randint(*vol_map[profile["review_volume"]])

            records.append({
                "asin": f"B{random.randint(10**8, 10**9-1)}",
                "brand": brand,
                "title": f"{brand} {line} {size} Luggage",
                "product_line": line,
                "size": size,
                "mrp": mrp,
                "selling_price": selling_price,
                "discount_pct": discount_pct,
                "rating": rating,
                "review_count": review_count,
                "segment": profile["segment"],
                "url": f"https://www.amazon.in/dp/B{random.randint(10**8, 10**9-1)}",
            })

    return pd.DataFrame(records)


def generate_reviews(products_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, product in products_df.iterrows():
        brand = product["brand"]
        profile = BRAND_PROFILES[brand]
        base_r = profile["base_rating"]

        # Rating distribution shaped around base_rating
        stars_pool = []
        for _ in range(REVIEWS_PER_PRODUCT):
            s = int(np.clip(np.round(np.random.normal(base_r, 0.7)), 1, 5))
            stars_pool.append(s)

        for idx, stars in enumerate(stars_pool):
            sentiment = _star_to_sentiment(stars)
            text = _pick_review_text(brand, sentiment)
            date = datetime.now() - timedelta(days=random.randint(1, 730))

            records.append({
                "review_id": f"R{random.randint(10**9, 10**10-1)}",
                "asin": product["asin"],
                "brand": brand,
                "product_title": product["title"],
                "stars": stars,
                "sentiment_label": sentiment,
                "review_text": text,
                "review_date": date.strftime("%Y-%m-%d"),
                "verified_purchase": random.random() > 0.15,
                "helpful_votes": random.randint(0, 50) if random.random() > 0.7 else 0,
            })

    return pd.DataFrame(records)


def run():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("Generating mock product data")
    products = generate_products()
    products.to_csv(PRODUCTS_CSV, index=False)
    print(f"   {len(products)} products saved → {PRODUCTS_CSV}")

    print(" Generating mock review data")
    reviews = generate_reviews(products)
    reviews.to_csv(REVIEWS_CSV, index=False)
    print(f" {len(reviews)} reviews saved → {REVIEWS_CSV}")

    print("\n Quick summary:")
    print(products.groupby("brand")[["selling_price", "discount_pct", "rating"]].mean().round(2))


if __name__ == "__main__":
    run()
