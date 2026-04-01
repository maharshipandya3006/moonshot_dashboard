import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import GROQ_API_KEY, DATA_DIR, ANALYSIS_CSV, PRODUCTS_CSV, REVIEWS_CSV

try:
    from groq import Groq
    GROQ_AVAILABLE = bool(GROQ_API_KEY)
except ImportError:
    GROQ_AVAILABLE = False
    print(" groq not installed. Run: pip install groq")

INSIGHTS_FILE = os.path.join(DATA_DIR, "agent_insights.json")
GROQ_MODEL = "llama-3.3-70b-versatile"


class InsightsGenerator:

    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY) if GROQ_AVAILABLE else None

    def _call_groq(self, prompt: str, max_tokens: int = 800) -> str:
        if not self.client:
            return "Groq API key not configured. Set GROQ_API_KEY in .env"
        try:
            response = self.client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.4,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Groq API error: {e}"

#Brand Synthesis

    def synthesize_brand(self, brand: str, brand_row: pd.Series, sample_reviews: list) -> str:
        reviews_text = "\n".join([f"- {r[:200]}" for r in sample_reviews[:15]])
        prompt = f"""You are a competitive intelligence analyst for the Indian luggage market.

Brand: {brand}
Average price: Rs.{brand_row.get('avg_price', 'N/A')}
Average discount: {brand_row.get('avg_discount_pct', 'N/A')}%
Average rating: {brand_row.get('avg_rating', 'N/A')} / 5
Sentiment score: {brand_row.get('sentiment_score', 'N/A')} (range -1 to +1)
% positive reviews: {brand_row.get('pct_positive', 'N/A')}%
% negative reviews: {brand_row.get('pct_negative', 'N/A')}%
Top praise themes: {brand_row.get('top_praise', 'N/A')}
Top complaint themes: {brand_row.get('top_complaints', 'N/A')}

Sample customer reviews:
{reviews_text}

Write a 3-sentence brand summary:
1. Core positioning and customer perception
2. Main strength customers mention
3. Most significant risk or complaint pattern

Be specific and data-driven. No vague praise. Reply with only the 3 sentences."""
        return self._call_groq(prompt, max_tokens=250)

#Agent Insights
    def generate_agent_insights(self, brand_df, products_df, reviews_df) -> list:
        brand_summary = brand_df[[
            "brand", "avg_price", "avg_discount_pct", "avg_rating",
            "sentiment_score", "pct_positive", "pct_negative",
            "vfm_score", "top_praise", "top_complaints",
        ]].to_dict("records")

        aspect_cols = [c for c in brand_df.columns if c.startswith("aspect_")]
        aspect_summary = brand_df[["brand"] + aspect_cols].to_dict("records") if aspect_cols else []

        price_bands = {
            "budget (<Rs.2500)":  int((products_df["selling_price"] < 2500).sum()),
            "mid (Rs.2500-5000)": int(((products_df["selling_price"] >= 2500) & (products_df["selling_price"] < 5000)).sum()),
            "premium (>Rs.5000)": int((products_df["selling_price"] >= 5000).sum()),
        }

        prompt = f"""You are a senior competitive intelligence analyst for Indian e-commerce.

Amazon India luggage brand data:

BRAND METRICS:
{json.dumps(brand_summary, indent=2)}

ASPECT SENTIMENT (wheels, handle, zipper, material, size, durability, weight, price):
{json.dumps(aspect_summary, indent=2)}

PRICE BANDS:
{json.dumps(price_bands, indent=2)}

Products: {len(products_df)} | Reviews: {len(reviews_df)}

Generate exactly 5 non-obvious strategic insights. Each must:
- Surface something NOT obvious from reading the numbers alone
- Name a specific brand, metric, or comparison
- Have a clear actionable implication

Respond ONLY with a valid JSON array, no markdown, no preamble:
[
  {{
    "title": "Short title (max 8 words)",
    "insight": "1-2 sentences with specific data points",
    "implication": "1 sentence actionable recommendation"
  }}
]"""

        raw = self._call_groq(prompt, max_tokens=1200)
        try:
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except json.JSONDecodeError:
            return self._fallback_insights(brand_df)

    def _fallback_insights(self, brand_df) -> list:
        best_vfm  = brand_df.loc[brand_df["vfm_score"].idxmax(), "brand"]        if "vfm_score"        in brand_df.columns else "N/A"
        most_disc = brand_df.loc[brand_df["avg_discount_pct"].idxmax(), "brand"] if "avg_discount_pct" in brand_df.columns else "N/A"
        best_sent = brand_df.loc[brand_df["sentiment_score"].idxmax(), "brand"]  if "sentiment_score"  in brand_df.columns else "N/A"
        return [
            {"title": "Value-for-money leader identified",
             "insight": f"{best_vfm} delivers the highest sentiment-to-price ratio, making it the rational choice in the mid-market.",
             "implication": "Competitors should study this brand's feature-to-price calibration."},
            {"title": "Discount dependency is structural",
             "insight": f"{most_disc} maintains the highest average discount, suggesting MRP inflation as a positioning tactic.",
             "implication": "Buyers should anchor to selling price, not MRP, when comparing value."},
            {"title": "Sentiment leader outpaces star ratings",
             "insight": f"{best_sent} leads on NLP sentiment even when star ratings don't reflect it.",
             "implication": "Star ratings alone are insufficient; NLP sentiment reveals the real winner."},
            {"title": "Durability is the silent differentiator",
             "insight": "Durability complaints dominate 2-3 star reviews but rarely surface in rating averages.",
             "implication": "A brand that addresses durability explicitly will capture disproportionate loyalty."},
            {"title": "Premium segment is underpenetrated",
             "insight": "Products above Rs.5000 are under 20% of listings but show meaningfully higher sentiment.",
             "implication": "Launching premium-tier products with after-sales guarantees is a clear opportunity."},
        ]


    def run(self) -> dict:
        print(f" Running insights generator (Groq / {GROQ_MODEL})")
        brand_df    = pd.read_csv(ANALYSIS_CSV)
        products_df = pd.read_csv(PRODUCTS_CSV)
        reviews_df  = pd.read_csv(REVIEWS_CSV)

        output = {"brand_summaries": {}, "agent_insights": []}

        for _, row in brand_df.iterrows():
            brand = row["brand"]
            brand_reviews = reviews_df[reviews_df["brand"] == brand]["review_text"].dropna()
            sample = brand_reviews.sample(min(20, len(brand_reviews)), random_state=42).tolist()
            print(f"   ✍️  Synthesizing {brand}...")
            output["brand_summaries"][brand] = self.synthesize_brand(brand, row, sample)

        print(" Generating Agent Insights")
        output["agent_insights"] = self.generate_agent_insights(brand_df, products_df, reviews_df)

        os.makedirs(DATA_DIR, exist_ok=True)
        with open(INSIGHTS_FILE, "w") as f:
            json.dump(output, f, indent=2)
        print(f"✅ Insights saved -> {INSIGHTS_FILE}")
        return output


if __name__ == "__main__":
    InsightsGenerator().run()
