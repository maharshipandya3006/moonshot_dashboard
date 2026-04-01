
import os
import sys
import re
import pandas as pd
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import (
    PRODUCTS_CSV, REVIEWS_CSV, ANALYSIS_CSV, DATA_DIR, ASPECT_KEYWORDS
)

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("VaderSentiment not installed. Run: pip install vaderSentiment")


class SentimentAnalyzer:

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if VADER_AVAILABLE else None


    def score_review(self, text: str) -> dict:
        if not self.vader or not text:
            return {"compound": 0.0, "pos": 0.0, "neg": 0.0, "neu": 1.0}
        scores = self.vader.polarity_scores(str(text))
        return scores

    def label_from_compound(self, compound: float) -> str:
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        return "neutral"


    def extract_aspect_sentences(self, text: str) -> dict[str, list[str]]:
        """Split text into sentences and tag each sentence with relevant aspects."""
        sentences = re.split(r"[.!?]+", str(text))
        aspect_sentences = defaultdict(list)
        for sentence in sentences:
            sentence_lower = sentence.lower()
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(kw in sentence_lower for kw in keywords):
                    aspect_sentences[aspect].append(sentence.strip())
        return dict(aspect_sentences)

    def score_aspects(self, text: str) -> dict[str, float]:
        """Return compound sentiment score per aspect for a review text."""
        aspect_sentences = self.extract_aspect_sentences(text)
        aspect_scores = {}
        for aspect, sentences in aspect_sentences.items():
            if not sentences:
                continue
            scores = [self.score_review(s)["compound"] for s in sentences]
            aspect_scores[aspect] = round(np.mean(scores), 3)
        return aspect_scores

#Theme Extraction

    PRAISE_INDICATORS = [
        "great", "excellent", "good", "love", "amazing", "perfect", "best",
        "fantastic", "wonderful", "awesome", "happy", "satisfied", "recommend",
        "quality", "durable", "sturdy", "smooth", "lightweight", "spacious",
    ]
    COMPLAINT_INDICATORS = [
        "bad", "poor", "worst", "broke", "broken", "issue", "problem", "defect",
        "disappoint", "return", "refund", "damaged", "crack", "cheap", "flimsy",
        "regret", "waste", "useless", "terrible", "horrible", "pathetic",
    ]

    def extract_themes(self, reviews: list[str], top_n: int = 6) -> dict:
        """
        Extract recurring praise and complaint themes from a list of reviews.
        Returns dict with 'praise' and 'complaints' as lists of (theme, count).
        """
        praise_counts = Counter()
        complaint_counts = Counter()

        for review in reviews:
            text_lower = str(review).lower()
            words = re.findall(r"\b\w+\b", text_lower)
            word_set = set(words)

            # Check if review is positive or negative in tone
            if self.vader:
                compound = self.score_review(review)["compound"]
                is_positive = compound > 0.05
                is_negative = compound < -0.05
            else:
                is_positive = any(w in word_set for w in self.PRAISE_INDICATORS)
                is_negative = any(w in word_set for w in self.COMPLAINT_INDICATORS)

            # Aspect level phrase tagging
            for aspect, keywords in ASPECT_KEYWORDS.items():
                if any(kw in text_lower for kw in keywords):
                    if is_positive:
                        praise_counts[aspect] += 1
                    elif is_negative:
                        complaint_counts[aspect] += 1

            # General praise/complaint words
            for word in self.PRAISE_INDICATORS:
                if word in word_set and is_positive:
                    praise_counts[word] += 1
            for word in self.COMPLAINT_INDICATORS:
                if word in word_set and is_negative:
                    complaint_counts[word] += 1

        return {
            "praise": praise_counts.most_common(top_n),
            "complaints": complaint_counts.most_common(top_n),
        }

    #Anomaly detection

    def detect_anomalies(self, products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> list[dict]:
        """
        Surface non-obvious patterns like:
        - High rating but high complaint frequency about specific aspects
        - Heavy discounting without sentiment improvement
        - Brand with high volume but low sentiment
        """
        anomalies = []

        brand_ratings = products_df.groupby("brand")["rating"].mean()
        brand_sentiments = reviews_df.groupby("brand")["compound_score"].mean()

        # Anomaly 1: High star rating but low VADER sentiment
        for brand in brand_ratings.index:
            if brand not in brand_sentiments.index:
                continue
            star_norm = (brand_ratings[brand] - 1) / 4  # 0–1
            vader_norm = (brand_sentiments[brand] + 1) / 2  # 0–1
            if star_norm > 0.7 and vader_norm < 0.5:
                anomalies.append({
                    "brand": brand,
                    "type": "rating_sentiment_gap",
                    "description": f"{brand} has high star ratings ({brand_ratings[brand]:.1f}★) "
                                   f"but reviews reveal subdued sentiment — possible rating inflation.",
                })

        # Anomaly 2: Brands with >45% discount but sentiment not above average
        brand_discounts = products_df.groupby("brand")["discount_pct"].mean()
        avg_sentiment = brand_sentiments.mean()
        for brand in brand_discounts.index:
            if brand not in brand_sentiments.index:
                continue
            if brand_discounts[brand] > 45 and brand_sentiments[brand] < avg_sentiment:
                anomalies.append({
                    "brand": brand,
                    "type": "discount_dependency",
                    "description": f"{brand} relies on {brand_discounts[brand]:.0f}% average discount "
                                   f"but still underperforms on customer sentiment — discounting is structural.",
                })

        # Anomaly 3: Durability complaints in high-rated products
        durability_keywords = ASPECT_KEYWORDS["durability"]
        for brand in reviews_df["brand"].unique():
            brand_reviews = reviews_df[reviews_df["brand"] == brand]
            neg_reviews = brand_reviews[brand_reviews["compound_score"] < -0.05]
            dura_complaints = neg_reviews["review_text"].apply(
                lambda t: any(kw in str(t).lower() for kw in durability_keywords)
            ).sum()
            dura_pct = dura_complaints / len(brand_reviews) * 100 if len(brand_reviews) > 0 else 0
            avg_rating = brand_ratings.get(brand, 0)
            if dura_pct > 15 and avg_rating >= 4.0:
                anomalies.append({
                    "brand": brand,
                    "type": "hidden_durability_issue",
                    "description": f"{brand} shows {dura_pct:.0f}% of reviews contain durability complaints "
                                   f"despite a {avg_rating:.1f}★ average — a hidden quality risk.",
                })

        return anomalies

    # Review trust signals

    def detect_trust_issues(self, reviews_df: pd.DataFrame) -> dict[str, list]:
        """
        Flag potential fake/suspicious review patterns per brand.
        """
        flags = defaultdict(list)
        for brand, group in reviews_df.groupby("brand"):
            # 1. Suspiciously high 5-star ratio
            five_star_pct = (group["stars"] == 5).mean() * 100
            if five_star_pct > 65:
                flags[brand].append(f" {five_star_pct:.0f}% five-star reviews (possible padding)")

            # 2. Very short reviews dominating
            avg_len = group["review_text"].apply(lambda x: len(str(x).split())).mean()
            if avg_len < 8:
                flags[brand].append(f" Avg review length only {avg_len:.0f} words (low-quality reviews)")

            # 3. Low verified purchase ratio
            if "verified_purchase" in group.columns:
                verified_pct = group["verified_purchase"].mean() * 100
                if verified_pct < 60:
                    flags[brand].append(f" Only {verified_pct:.0f}% verified purchases")

        return dict(flags)

    # Pipeline

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print(" Loading data.")
        products_df = pd.read_csv(PRODUCTS_CSV)
        reviews_df = pd.read_csv(REVIEWS_CSV)

        # Step 1: Score every review
        print("Scoring reviews with VADER")
        vader_scores = reviews_df["review_text"].apply(
            lambda t: pd.Series(self.score_review(t))
        )
        reviews_df = pd.concat([reviews_df, vader_scores], axis=1)
        reviews_df.rename(columns={"compound": "compound_score"}, inplace=True)
        reviews_df["vader_label"] = reviews_df["compound_score"].apply(self.label_from_compound)

        # Step 2: Aspect scores per review
        print("Extracting aspect-level sentiments")
        aspect_data = reviews_df["review_text"].apply(
            lambda t: pd.Series(self.score_aspects(t))
        )
        for aspect in ASPECT_KEYWORDS.keys():
            if aspect in aspect_data.columns:
                reviews_df[f"aspect_{aspect}"] = aspect_data[aspect]
            else:
                reviews_df[f"aspect_{aspect}"] = np.nan

        # Step 3: Brand-level aggregation
        print("Aggregating brand-level analysis")
        brand_records = []
        for brand in products_df["brand"].unique():
            b_products = products_df[products_df["brand"] == brand]
            b_reviews = reviews_df[reviews_df["brand"] == brand]

            themes = self.extract_themes(b_reviews["review_text"].tolist())
            top_praise = [t for t, _ in themes["praise"][:5]]
            top_complaints = [t for t, _ in themes["complaints"][:5]]

            # Aspect averages
            aspect_avgs = {}
            for aspect in ASPECT_KEYWORDS.keys():
                col = f"aspect_{aspect}"
                if col in b_reviews.columns:
                    val = b_reviews[col].dropna()
                    aspect_avgs[f"aspect_{aspect}_avg"] = round(val.mean(), 3) if len(val) > 0 else np.nan

            record = {
                "brand": brand,
                "avg_price": round(b_products["selling_price"].mean(), 0),
                "avg_mrp": round(b_products["mrp"].mean(), 0),
                "avg_discount_pct": round(b_products["discount_pct"].mean(), 1),
                "min_price": b_products["selling_price"].min(),
                "max_price": b_products["selling_price"].max(),
                "avg_rating": round(b_products["rating"].mean(), 2),
                "total_review_count": int(b_products["review_count"].sum()),
                "avg_review_count": round(b_products["review_count"].mean(), 0),
                "product_count": len(b_products),
                "sentiment_score": round(b_reviews["compound_score"].mean(), 3),
                "pct_positive": round((b_reviews["vader_label"] == "positive").mean() * 100, 1),
                "pct_negative": round((b_reviews["vader_label"] == "negative").mean() * 100, 1),
                "pct_neutral": round((b_reviews["vader_label"] == "neutral").mean() * 100, 1),
                "top_praise": " | ".join(top_praise),
                "top_complaints": " | ".join(top_complaints),
                "segment": b_products["segment"].mode()[0] if len(b_products) > 0 else "unknown",
                **aspect_avgs,
            }
            brand_records.append(record)

        brand_df = pd.DataFrame(brand_records)

        # Step 4: Value-for-money score (sentiment / normalized price)
        max_price = brand_df["avg_price"].max()
        brand_df["price_norm"] = brand_df["avg_price"] / max_price
        brand_df["vfm_score"] = round(
            (brand_df["sentiment_score"] + 1) / 2 / brand_df["price_norm"], 3
        )

        # Save outputs
        os.makedirs(DATA_DIR, exist_ok=True)
        reviews_df.to_csv(REVIEWS_CSV, index=False)
        brand_df.to_csv(ANALYSIS_CSV, index=False)

        print(f"Analysis complete.")
        print(f" Reviews scored: {len(reviews_df)}")
        print(f" Brands analyzed: {len(brand_df)}")

        anomalies = self.detect_anomalies(products_df, reviews_df)
        trust_flags = self.detect_trust_issues(reviews_df)

        return brand_df, reviews_df, products_df


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.run()
