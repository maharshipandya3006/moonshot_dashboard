import os
from dotenv import load_dotenv

load_dotenv()

BRANDS = [
    "Safari",
    "Skybags",
    "American Tourister",
    "VIP",
    "Aristocrat",
    "Nasher Miles",
]

BRAND_COLORS = {
    "Safari": "#E63946",
    "Skybags": "#457B9D",
    "American Tourister": "#F4A261",
    "VIP": "#2A9D8F",
    "Aristocrat": "#8338EC",
    "Nasher Miles": "#FB5607",
}
PRODUCTS_PER_BRAND = 12
REVIEWS_PER_PRODUCT = 60

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "output")
PRODUCTS_CSV = os.path.join(DATA_DIR, "products.csv")
REVIEWS_CSV = os.path.join(DATA_DIR, "reviews.csv")
ANALYSIS_CSV = os.path.join(DATA_DIR, "brand_analysis.csv")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
USE_MOCK_DATA = os.getenv("USE_MOCK_DATA", "true").lower() == "true"

SCRAPE_DELAY = float(os.getenv("SCRAPE_DELAY_SECONDS", "3"))
AMAZON_BASE_URL = "https://www.amazon.in"

ASPECT_KEYWORDS = {
    "wheels": ["wheel", "wheels", "rolling", "rolls", "spinner", "rotation", "rotate"],
    "handle": ["handle", "handles", "grip", "telescopic", "extendable", "retract"],
    "zipper": ["zipper", "zippers", "zip", "zips", "lock", "locking", "closure"],
    "material": ["material", "fabric", "hard shell", "hardside", "polycarbonate",
                 "abs", "nylon", "canvas", "shell", "hard case"],
    "size": ["size", "capacity", "spacious", "roomy", "fits", "large", "small",
             "cabin", "check-in", "40l", "55l", "75l"],
    "durability": ["durable", "durability", "strong", "sturdy", "broke", "broken",
                   "crack", "cracked", "damage", "damaged", "quality"],
    "weight": ["weight", "lightweight", "heavy", "light", "weighs", "kg"],
    "price": ["price", "value", "worth", "money", "expensive", "cheap", "affordable",
              "cost", "budget"],
}
