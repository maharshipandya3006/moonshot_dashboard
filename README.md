# 🧳 Moonshot Competitive Intelligence Dashboard

> Amazon India Luggage Brand Analysis

The Moonshot Dashboard is a full stack intelligence platform designed to help analyze and visualize product and brand performance on Amazon India. It combines data scraping, sentiment analysis, and AI-generated insights to provide actionable intelligence for decision-making.

## What This Does

An end-to-end intelligence pipeline that:
1. **Scrapes** product listings + reviews from Amazon India (with mock fallback)
2. **Analyzes** sentiment at review, brand, and aspect level (wheels, zippers, handles, etc.)
3. **Generates** LLM-powered insights via GROQ API
4. **Displays** everything in an interactive Streamlit dashboard

---

# Deployed Project
https://moonshotdashboard-maharshi3006.streamlit.app/

<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/066b28d5-bb0b-4500-aff7-df56af13bae0" />
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/b6d1fba2-86a4-4168-8078-1e3b00cb665f" />
<img width="1470" height="956" alt="Screenshot 2026-04-02 at 12 18 42 AM" src="https://github.com/user-attachments/assets/4d93eb7c-d15c-476c-812e-9d11fd6d0e4e" />
<img width="1470" height="956" alt="Screenshot 2026-04-02 at 12 18 56 AM" src="https://github.com/user-attachments/assets/148b6b39-0672-4d79-b5a3-6bdfc601f2ec" />
<img width="2940" height="1912" alt="image" src="https://github.com/user-attachments/assets/9aafd442-3408-4542-8329-bd16c967a936" />
<img width="1470" height="956" alt="Screenshot 2026-04-02 at 12 19 44 AM" src="https://github.com/user-attachments/assets/cb93dfe7-78bb-4384-85af-446edb60b656" />

## Architecture 

```
moonshot-assignment/
├── config.py                     # Central config (brands, paths, API keys)
├── run_pipeline.py               # Orchestrates all 3 pipeline steps
├── requirements.txt
│
├── data/
│   ├── mock_data_generator.py    # Generates realistic mock data (no scraping needed)
│   └── output/                   # Generated CSVs live here
│       ├── products.csv
│       ├── reviews.csv
│       ├── brand_analysis.csv
│       └── agent_insights.json
│
├── scraper/
│   └── amazon_scraper.py         # Playwright-based live scraper 
│
├── analysis/
│   ├── sentiment_analyzer.py     # VADER + aspect-level + anomaly detection
│   └── insights_generator.py     # GROQ API for brand summaries + Agent Insights
│
└── dashboard/
    └── app.py                    # Streamlit dashboard (5 tabs)
```
## Tech Stack

| Component | Technology |
|---|---|
| Scraping | Playwright (async), BeautifulSoup |
| Data processing | Pandas, NumPy |
| Sentiment | VADER Sentiment, Groq Cloud (llama-3.3-70b-versatile) |
| Dashboard | Streamlit |
| Charts | Plotly |
| Storage | CSV (portable, no DB needed) |

---

---

## Step-by-Step Setup

### Step 1 — Clone / create the project

```bash
git clone https://github.com/<maharshipandya3006>/moonshot_dashboard.git
cd moonshot_dashboard
```

### Step 2 — Create a Python virtual environment

```bash
python -m venv venv

# On Mac/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Install Playwright browsers

```bash
playwright install chromium
```

### Step 5 — Set up environment variables

```bash
cp .env
# Open .env and add your Groq API key:
# Get a free key at https://console.groq.com
# GROQ_API_KEY=gsk_...
# USE_MOCK_DATA=true   (set to false if you want live scraping)
```

### Step 6 — Run the pipeline

**Option A: Mock data (recommended to start)**
```bash
python run_pipeline.py
```

**Option B: Try live Amazon scraping first**
```bash
python run_pipeline.py --scrape
```

**Option C: Skip LLM insights (no API key)**
```bash
python run_pipeline.py --skip-insights
```

### Step 7 — Launch the dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser. 🎉

---

## Dashboard Tabs

| Tab | What's in it |
|-----|-------------|
| 📊 Overview | KPI cards, sentiment bar chart, price vs rating bubble, price band distribution |
| 🏆 Brand Comparison | Radar chart, discount comparison, full sortable table, praise/complaint themes, review distribution |
| 🔍 Product Drilldown | Per-product expandable cards, star distribution, sample reviews with sentiment tags |
| 🎯 Aspect Analysis | Heatmap of sentiment by aspect (wheels/zippers/etc.), value-for-money ranking |
| 🤖 Agent Insights | LLM-generated strategic conclusions, anomaly detection, trust signal flags |

---

## Sentiment Methodology

1. **VADER (Valence Aware Dictionary and sEntiment Reasoner)**: Applied to every review text. Produces compound score (-1 to +1).
2. **Aspect extraction**: Review text is split into sentences; each sentence is matched against keyword lists for 8 product aspects (wheels, handle, zipper, material, size, durability, weight, price). VADER is then applied per-sentence for that aspect.
3. **Brand aggregation**: Mean compound score across all reviews per brand. Percent positive/negative/neutral tallied.
4. **Value-for-money score**: `(sentiment_score + 1) / 2` normalized by `avg_price / max_price`. Higher = better sentiment relative to price.
5. **LLM synthesis**: GROQ API summarizes review samples and generates non-obvious strategic insights.

---

## Anomaly Detection Logic

| Anomaly Type | Detection Method |
|---|---|
| Rating–sentiment gap | Star rating > 3.5 but VADER sentiment < 0 |
| Discount dependency | Avg discount > 45% AND sentiment below mean |
| Hidden durability issues | Durability keywords in >15% of negative reviews despite high rating |
| Review trust signals | >65% five-star ratio, avg review length < 8 words, or <60% verified purchases |

---

## Dataset

- 6 brands: Safari, Skybags, American Tourister, VIP, Aristocrat, Nasher Miles
- 12 products per brand (72 total)
- 60 reviews per product (4,320 total)
- All data in `/data/output/*.csv`

---

## Limitations

1. **Mock data**: The mock generator uses research-based brand profiles but is synthetic. Replace with live scraper output for real analysis.
2. **Amazon blocking**: Amazon India actively detects and blocks automated scraping. Add proxies and longer delays for production use.
3. **VADER limitations**: VADER is lexicon-based and may misclassify Hindi–English (Hinglish) reviews common on Amazon India. A fine-tuned multilingual model would improve accuracy.
4. **Review volume**: Real review volumes on Amazon India can reach 50,000+ for popular products; this analysis uses a 60-review sample per product.

---

## Future Improvements

- [ ] Proxy rotation for reliable live scraping
- [ ] Multilingual sentiment model for Hinglish reviews
- [ ] Time-series trend analysis (sentiment over months)
- [ ] Seller-level analysis (3P vs 1P)
- [ ] Export to PDF report button
- [ ] Real-time price tracking alerts

---


The pipeline works **without an API key** (mock data + VADER sentiment). Add `GROQ_API_KEY` (free at [console.groq.com](https://console.groq.com)) to unlock AI-generated brand summaries and Agent Insights.
