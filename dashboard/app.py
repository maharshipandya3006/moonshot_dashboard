
import os
import sys
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import (
    BRAND_COLORS, ANALYSIS_CSV, PRODUCTS_CSV, REVIEWS_CSV,
    DATA_DIR, ASPECT_KEYWORDS
)

INSIGHTS_FILE = os.path.join(DATA_DIR, "agent_insights.json")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Moonshot | Luggage Intelligence",
    page_icon="🧳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1c2333, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 12px !important; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; font-size: 28px !important; font-weight: 700; }
    
    /* Headers */
    h1, h2, h3 { color: #e6edf3 !important; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 8px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { background: transparent; color: #8b949e; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #21262d !important; color: #58a6ff !important; }
    
    /* Insight card */
    .insight-card {
        background: linear-gradient(135deg, #1c2333, #161b22);
        border: 1px solid #30363d;
        border-left: 3px solid #58a6ff;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 8px 0;
    }
    .insight-title { color: #58a6ff; font-weight: 700; font-size: 15px; margin-bottom: 8px; }
    .insight-text { color: #c9d1d9; font-size: 14px; line-height: 1.6; }
    .insight-implication { color: #3fb950; font-size: 13px; margin-top: 8px; }
    
    /* Warning card */
    .anomaly-card {
        background: #1c1a14;
        border: 1px solid #9e6a03;
        border-left: 3px solid #e3b341;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        color: #c9d1d9;
        font-size: 13px;
    }
    
    /* Badge */
    .badge-premium { background: #58a6ff22; color: #58a6ff; padding: 2px 8px; border-radius: 20px; font-size: 12px; }
    .badge-value { background: #3fb95022; color: #3fb950; padding: 2px 8px; border-radius: 20px; font-size: 12px; }
    .badge-budget { background: #e3b34122; color: #e3b341; padding: 2px 8px; border-radius: 20px; font-size: 12px; }

    /* Table */
    .dataframe { background: #161b22 !important; color: #e6edf3 !important; }
    
    /* Divider */
    hr { border-color: #30363d; }
    
    /* Selectbox */
    .stSelectbox label, .stMultiSelect label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    try:
        brand_df = pd.read_csv(ANALYSIS_CSV)
        products_df = pd.read_csv(PRODUCTS_CSV)
        reviews_df = pd.read_csv(REVIEWS_CSV)
    except FileNotFoundError:
        st.error("Data not found. Run `python run_pipeline.py` first.")
        st.stop()

    insights = {"brand_summaries": {}, "agent_insights": []}
    if os.path.exists(INSIGHTS_FILE):
        with open(INSIGHTS_FILE) as f:
            insights = json.load(f)

    return brand_df, products_df, reviews_df, insights


brand_df, products_df, reviews_df, insights = load_data()

for col in ["avg_price", "avg_discount_pct", "avg_rating", "sentiment_score",
            "pct_positive", "pct_negative", "vfm_score", "total_review_count"]:
    if col in brand_df.columns:
        brand_df[col] = pd.to_numeric(brand_df[col], errors="coerce")

ALL_BRANDS = sorted(brand_df["brand"].tolist())


# ── Helper functions

def brand_color(brand: str) -> str:
    return BRAND_COLORS.get(brand, "#888888")


def sentiment_color(score: float) -> str:
    if score >= 0.2:
        return "#3fb950"
    elif score >= 0.0:
        return "#e3b341"
    return "#f85149"


def make_radar_chart(brands_selected: list) -> go.Figure:
    """Radar / spider chart for multi-brand comparison."""
    df = brand_df[brand_df["brand"].isin(brands_selected)].copy()
    
    # Normalize metrics 0-1
    metrics = {
        "Sentiment": "sentiment_score",
        "Rating": "avg_rating",
        "Value/Money": "vfm_score",
        "Review Vol.": "total_review_count",
        "Discount": "avg_discount_pct",
    }

    def hex_to_rgba(hex_color, alpha):
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    fig = go.Figure()
    for _, row in df.iterrows():
        brand = row["brand"]
        values = []
        for label, col in metrics.items():
            col_max = brand_df[col].max()
            col_min = brand_df[col].min()
            val = (row[col] - col_min) / (col_max - col_min + 1e-9)
            values.append(round(val, 3))
        values.append(values[0])  # close polygon
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=list(metrics.keys()) + [list(metrics.keys())[0]],
            fill="toself",
            name=brand,
            line_color=brand_color(brand),
            fillcolor=hex_to_rgba(brand_color(brand), 0.2)
        ))


    fig.update_layout(
        polar=dict(
            bgcolor="#161b22",
            radialaxis=dict(visible=True, range=[0, 1], color="#8b949e", gridcolor="#30363d"),
            angularaxis=dict(color="#8b949e", gridcolor="#30363d"),
        ),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#0f1117",
        font_color="#e6edf3",
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


def plot_theme_horizontal(themes: str, color: str, title: str) -> go.Figure:
    if not themes or pd.isna(themes):
        return go.Figure()
    items = [t.strip() for t in str(themes).split("|") if t.strip()]
    counts = list(range(len(items), 0, -1))
    fig = go.Figure(go.Bar(
        x=counts, y=items,
        orientation="h",
        marker_color=color,
        text=items, textposition="inside",
    ))
    fig.update_layout(
        title=dict(text=title, font_color="#e6edf3", font_size=13),
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=8, r=8, t=32, b=8),
        height=160,
    )
    return fig


# SIDEBAR

with st.sidebar:
    st.markdown("### 🧳 Moonshot Intelligence")
    st.markdown("*Amazon India Luggage Market*")
    st.divider()

    st.markdown("**🔍 Filters**")
    selected_brands = st.multiselect(
        "Brands", ALL_BRANDS, default=ALL_BRANDS, key="brand_filter"
    )

    price_min = int(products_df["selling_price"].min())
    price_max = int(products_df["selling_price"].max())
    price_range = st.slider(
        "Price Range (₹)", price_min, price_max, (price_min, price_max), step=100
    )

    min_rating = st.slider("Min Rating ★", 1.0, 5.0, 1.0, step=0.1)
    min_sentiment = st.slider("Min Sentiment Score", -1.0, 1.0, -1.0, step=0.05)

    st.divider()
    st.markdown("**📦 Scope**")
    st.metric("Brands", len(brand_df))
    st.metric("Products", len(products_df))
    st.metric("Reviews", len(reviews_df))


# Apply filters
filtered_brands_df = brand_df[
    (brand_df["brand"].isin(selected_brands)) &
    (brand_df["avg_price"] >= price_range[0]) &
    (brand_df["avg_price"] <= price_range[1]) &
    (brand_df["avg_rating"] >= min_rating) &
    (brand_df["sentiment_score"] >= min_sentiment)
]

filtered_products_df = products_df[
    (products_df["brand"].isin(selected_brands)) &
    (products_df["selling_price"] >= price_range[0]) &
    (products_df["selling_price"] <= price_range[1]) &
    (products_df["rating"] >= min_rating)
]

filtered_reviews_df = reviews_df[reviews_df["brand"].isin(selected_brands)]

#  HEADER

st.markdown("# 🧳 Luggage Brand Intelligence Dashboard")
st.markdown("*Competitive analysis of major luggage brands on Amazon India*")
st.divider()

#  TABS

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏆 Brand Comparison",
    "🔍 Product Drilldown",
    "🎯 Aspect Analysis",
    "🤖 Agent Insights",
])

#  TAB 1: OVERVIEW

with tab1:
    avg_sentiment = filtered_brands_df["sentiment_score"].mean() if len(filtered_brands_df) > 0 else 0
    avg_price = filtered_brands_df["avg_price"].mean() if len(filtered_brands_df) > 0 else 0
    avg_discount = filtered_brands_df["avg_discount_pct"].mean() if len(filtered_brands_df) > 0 else 0
    avg_rating = filtered_brands_df["avg_rating"].mean() if len(filtered_brands_df) > 0 else 0
    best_vfm = filtered_brands_df.loc[filtered_brands_df["vfm_score"].idxmax(), "brand"] if len(filtered_brands_df) > 0 else "N/A"

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Brands Tracked", len(filtered_brands_df))
    k2.metric("Avg Selling Price", f"₹{avg_price:,.0f}")
    k3.metric("Avg Discount", f"{avg_discount:.1f}%")
    k4.metric("Avg Rating", f"{avg_rating:.2f} ★")
    k5.metric("Best Value/Money", best_vfm)

    st.markdown("")

    col_left, col_right = st.columns(2)

    # ── Sentiment score by brand ──────────────────────────────────────────────
    with col_left:
        st.markdown("#### Sentiment Score by Brand")
        fig_sent = px.bar(
            filtered_brands_df.sort_values("sentiment_score", ascending=True),
            x="sentiment_score", y="brand",
            orientation="h",
            color="sentiment_score",
            color_continuous_scale=[[0, "#f85149"], [0.5, "#e3b341"], [1, "#3fb950"]],
            range_color=[-0.3, 0.6],
            text="sentiment_score",
        )
        fig_sent.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_sent.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3", coloraxis_showscale=False,
            xaxis=dict(range=[-0.5, 0.8], gridcolor="#30363d"),
            yaxis_title=None, xaxis_title="Compound Sentiment",
            margin=dict(l=10, r=60, t=10, b=10), height=300,
        )
        st.plotly_chart(fig_sent, use_container_width=True)

    # ── Avg price vs rating bubble ────────────────────────────────────────────
    with col_right:
        st.markdown("#### Price vs Rating (bubble = review volume)")
        fig_bubble = px.scatter(
            filtered_brands_df,
            x="avg_price", y="avg_rating",
            size="total_review_count",
            color="brand",
            color_discrete_map=BRAND_COLORS,
            text="brand",
            hover_data=["avg_discount_pct", "sentiment_score"],
            size_max=50,
        )
        fig_bubble.update_traces(textposition="top center")
        fig_bubble.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            xaxis=dict(title="Avg Selling Price (₹)", gridcolor="#30363d"),
            yaxis=dict(title="Avg Rating", gridcolor="#30363d"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10), height=300,
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    # ── Discount vs price band ────────────────────────────────────────────────
    st.markdown("#### Price Band Distribution by Brand")
    products_copy = filtered_products_df.copy()
    products_copy["price_band"] = pd.cut(
        products_copy["selling_price"],
        bins=[0, 2500, 5000, 9000, 100000],
        labels=["Budget (<₹2.5k)", "Mid (₹2.5k–5k)", "Premium (₹5k–9k)", "Luxury (>₹9k)"]
    )
    band_dist = products_copy.groupby(["brand", "price_band"]).size().reset_index(name="count")
    fig_band = px.bar(
        band_dist, x="brand", y="count", color="price_band",
        barmode="stack",
        color_discrete_map={
            "Budget (<₹2.5k)": "#e3b341",
            "Mid (₹2.5k–5k)": "#58a6ff",
            "Premium (₹5k–9k)": "#3fb950",
            "Luxury (>₹9k)": "#bc8cff",
        }
    )
    fig_band.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d"),
        legend=dict(bgcolor="#161b22", bordercolor="#30363d"),
        margin=dict(l=10, r=10, t=10, b=10), height=280,
    )
    st.plotly_chart(fig_band, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2: BRAND COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("#### 🕸️ Competitive Radar")
        radar_brands = st.multiselect(
            "Select brands for radar", ALL_BRANDS, default=ALL_BRANDS[:4], key="radar_sel"
        )
        if radar_brands:
            st.plotly_chart(make_radar_chart(radar_brands), use_container_width=True)

    with col_b:
        st.markdown("#### 📉 Avg Discount % by Brand")
        fig_disc = px.bar(
            filtered_brands_df.sort_values("avg_discount_pct", ascending=False),
            x="brand", y="avg_discount_pct",
            color="brand", color_discrete_map=BRAND_COLORS,
            text="avg_discount_pct",
        )
        fig_disc.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
        fig_disc.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            xaxis=dict(gridcolor="#30363d"), yaxis=dict(gridcolor="#30363d", title="Discount %"),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
        )
        st.plotly_chart(fig_disc, use_container_width=True)

    st.divider()

    # ── Comparison table ──────────────────────────────────────────────────────
    st.markdown("#### 📋 Full Brand Comparison Table")

    display_cols = {
        "brand": "Brand",
        "segment": "Segment",
        "avg_price": "Avg Price (₹)",
        "avg_discount_pct": "Discount %",
        "avg_rating": "Rating ★",
        "sentiment_score": "Sentiment",
        "pct_positive": "% Positive",
        "pct_negative": "% Negative",
        "total_review_count": "Total Reviews",
        "vfm_score": "Value Score",
        "top_praise": "Top Praise",
        "top_complaints": "Top Complaints",
    }
    
    display_df = filtered_brands_df[[c for c in display_cols if c in filtered_brands_df.columns]].copy()
    display_df.columns = [display_cols[c] for c in display_df.columns]
    display_df = display_df.sort_values("Value Score", ascending=False)

    st.dataframe(
        display_df.style.background_gradient(
            subset=["Sentiment", "Rating ★", "Value Score"], cmap="RdYlGn"
        ).background_gradient(
            subset=["Discount %"], cmap="YlOrRd"
        ).format({
            "Avg Price (₹)": "₹{:.0f}",
            "Discount %": "{:.1f}%",
            "Rating ★": "{:.2f}",
            "Sentiment": "{:.3f}",
            "% Positive": "{:.1f}%",
            "% Negative": "{:.1f}%",
            "Total Reviews": "{:,.0f}",
            "Value Score": "{:.3f}",
        }),
        use_container_width=True, height=280,
    )

    st.divider()

    # ── Per-brand praise/complaint themes ─────────────────────────────────────
    st.markdown("#### 💬 Praise vs Complaint Themes")
    theme_brand = st.selectbox("Select brand for theme breakdown", ALL_BRANDS, key="theme_brand")

    row = brand_df[brand_df["brand"] == theme_brand].iloc[0]
    tc1, tc2 = st.columns(2)

    with tc1:
        fig_p = plot_theme_horizontal(row.get("top_praise", ""), "#3fb950", "✅ Top Praise Themes")
        st.plotly_chart(fig_p, use_container_width=True)

    with tc2:
        fig_c = plot_theme_horizontal(row.get("top_complaints", ""), "#f85149", "⚠️ Top Complaint Themes")
        st.plotly_chart(fig_c, use_container_width=True)

    # Brand summary from LLM
    brand_summary = insights["brand_summaries"].get(theme_brand, "")
    if brand_summary:
        st.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🤖 AI Brand Summary: {theme_brand}</div>
            <div class="insight-text">{brand_summary}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Sentiment distribution ────────────────────────────────────────────────
    st.markdown("#### 📊 Review Sentiment Distribution")
    fig_dist = go.Figure()
    for brand in (selected_brands or ALL_BRANDS):
        b_reviews = filtered_reviews_df[filtered_reviews_df["brand"] == brand]
        if "compound_score" not in b_reviews.columns:
            continue
        fig_dist.add_trace(go.Box(
            y=b_reviews["compound_score"],
            name=brand,
            marker_color=brand_color(brand),
            boxmean=True,
        ))
    fig_dist.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
        font_color="#e6edf3",
        yaxis=dict(title="VADER Compound Score", gridcolor="#30363d"),
        xaxis=dict(gridcolor="#30363d"),
        legend=dict(bgcolor="#161b22"),
        margin=dict(l=10, r=10, t=10, b=10), height=320,
    )
    st.plotly_chart(fig_dist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3: PRODUCT DRILLDOWN
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    drill_brand = st.selectbox("Select Brand", ALL_BRANDS, key="drill_brand")
    brand_products = filtered_products_df[filtered_products_df["brand"] == drill_brand].copy()

    if brand_products.empty:
        st.warning("No products match current filters.")
    else:
        # Sort options
        sort_col = st.selectbox("Sort products by", ["rating", "selling_price", "discount_pct", "review_count"], key="sort_col")
        sort_asc = st.checkbox("Ascending", value=False, key="sort_asc")
        brand_products = brand_products.sort_values(sort_col, ascending=sort_asc)

        # Product cards
        for _, product in brand_products.iterrows():
            with st.expander(f"🧳 {product['title']}   |   ₹{product['selling_price']:,.0f}   |   {'⭐' * int(round(product['rating']))}  ({product['rating']})"):
                pc1, pc2, pc3, pc4 = st.columns(4)
                pc1.metric("Selling Price", f"₹{product['selling_price']:,.0f}")
                pc2.metric("MRP", f"₹{product['mrp']:,.0f}")
                pc3.metric("Discount", f"{product['discount_pct']:.0f}%")
                pc4.metric("Rating", f"{product['rating']} ★  ({product['review_count']:,} reviews)")

                # Reviews for this product
                product_reviews = filtered_reviews_df[filtered_reviews_df["asin"] == product["asin"]]
                if len(product_reviews) > 0:
                    # Sentiment bar
                    pos = (product_reviews["stars"] >= 4).sum()
                    neu = (product_reviews["stars"] == 3).sum()
                    neg = (product_reviews["stars"] <= 2).sum()
                    total = len(product_reviews)

                    rv1, rv2, rv3 = st.columns(3)
                    rv1.metric("✅ Positive", f"{pos/total*100:.0f}%")
                    rv2.metric("😐 Neutral", f"{neu/total*100:.0f}%")
                    rv3.metric("❌ Negative", f"{neg/total*100:.0f}%")

                    # Star distribution
                    star_counts = product_reviews["stars"].value_counts().sort_index()
                    fig_stars = px.bar(
                        x=star_counts.index, y=star_counts.values,
                        labels={"x": "Stars", "y": "Reviews"},
                        color=star_counts.index,
                        color_continuous_scale=[[0, "#f85149"], [0.5, "#e3b341"], [1, "#3fb950"]],
                    )
                    fig_stars.update_layout(
                        paper_bgcolor="#161b22", plot_bgcolor="#161b22",
                        font_color="#e6edf3", showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(l=10, r=10, t=10, b=10), height=180,
                    )
                    st.plotly_chart(fig_stars, use_container_width=True)

                    # Sample reviews
                    st.markdown("**Sample Reviews:**")
                    sample = product_reviews.sample(min(5, len(product_reviews)), random_state=42)
                    for _, rev in sample.iterrows():
                        stars_display = "⭐" * int(rev["stars"]) + "☆" * (5 - int(rev["stars"]))
                        label_color = {"positive": "#3fb950", "neutral": "#e3b341", "negative": "#f85149"}
                        label = rev.get("vader_label", rev.get("sentiment_label", "neutral"))
                        color = label_color.get(label, "#8b949e")
                        st.markdown(
                            f'<div style="background:#161b22;border:1px solid #30363d;border-left:3px solid {color};'
                            f'border-radius:6px;padding:10px 14px;margin:4px 0;color:#c9d1d9;font-size:13px;">'
                            f'{stars_display} &nbsp; <span style="color:{color};font-size:11px;">[{label.upper()}]</span><br>'
                            f'{str(rev["review_text"])[:300]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("No reviews found for this product in the current filtered dataset.")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4: ASPECT ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("#### 🎯 Aspect-Level Sentiment Heatmap")
    st.markdown("*How customers feel about specific product attributes across brands*")

    aspect_cols = [c for c in brand_df.columns if c.startswith("aspect_") and c.endswith("_avg")]
    aspect_labels = {f"aspect_{a}_avg": a.capitalize() for a in ASPECT_KEYWORDS.keys()}

    if aspect_cols:
        heat_df = filtered_brands_df[["brand"] + [c for c in aspect_cols if c in filtered_brands_df.columns]].set_index("brand")
        heat_df.columns = [aspect_labels.get(c, c) for c in heat_df.columns]
        heat_df = heat_df.dropna(how="all", axis=1)

        fig_heat = px.imshow(
            heat_df,
            color_continuous_scale=[[0, "#f85149"], [0.5, "#e3b341"], [1, "#3fb950"]],
            range_color=[-0.5, 0.7],
            aspect="auto",
            text_auto=".2f",
        )
        fig_heat.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#0f1117",
            font_color="#e6edf3",
            coloraxis_colorbar=dict(tickfont_color="#e6edf3"),
            margin=dict(l=10, r=10, t=10, b=10), height=350,
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Aspect-level data not available. Ensure sentiment analysis was run.")

    st.divider()

    # ── Value-for-money analysis ──────────────────────────────────────────────
    st.markdown("#### 💰 Value-for-Money Analysis")
    st.markdown("*Sentiment score adjusted by price band — who delivers the most per rupee?*")

    if "vfm_score" in filtered_brands_df.columns:
        fig_vfm = px.bar(
            filtered_brands_df.sort_values("vfm_score", ascending=False),
            x="brand", y="vfm_score",
            color="brand", color_discrete_map=BRAND_COLORS,
            text="vfm_score",
        )
        fig_vfm.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_vfm.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3", showlegend=False,
            yaxis=dict(title="Value-for-Money Score", gridcolor="#30363d"),
            xaxis=dict(gridcolor="#30363d"),
            margin=dict(l=10, r=10, t=10, b=10), height=300,
        )
        st.plotly_chart(fig_vfm, use_container_width=True)

    # ── Aspect deep-dive per brand ────────────────────────────────────────────
    st.markdown("#### 🔬 Aspect Breakdown per Brand")
    aspect_brand = st.selectbox("Select brand for aspect detail", ALL_BRANDS, key="aspect_brand")

    b_row = brand_df[brand_df["brand"] == aspect_brand].iloc[0]
    aspect_scores = {}
    for asp in ASPECT_KEYWORDS.keys():
        col = f"aspect_{asp}_avg"
        if col in b_row.index and not pd.isna(b_row[col]):
            aspect_scores[asp.capitalize()] = b_row[col]

    if aspect_scores:
        fig_asp = go.Figure(go.Bar(
            x=list(aspect_scores.keys()),
            y=list(aspect_scores.values()),
            marker_color=[sentiment_color(v) for v in aspect_scores.values()],
            text=[f"{v:.3f}" for v in aspect_scores.values()],
            textposition="outside",
        ))
        fig_asp.add_hline(y=0, line_dash="dash", line_color="#8b949e")
        fig_asp.update_layout(
            paper_bgcolor="#161b22", plot_bgcolor="#161b22",
            font_color="#e6edf3",
            yaxis=dict(title="Avg Sentiment Score", gridcolor="#30363d"),
            xaxis=dict(gridcolor="#30363d"),
            margin=dict(l=10, r=10, t=10, b=10), height=280,
        )
        st.plotly_chart(fig_asp, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5: AGENT INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("#### 🤖 Agent Insights")
    st.markdown("*Non-obvious strategic conclusions generated by AI from the full dataset*")
    st.divider()

    agent_insights = insights.get("agent_insights", [])

    if agent_insights:
        for i, item in enumerate(agent_insights, 1):
            title = item.get("title", f"Insight {i}")
            insight = item.get("insight", "")
            implication = item.get("implication", "")
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">#{i} — {title}</div>
                <div class="insight-text">{insight}</div>
                <div class="insight-implication">💡 Implication: {implication}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No LLM insights available. Get a free key at console.groq.com, set GROQ_API_KEY in .env, and re-run `python run_pipeline.py`")

    st.divider()

    # ── Anomaly detection ─────────────────────────────────────────────────────
    st.markdown("#### 🚨 Anomaly & Trust Signals")

    from analysis.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()

    if "compound_score" in filtered_reviews_df.columns:
        anomalies = analyzer.detect_anomalies(filtered_products_df, filtered_reviews_df)
        trust_flags = analyzer.detect_trust_issues(filtered_reviews_df)

        if anomalies:
            st.markdown("**Detected Anomalies:**")
            for anomaly in anomalies:
                st.markdown(
                    f'<div class="anomaly-card">⚠️ <strong>[{anomaly["type"].replace("_", " ").title()}]</strong> — {anomaly["description"]}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.success("✅ No significant anomalies detected.")

        if trust_flags:
            st.markdown("**Review Trust Flags:**")
            for brand, flags in trust_flags.items():
                for flag in flags:
                    st.markdown(
                        f'<div class="anomaly-card"><strong>{brand}:</strong> {flag}</div>',
                        unsafe_allow_html=True,
                    )

    st.divider()

    # ── Competitive summary ───────────────────────────────────────────────────
    st.markdown("#### 📌 Competitive Summary: Who Is Winning and Why?")

    winner_sentiment = brand_df.loc[brand_df["sentiment_score"].idxmax()]
    winner_vfm = brand_df.loc[brand_df["vfm_score"].idxmax()]
    winner_rating = brand_df.loc[brand_df["avg_rating"].idxmax()]
    most_discount = brand_df.loc[brand_df["avg_discount_pct"].idxmax()]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">😊 Sentiment Leader</div>
            <div class="insight-text"><b>{winner_sentiment['brand']}</b><br>{winner_sentiment['sentiment_score']:.3f} score</div>
        </div>
    """, unsafe_allow_html=True)
    c2.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">💰 Best Value/₹</div>
            <div class="insight-text"><b>{winner_vfm['brand']}</b><br>{winner_vfm['vfm_score']:.3f} VFM</div>
        </div>
    """, unsafe_allow_html=True)
    c3.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">⭐ Rating Leader</div>
            <div class="insight-text"><b>{winner_rating['brand']}</b><br>{winner_rating['avg_rating']:.2f} ★</div>
        </div>
    """, unsafe_allow_html=True)
    c4.markdown(f"""
        <div class="insight-card">
            <div class="insight-title">🏷️ Most Discounted</div>
            <div class="insight-text"><b>{most_discount['brand']}</b><br>{most_discount['avg_discount_pct']:.1f}% avg</div>
        </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#8b949e;font-size:12px;'>Moonshot AI Agent Assignment · Amazon India Luggage Intelligence · Built with Streamlit + Plotly + VADER + Groq (llama-3.3-70b)</div>",
    unsafe_allow_html=True,
)
