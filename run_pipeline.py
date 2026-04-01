
import sys
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Moonshot Competitive Intelligence Pipeline")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="Use mock data (default)")
    parser.add_argument("--scrape", action="store_true",
                        help="Try live Amazon scraping first, fall back to mock")
    parser.add_argument("--skip-insights", action="store_true",
                        help="Skip LLM insight generation (faster, no API needed)")
    args = parser.parse_args()

    print("=" * 60)
    print("  MOONSHOT COMPETITIVE INTELLIGENCE PIPELINE")
    print("=" * 60)

    # Step 1: Data collection
    if args.scrape:
        print("\n[1/3]   Attempting live Amazon scrape")
        from scraper.amazon_scraper import run as scrape_run
        scrape_run()
    else:
        print("\n[1/3]  Generating mock data")
        from data.mock_data_generator import run as mock_run
        mock_run()

    # Step 2: Sentiment analysis
    print("\n[2/3] Running sentiment analysis")
    from analysis.sentiment_analyzer import SentimentAnalyzer
    analyzer = SentimentAnalyzer()
    brand_df, reviews_df, products_df = analyzer.run()

    # Step 3: LLM insights
    if not args.skip_insights:
        print("\n[3/3]  Generating Agent Insights")
        from analysis.insights_generator import InsightsGenerator
        gen = InsightsGenerator()
        gen.run()
    else:
        print("\n[3/3]  Skipping insights (--skip-insights flag set)")

    print("\n" + "=" * 60)
    print("  ✅ PIPELINE COMPLETE")
    print("  Launch dashboard with: streamlit run dashboard/app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
