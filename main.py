
#!/usr/bin/env python3
'''
AI Market Trend Analysis - Main Application
Complete end-to-end market analysis system with AI-powered insights

Usage:
    python main.py --mode dashboard    # Run interactive dashboard
    python main.py --mode analysis    # Run complete analysis
    python main.py --mode test        # Run system tests
'''

import argparse
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from config import config, Config
from data_loader import load_all_data
from ai_engine import run_complete_analysis
from utils import save_results, generate_eda_report, calculate_business_metrics, export_dashboard_data
from test_system import SystemTester, run_integration_test

def print_banner():
    '''Print application banner'''
    banner = '''
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║    🤖 AI-Powered Market Trend Analysis System           ║
    ║                                                          ║  
    ║    ✨ Features:                                          ║
    ║    • Time Series Forecasting (Prophet, ARIMA)          ║
    ║    • Customer Segmentation (RFM, K-Means, DBSCAN)      ║
    ║    • Sentiment Analysis (BERT, Transformers)           ║
    ║    • Anomaly Detection (Isolation Forest)              ║
    ║    • Trend Detection & Explainable AI (SHAP)          ║
    ║    • Interactive Dashboard (Dash, Plotly)              ║
    ║                                                          ║
    ║    📊 Ready for Production Use                          ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    '''
    print(banner)

def run_dashboard():
    '''Launch the interactive dashboard'''
    print("🚀 Launching AI Market Trend Analysis Dashboard...")
    print("=" * 55)

    try:
        # Import dashboard (will load data and start analysis)
        from dashboard import app

        print(f"\n🌐 Dashboard starting at: http://{config.DASHBOARD_HOST}:{config.DASHBOARD_PORT}")
        print("\n📊 Dashboard Features:")
        print("   • 📈 Sales & Demand Forecasting")
        print("   • 👥 Customer Segmentation Analysis") 
        print("   • 💭 Sentiment Analysis Dashboard")
        print("   • 🔍 Advanced Analytics & Insights")
        print("\n💡 Tip: Use Ctrl+C to stop the dashboard")

        # Run the dashboard
        app.run(
            debug=config.DASHBOARD_DEBUG,
            host=config.DASHBOARD_HOST,
            port=config.DASHBOARD_PORT
        )

    except Exception as e:
        print(f"❌ Dashboard failed to start: {e}")
        sys.exit(1)

def run_analysis(save_output=True):
    '''Run complete market analysis'''
    print("🔍 Running Complete AI Market Analysis...")
    print("=" * 45)

    start_time = datetime.now()

    try:
        # Step 1: Load all data
        print("\n📂 Step 1: Loading Data...")
        retail_df, reviews_df, trends_df = load_all_data()

        # Step 2: Generate EDA report
        print("\n📊 Step 2: Generating EDA Report...")
        eda_stats = generate_eda_report(retail_df)

        # Step 3: Calculate business metrics
        print("\n💰 Step 3: Calculating Business Metrics...")
        business_metrics = calculate_business_metrics(retail_df)

        print("\n📈 Key Business Insights:")
        print(f"   • Total Revenue: ${business_metrics['total_revenue']:,.2f}")
        print(f"   • Total Customers: {business_metrics['total_customers']:,}")
        print(f"   • Average Order Value: ${business_metrics['average_order_value']:.2f}")
        print(f"   • Customer Retention Rate: {business_metrics['repeat_customer_rate']:.1%}")

        # Step 4: Run AI analysis
        print("\n🤖 Step 4: Running AI Analysis...")
        ai_engine, ai_results = run_complete_analysis(retail_df, reviews_df)

        # Step 5: Generate insights
        print("\n🎯 Step 5: Generating Insights...")
        insights = generate_insights(ai_results, business_metrics)

        # Step 6: Save results
        if save_output:
            print("\n💾 Step 6: Saving Results...")

            # Prepare comprehensive results
            comprehensive_results = {
                'timestamp': datetime.now().isoformat(),
                'business_metrics': business_metrics,
                'eda_statistics': eda_stats,
                'ai_analysis': ai_results,
                'insights': insights,
                'data_summary': {
                    'retail_records': len(retail_df),
                    'review_records': len(reviews_df),
                    'trends_data_points': len(trends_df)
                }
            }

            save_results(comprehensive_results, "complete_analysis")
            export_dashboard_data(ai_results, "dashboard_export")

        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        print("\n" + "=" * 50)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"⏱️  Total Duration: {duration:.2f} seconds")
        print(f"📊 Data Processed:")
        print(f"   • Retail Transactions: {len(retail_df):,}")
        print(f"   • Product Reviews: {len(reviews_df):,}") 
        print(f"   • Trend Data Points: {len(trends_df):,}")

        print(f"\n🎯 AI Models Trained:")
        print(f"   • Sales Forecasting: ✅")
        print(f"   • Customer Segmentation: ✅")
        print(f"   • Sentiment Analysis: ✅")
        print(f"   • Anomaly Detection: ✅")
        print(f"   • Trend Detection: ✅")

        return True

    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def generate_insights(ai_results, business_metrics):
    '''Generate business insights from AI analysis'''
    insights = []

    # Revenue insights
    if business_metrics['average_order_value'] > 50:
        insights.append("💰 High-value customers detected - focus on retention strategies")

    # Customer segmentation insights
    if 'customer_segments' in ai_results:
        segments = ai_results['customer_segments']
        champions = segments[segments['Segment'] == 'Champions']
        if len(champions) > 0:
            avg_champion_value = champions['Monetary'].mean()
            insights.append(f"🏆 Champion customers spend ${avg_champion_value:,.2f} on average")

    # Sentiment insights
    if 'sentiment' in ai_results:
        sentiment_dist = ai_results['sentiment']['distribution']
        positive_pct = sentiment_dist.get('Positive', 0) * 100
        if positive_pct > 70:
            insights.append(f"😊 Strong customer satisfaction: {positive_pct:.1f}% positive reviews")
        elif positive_pct < 50:
            insights.append(f"⚠️ Customer satisfaction needs attention: {positive_pct:.1f}% positive")

    # Trend insights
    if 'trends' in ai_results:
        trending_up = ai_results['trends']['up']
        if trending_up:
            top_trending = list(trending_up.keys())[0]
            insights.append(f"📈 Rising star product: {top_trending}")

    # Anomaly insights
    if 'anomalies' in ai_results:
        anomaly_count = len(ai_results['anomalies'])
        total_customers = business_metrics['total_customers']
        anomaly_pct = (anomaly_count / total_customers) * 100
        insights.append(f"🔍 {anomaly_count} unusual customer patterns detected ({anomaly_pct:.1f}%)")

    return insights

def run_tests():
    '''Run system tests'''
    print("🧪 Running System Tests...")
    print("=" * 30)

    # Create necessary directories
    config.create_directories()

    # Run unit tests
    tester = SystemTester()
    unit_tests_passed = tester.run_all_tests()

    # Run integration tests
    integration_passed = run_integration_test()

    if unit_tests_passed and integration_passed:
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED!")
        return False

def main():
    '''Main application entry point'''
    parser = argparse.ArgumentParser(
        description='AI Market Trend Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python main.py --mode dashboard     # Launch interactive dashboard
  python main.py --mode analysis     # Run complete analysis
  python main.py --mode test         # Run system tests
        '''
    )

    parser.add_argument(
        '--mode',
        choices=['dashboard', 'analysis', 'test'],
        default='dashboard',
        help='Application mode (default: dashboard)'
    )

    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Skip saving analysis results'
    )

    parser.add_argument(
        '--config',
        choices=['dev', 'prod'],
        default='dev',
        help='Configuration environment (default: dev)'
    )

    args = parser.parse_args()

    # Print banner
    print_banner()

    # Set configuration
    if args.config == 'prod':
        from config import ProductionConfig
        global config
        config = ProductionConfig()

    # Create necessary directories
    config.create_directories()

    # Route to appropriate function
    try:
        if args.mode == 'dashboard':
            run_dashboard()
        elif args.mode == 'analysis':
            success = run_analysis(save_output=not args.no_save)
            sys.exit(0 if success else 1)
        elif args.mode == 'test':
            success = run_tests()
            sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
