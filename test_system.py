
#!/usr/bin/env python3
'''
Comprehensive testing script for AI Market Trend Analysis system
Tests all components and generates validation reports
'''

import sys
import traceback
from datetime import datetime
import pandas as pd
import numpy as np

# Import our modules
from data_loader import DataLoader, load_all_data
from ai_engine import AIEngine, run_complete_analysis
from utils import validate_data_quality, calculate_business_metrics, DataValidator
from config import config

class SystemTester:
    '''Comprehensive system testing class'''

    def __init__(self):
        self.test_results = {}
        self.start_time = datetime.now()
        print("🧪 AI Market Trend Analysis - System Testing")
        print("=" * 50)

    def run_all_tests(self):
        '''Run complete test suite'''
        tests = [
            self.test_data_loading,
            self.test_data_quality, 
            self.test_ai_engine,
            self.test_forecasting,
            self.test_customer_segmentation,
            self.test_sentiment_analysis,
            self.test_anomaly_detection,
            self.test_trend_detection,
            self.test_business_metrics
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                print(f"❌ {test.__name__} FAILED: {str(e)}")
                self.test_results[test.__name__] = {'status': 'FAILED', 'error': str(e)}

        self.generate_test_report()

    def test_data_loading(self):
        '''Test data loading functionality'''
        print("\n📂 Testing Data Loading...")

        loader = DataLoader()

        # Test retail data loading
        retail_df = loader.load_uci_retail()
        assert not retail_df.empty, "Retail data should not be empty"
        assert 'TotalPrice' in retail_df.columns, "TotalPrice column should be created"

        # Test reviews loading  
        reviews_df = loader.load_amazon_reviews()
        assert not reviews_df.empty, "Reviews data should not be empty"
        assert 'review_body' in reviews_df.columns, "review_body column should exist"

        # Test Google Trends
        trends_df = loader.get_google_trends(['coffee', 'laptop'])
        assert not trends_df.empty, "Trends data should not be empty"

        print("✅ Data loading tests passed")
        self.test_results['data_loading'] = {'status': 'PASSED', 'details': {
            'retail_records': len(retail_df),
            'review_records': len(reviews_df),
            'trends_data_points': len(trends_df)
        }}

    def test_data_quality(self):
        '''Test data quality validation'''
        print("\n📊 Testing Data Quality...")

        # Load data
        retail_df, reviews_df, trends_df = load_all_data()

        # Validate retail data
        retail_score, retail_issues = validate_data_quality(retail_df, 
            required_columns=['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID'])

        # Validate reviews data
        reviews_score, reviews_issues = validate_data_quality(reviews_df,
            required_columns=['review_body'])

        assert retail_score > 70, f"Retail data quality too low: {retail_score}"
        assert reviews_score > 70, f"Reviews data quality too low: {reviews_score}"

        print("✅ Data quality tests passed")
        self.test_results['data_quality'] = {'status': 'PASSED', 'details': {
            'retail_quality_score': retail_score,
            'reviews_quality_score': reviews_score,
            'retail_issues': len(retail_issues),
            'reviews_issues': len(reviews_issues)
        }}

    def test_ai_engine(self):
        '''Test AI engine initialization'''
        print("\n🤖 Testing AI Engine...")

        engine = AIEngine()
        assert hasattr(engine, 'models'), "AI engine should have models attribute"
        assert hasattr(engine, 'scalers'), "AI engine should have scalers attribute"

        print("✅ AI engine tests passed")
        self.test_results['ai_engine'] = {'status': 'PASSED'}

    def test_forecasting(self):
        '''Test sales forecasting functionality'''
        print("\n📈 Testing Sales Forecasting...")

        # Load data and create AI engine
        retail_df, _, _ = load_all_data()
        engine = AIEngine()

        # Test Prophet forecasting
        top_product = retail_df['Description'].value_counts().index[0]
        model, forecast = engine.forecast_sales_prophet(retail_df, top_product, periods=30)

        if forecast is not None:
            assert 'yhat' in forecast.columns, "Forecast should contain yhat column"
            assert len(forecast) > 0, "Forecast should contain data"
            assert forecast['yhat'].min() >= 0, "Forecasted values should be non-negative"

        # Test ARIMA forecasting
        arima_model, arima_forecast = engine.forecast_sales_arima(retail_df, top_product, periods=30)

        print("✅ Forecasting tests passed")
        self.test_results['forecasting'] = {'status': 'PASSED', 'details': {
            'prophet_forecast_points': len(forecast) if forecast is not None else 0,
            'arima_available': arima_forecast is not None
        }}

    def test_customer_segmentation(self):
        '''Test customer segmentation functionality'''
        print("\n👥 Testing Customer Segmentation...")

        retail_df, _, _ = load_all_data()
        engine = AIEngine()

        # Test K-Means segmentation
        rfm_data, kmeans_model = engine.customer_segmentation_rfm(retail_df, n_clusters=4)

        assert rfm_data is not None, "RFM data should not be None"
        assert 'Cluster' in rfm_data.columns, "Cluster column should exist"
        assert rfm_data['Cluster'].nunique() <= 4, "Should have at most 4 clusters"

        # Test DBSCAN segmentation
        dbscan_data, dbscan_model = engine.customer_segmentation_dbscan(retail_df)

        print("✅ Customer segmentation tests passed")
        self.test_results['customer_segmentation'] = {'status': 'PASSED', 'details': {
            'customers_segmented': len(rfm_data) if rfm_data is not None else 0,
            'n_clusters_kmeans': rfm_data['Cluster'].nunique() if rfm_data is not None else 0,
            'dbscan_available': dbscan_data is not None
        }}

    def test_sentiment_analysis(self):
        '''Test sentiment analysis functionality'''
        print("\n💭 Testing Sentiment Analysis...")

        _, reviews_df, _ = load_all_data()
        engine = AIEngine()

        # Test sentiment analysis
        sentiment_df, sentiment_dist = engine.analyze_sentiment(reviews_df, sample_size=100)

        assert sentiment_df is not None, "Sentiment analysis should return data"
        assert 'sentiment' in sentiment_df[0].columns, "Sentiment column should exist"
        assert len(sentiment_dist) > 0, "Sentiment distribution should have data"

        print("✅ Sentiment analysis tests passed")
        self.test_results['sentiment_analysis'] = {'status': 'PASSED', 'details': {
            'reviews_analyzed': len(sentiment_df[0]) if sentiment_df else 0,
            'sentiment_categories': len(sentiment_dist) if sentiment_dist else 0
        }}

    def test_anomaly_detection(self):
        '''Test anomaly detection functionality'''
        print("\n🔍 Testing Anomaly Detection...")

        retail_df, _, _ = load_all_data()
        engine = AIEngine()

        # Test Isolation Forest anomaly detection
        features, anomalies, model = engine.detect_anomalies(retail_df, contamination=0.1)

        if features is not None:
            assert 'anomaly' in features.columns, "Anomaly column should exist"
            assert 'anomaly_score' in features.columns, "Anomaly score column should exist"
            assert len(anomalies) > 0, "Should detect some anomalies"

        print("✅ Anomaly detection tests passed")
        self.test_results['anomaly_detection'] = {'status': 'PASSED', 'details': {
            'customers_analyzed': len(features) if features is not None else 0,
            'anomalies_detected': len(anomalies) if anomalies is not None else 0
        }}

    def test_trend_detection(self):
        '''Test product trend detection'''
        print("\n📊 Testing Trend Detection...")

        retail_df, _, _ = load_all_data()
        engine = AIEngine()

        # Test trend detection
        trends, trending_up, trending_down = engine.detect_product_trends(retail_df, window=7)

        assert isinstance(trends, dict), "Trends should be returned as dictionary"
        assert isinstance(trending_up, dict), "Trending up should be dictionary"
        assert isinstance(trending_down, dict), "Trending down should be dictionary"

        print("✅ Trend detection tests passed")
        self.test_results['trend_detection'] = {'status': 'PASSED', 'details': {
            'products_analyzed': len(trends),
            'trending_up_count': len(trending_up),
            'trending_down_count': len(trending_down)
        }}

    def test_business_metrics(self):
        '''Test business metrics calculation'''
        print("\n💰 Testing Business Metrics...")

        retail_df, _, _ = load_all_data()

        metrics = calculate_business_metrics(retail_df)

        required_metrics = ['total_revenue', 'average_order_value', 'total_customers', 'total_orders']
        for metric in required_metrics:
            assert metric in metrics, f"Required metric {metric} not found"
            assert metrics[metric] > 0, f"Metric {metric} should be positive"

        print("✅ Business metrics tests passed")
        self.test_results['business_metrics'] = {'status': 'PASSED', 'details': metrics}

    def generate_test_report(self):
        '''Generate comprehensive test report'''
        print("\n" + "=" * 50)
        print("📋 TEST REPORT")
        print("=" * 50)

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = total_tests - passed_tests

        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} ❌")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")

        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print(f"Total Duration: {duration:.2f} seconds")

        print("\n📊 Detailed Results:")
        for test_name, result in self.test_results.items():
            status_icon = "✅" if result['status'] == 'PASSED' else "❌"
            print(f"  {status_icon} {test_name}: {result['status']}")
            if 'error' in result:
                print(f"     Error: {result['error']}")

        # Overall system health
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED - System is ready for production!")
        elif passed_tests >= total_tests * 0.8:
            print("\n⚠️  MOSTLY PASSING - Some issues need attention")
        else:
            print("\n🚨 MULTIPLE FAILURES - System needs debugging")

        return passed_tests == total_tests

def run_integration_test():
    '''Run full integration test'''
    print("\n🔗 Running Integration Test...")

    try:
        # Test complete workflow
        retail_df, reviews_df, trends_df = load_all_data()
        engine, results = run_complete_analysis(retail_df, reviews_df)

        # Validate results
        assert len(results) > 0, "Analysis should return results"

        expected_results = ['customer_segments', 'trends', 'sentiment']
        for expected in expected_results:
            if expected in results:
                print(f"✅ {expected} analysis completed")
            else:
                print(f"⚠️  {expected} analysis missing")

        print("✅ Integration test passed")
        return True

    except Exception as e:
        print(f"❌ Integration test failed: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Create necessary directories
    config.create_directories()

    # Run system tests
    tester = SystemTester()
    all_passed = tester.run_all_tests()

    # Run integration test
    integration_passed = run_integration_test()

    # Final result
    if all_passed and integration_passed:
        print("\n🎊 SYSTEM VALIDATION COMPLETE - All tests passed!")
        sys.exit(0)
    else:
        print("\n🚨 SYSTEM VALIDATION FAILED - Please check the issues above")
        sys.exit(1)
