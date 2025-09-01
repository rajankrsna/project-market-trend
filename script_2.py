# 3. ai_engine.py - Complete AI/ML engine with all required techniques
ai_engine_py = """
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Time Series Forecasting
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA

# Machine Learning
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# NLP and Sentiment Analysis
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Explainable AI
import shap

class AIEngine:
    '''Complete AI engine for market trend analysis'''
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.sentiment_analyzer = None
        self.explainer = None
        print("🤖 AI Engine initialized")
        
    def forecast_sales_prophet(self, df, product_name, periods=90):
        '''Time series forecasting using Facebook Prophet'''
        try:
            # Filter data for specific product
            product_data = df[df['Description'].str.contains(product_name, case=False, na=False)]
            
            if len(product_data) < 10:
                print(f"⚠️  Insufficient data for {product_name}, using aggregated data")
                daily_sales = df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
            else:
                daily_sales = product_data.groupby('InvoiceDate')['Quantity'].sum().reset_index()
            
            # Prepare data for Prophet
            prophet_df = daily_sales.rename(columns={'InvoiceDate': 'ds', 'Quantity': 'y'})
            prophet_df = prophet_df.sort_values('ds')
            
            # Initialize and fit Prophet model
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            
            model.fit(prophet_df)
            
            # Make future predictions
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            
            # Store model
            self.models[f'prophet_{product_name}'] = model
            
            print(f"✓ Prophet forecast generated for {product_name}")
            return model, forecast
            
        except Exception as e:
            print(f"❌ Prophet forecast error: {e}")
            return None, None
    
    def forecast_sales_arima(self, df, product_name, periods=30):
        '''ARIMA time series forecasting'''
        try:
            # Filter and aggregate data
            product_data = df[df['Description'].str.contains(product_name, case=False, na=False)]
            daily_sales = product_data.groupby('InvoiceDate')['Quantity'].sum()
            
            if len(daily_sales) < 50:
                print(f"⚠️  Insufficient data for ARIMA model")
                return None, None
            
            # Fit ARIMA model
            model = ARIMA(daily_sales, order=(1,1,1))
            fitted_model = model.fit()
            
            # Make predictions
            forecast = fitted_model.forecast(steps=periods)
            
            self.models[f'arima_{product_name}'] = fitted_model
            
            print(f"✓ ARIMA forecast generated for {product_name}")
            return fitted_model, forecast
            
        except Exception as e:
            print(f"❌ ARIMA forecast error: {e}")
            return None, None
    
    def customer_segmentation_rfm(self, df, n_clusters=4):
        '''RFM-based customer segmentation using K-Means'''
        try:
            # Calculate RFM metrics
            snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
            
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,  # Recency
                'InvoiceNo': 'nunique',  # Frequency
                'TotalPrice': 'sum'  # Monetary
            }).rename(columns={
                'InvoiceDate': 'Recency',
                'InvoiceNo': 'Frequency', 
                'TotalPrice': 'Monetary'
            })
            
            # Scale the features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            
            # K-Means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)
            
            # Label clusters
            cluster_labels = {
                0: 'Champions',
                1: 'Loyal Customers', 
                2: 'Potential Loyalists',
                3: 'At Risk'
            }
            
            rfm['Segment'] = rfm['Cluster'].map(cluster_labels)
            
            # Store models
            self.models['rfm_kmeans'] = kmeans
            self.scalers['rfm_scaler'] = scaler
            
            print(f"✓ Customer segmentation complete: {n_clusters} segments identified")
            return rfm, kmeans
            
        except Exception as e:
            print(f"❌ Customer segmentation error: {e}")
            return None, None
    
    def customer_segmentation_dbscan(self, df, eps=0.5, min_samples=5):
        '''DBSCAN clustering for customer segmentation'''
        try:
            # Calculate RFM metrics (same as above)
            snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)
            
            rfm = df.groupby('CustomerID').agg({
                'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
                'InvoiceNo': 'nunique',
                'TotalPrice': 'sum'
            }).rename(columns={
                'InvoiceDate': 'Recency',
                'InvoiceNo': 'Frequency',
                'TotalPrice': 'Monetary'
            })
            
            # Scale features
            scaler = StandardScaler()
            rfm_scaled = scaler.fit_transform(rfm)
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            rfm['Cluster'] = dbscan.fit_predict(rfm_scaled)
            
            n_clusters = len(set(rfm['Cluster'])) - (1 if -1 in rfm['Cluster'] else 0)
            
            self.models['rfm_dbscan'] = dbscan
            self.scalers['rfm_scaler_dbscan'] = scaler
            
            print(f"✓ DBSCAN segmentation complete: {n_clusters} clusters found")
            return rfm, dbscan
            
        except Exception as e:
            print(f"❌ DBSCAN segmentation error: {e}")
            return None, None
    
    def initialize_sentiment_analyzer(self):
        '''Initialize sentiment analysis pipeline'''
        try:
            # Use a lightweight sentiment analysis model
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            print("✓ Sentiment analyzer initialized")
            
        except Exception as e:
            print(f"❌ Sentiment analyzer error: {e}")
            # Fallback to simple rule-based sentiment
            self.sentiment_analyzer = None
    
    def analyze_sentiment(self, reviews_df, sample_size=1000):
        '''Analyze sentiment of product reviews'''
        try:
            if self.sentiment_analyzer is None:
                self.initialize_sentiment_analyzer()
            
            # Sample reviews for efficiency
            sample_reviews = reviews_df.sample(min(sample_size, len(reviews_df)))
            
            if self.sentiment_analyzer:
                # Use transformer model
                reviews_text = sample_reviews['review_body'].tolist()
                sentiments = []
                
                for text in reviews_text[:500]:  # Process in batches for memory
                    try:
                        result = self.sentiment_analyzer(text[:512])[0]  # Truncate long texts
                        # Convert to simple positive/negative
                        if result['label'] == 'POSITIVE':
                            sentiments.append('Positive')
                        else:
                            sentiments.append('Negative')
                    except:
                        sentiments.append('Neutral')
                
                # Create results dataframe
                results_df = sample_reviews.head(len(sentiments)).copy()
                results_df['sentiment'] = sentiments
                
            else:
                # Fallback: rule-based sentiment
                results_df = self._rule_based_sentiment(sample_reviews)
            
            # Calculate sentiment distribution
            sentiment_dist = results_df['sentiment'].value_counts(normalize=True)
            
            print(f"✓ Sentiment analysis complete for {len(results_df)} reviews")
            return results_df, sentiment_dist
            
        except Exception as e:
            print(f"❌ Sentiment analysis error: {e}")
            return self._rule_based_sentiment(reviews_df.sample(min(100, len(reviews_df))))
    
    def _rule_based_sentiment(self, df):
        '''Simple rule-based sentiment analysis fallback'''
        positive_words = ['great', 'excellent', 'amazing', 'love', 'perfect', 'good']
        negative_words = ['bad', 'terrible', 'poor', 'worst', 'awful', 'hate']
        
        def classify_sentiment(text):
            text_lower = str(text).lower()
            pos_count = sum([word in text_lower for word in positive_words])
            neg_count = sum([word in text_lower for word in negative_words])
            
            if pos_count > neg_count:
                return 'Positive'
            elif neg_count > pos_count:
                return 'Negative'
            else:
                return 'Neutral'
        
        df['sentiment'] = df['review_body'].apply(classify_sentiment)
        sentiment_dist = df['sentiment'].value_counts(normalize=True)
        
        print("✓ Rule-based sentiment analysis complete")
        return df, sentiment_dist
    
    def detect_anomalies(self, df, contamination=0.1):
        '''Anomaly detection using Isolation Forest'''
        try:
            # Prepare features for anomaly detection
            features = df.groupby('CustomerID').agg({
                'TotalPrice': ['sum', 'mean', 'std'],
                'Quantity': ['sum', 'mean'],
                'InvoiceNo': 'nunique'
            }).fillna(0)
            
            # Flatten column names
            features.columns = ['_'.join(col).strip() for col in features.columns.values]
            
            # Initialize Isolation Forest
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            # Fit and predict
            anomaly_scores = iso_forest.fit_predict(features)
            features['anomaly'] = anomaly_scores
            features['anomaly_score'] = iso_forest.score_samples(features.drop('anomaly', axis=1))
            
            # Get anomalies
            anomalies = features[features['anomaly'] == -1]
            
            self.models['isolation_forest'] = iso_forest
            
            print(f"✓ Anomaly detection complete: {len(anomalies)} anomalies found")
            return features, anomalies, iso_forest
            
        except Exception as e:
            print(f"❌ Anomaly detection error: {e}")
            return None, None, None
    
    def predict_product_success(self, df, target_col='TotalPrice'):
        '''Supervised learning to predict product success'''
        try:
            # Create product features
            product_features = df.groupby('Description').agg({
                'Quantity': ['mean', 'sum', 'std'],
                'UnitPrice': ['mean', 'std'],
                'TotalPrice': ['mean', 'sum'],
                'CustomerID': 'nunique'
            }).fillna(0)
            
            # Flatten columns
            product_features.columns = ['_'.join(col).strip() for col in product_features.columns.values]
            
            # Create target variable (high/low success)
            target = (product_features['TotalPrice_sum'] > product_features['TotalPrice_sum'].median()).astype(int)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                product_features, target, test_size=0.3, random_state=42
            )
            
            # Train Random Forest
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            
            # Train Logistic Regression for classification
            lr_model = LogisticRegression(random_state=42)
            lr_model.fit(X_train, y_train)
            
            # Predictions
            rf_pred = rf_model.predict(X_test)
            lr_pred = lr_model.predict(X_test)
            
            # Store models
            self.models['product_success_rf'] = rf_model
            self.models['product_success_lr'] = lr_model
            
            accuracy = accuracy_score(y_test, lr_pred)
            print(f"✓ Product success prediction model trained (Accuracy: {accuracy:.3f})")
            
            return rf_model, lr_model, product_features
            
        except Exception as e:
            print(f"❌ Product success prediction error: {e}")
            return None, None, None
    
    def explain_with_shap(self, model, X, model_type='tree'):
        '''Generate SHAP explanations for model predictions'''
        try:
            if model_type == 'tree':
                explainer = shap.TreeExplainer(model)
            else:
                explainer = shap.LinearExplainer(model, X)
            
            shap_values = explainer.shap_values(X)
            
            self.explainer = explainer
            
            print("✓ SHAP explanations generated")
            return explainer, shap_values
            
        except Exception as e:
            print(f"❌ SHAP explanation error: {e}")
            return None, None
    
    def detect_product_trends(self, df, window=30):
        '''Detect trending products using moving averages'''
        try:
            # Calculate daily sales per product
            daily_sales = df.groupby(['InvoiceDate', 'Description'])['Quantity'].sum().reset_index()
            
            trends = {}
            for product in daily_sales['Description'].unique():
                product_sales = daily_sales[daily_sales['Description'] == product]
                product_sales = product_sales.sort_values('InvoiceDate')
                
                if len(product_sales) >= window:
                    # Calculate moving average
                    product_sales['MA'] = product_sales['Quantity'].rolling(window=window).mean()
                    
                    # Calculate trend (slope of recent period)
                    recent_data = product_sales.tail(window)
                    if len(recent_data) > 1:
                        trend_slope = np.polyfit(range(len(recent_data)), recent_data['MA'].fillna(0), 1)[0]
                        trends[product] = trend_slope
            
            # Sort by trend strength
            trending_up = dict(sorted(trends.items(), key=lambda x: x[1], reverse=True)[:10])
            trending_down = dict(sorted(trends.items(), key=lambda x: x[1])[:10])
            
            print(f"✓ Product trend analysis complete for {len(trends)} products")
            return trends, trending_up, trending_down
            
        except Exception as e:
            print(f"❌ Product trend detection error: {e}")
            return {}, {}, {}

# Convenience functions for easy import
def run_complete_analysis(retail_df, reviews_df=None):
    '''Run complete AI analysis pipeline'''
    engine = AIEngine()
    results = {}
    
    print("\\n🚀 Starting complete AI analysis...")
    
    # 1. Time Series Forecasting
    top_products = retail_df['Description'].value_counts().head(3).index.tolist()
    for product in top_products:
        model, forecast = engine.forecast_sales_prophet(retail_df, product)
        if forecast is not None:
            results[f'forecast_{product}'] = forecast
    
    # 2. Customer Segmentation
    rfm_data, kmeans_model = engine.customer_segmentation_rfm(retail_df)
    if rfm_data is not None:
        results['customer_segments'] = rfm_data
    
    # 3. Anomaly Detection
    anomaly_features, anomalies, iso_model = engine.detect_anomalies(retail_df)
    if anomalies is not None:
        results['anomalies'] = anomalies
    
    # 4. Product Success Prediction
    rf_model, lr_model, product_features = engine.predict_product_success(retail_df)
    if product_features is not None:
        results['product_success'] = product_features
    
    # 5. Trend Detection
    trends, trending_up, trending_down = engine.detect_product_trends(retail_df)
    results['trends'] = {'all': trends, 'up': trending_up, 'down': trending_down}
    
    # 6. Sentiment Analysis (if reviews available)
    if reviews_df is not None:
        sentiment_df, sentiment_dist = engine.analyze_sentiment(reviews_df)
        results['sentiment'] = {'data': sentiment_df, 'distribution': sentiment_dist}
    
    print("\\n✅ Complete AI analysis finished!")
    return engine, results

if __name__ == "__main__":
    print("🤖 AI Engine module loaded successfully!")
"""

with open("ai_engine.py", "w") as f:
    f.write(ai_engine_py)

print("✓ ai_engine.py created")