
import pandas as pd
import numpy as np
from pytrends.request import TrendReq
import requests
from io import BytesIO
import warnings
import os
from datetime import datetime, timedelta
import time

warnings.filterwarnings('ignore')

class DataLoader:
    '''Comprehensive data loader for Market Trend Analysis'''

    def __init__(self):
        self.pytrends = TrendReq(hl='en-US', tz=360)

    def load_uci_retail(self, file_path=None):
        '''Load UCI Online Retail dataset'''
        try:
            if file_path and os.path.exists(file_path):
                # Load local file
                if file_path.endswith('.xlsx'):
                    df = pd.read_excel(file_path, engine='openpyxl')
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
            else:
                # Generate sample data if file not found
                print("⚠️  File not found, generating sample retail data...")
                df = self._generate_sample_retail_data()

            # Clean and preprocess
            df = self._clean_retail_data(df)
            print(f"✓ UCI Retail data loaded: {len(df):,} records")
            return df

        except Exception as e:
            print(f"❌ Error loading retail data: {e}")
            return self._generate_sample_retail_data()

    def load_amazon_reviews(self, file_path=None):
        '''Load Amazon product reviews dataset'''
        try:
            if file_path and os.path.exists(file_path):
                df = pd.read_csv(file_path)
                # Keep only essential columns
                if 'review_body' not in df.columns and 'reviewText' in df.columns:
                    df = df.rename(columns={'reviewText': 'review_body'})
            else:
                print("⚠️  Reviews file not found, generating sample review data...")
                df = self._generate_sample_reviews()

            df = self._clean_reviews_data(df)
            print(f"✓ Amazon reviews loaded: {len(df):,} records")
            return df

        except Exception as e:
            print(f"❌ Error loading reviews: {e}")
            return self._generate_sample_reviews()

    def get_google_trends(self, keywords, timeframe='today 12-m', geo=''):
        '''Fetch Google Trends data'''
        try:
            self.pytrends.build_payload(keywords, timeframe=timeframe, geo=geo)
            trends_df = self.pytrends.interest_over_time()

            if not trends_df.empty:
                trends_df = trends_df.drop('isPartial', axis=1)
                trends_df.reset_index(inplace=True)
                print(f"✓ Google Trends data fetched for: {', '.join(keywords)}")
                return trends_df
            else:
                return self._generate_sample_trends(keywords)

        except Exception as e:
            print(f"⚠️  Google Trends API error: {e}, using sample data")
            return self._generate_sample_trends(keywords)

    def _clean_retail_data(self, df):
        '''Clean and preprocess retail data'''
        # Remove rows with missing CustomerID
        df = df.dropna(subset=['CustomerID'])

        # Convert InvoiceDate to datetime
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')

        # Remove returns (invoices starting with 'C')
        df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

        # Calculate TotalPrice
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        # Remove invalid transactions
        df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]

        # Add time-based features
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['Hour'] = df['InvoiceDate'].dt.hour

        return df

    def _clean_reviews_data(self, df):
        '''Clean reviews data'''
        # Remove empty reviews
        df = df.dropna(subset=['review_body'])
        df = df[df['review_body'].str.len() > 10]

        # Add rating if not present
        if 'rating' not in df.columns and 'overall' in df.columns:
            df['rating'] = df['overall']
        elif 'rating' not in df.columns:
            df['rating'] = np.random.choice([1,2,3,4,5], len(df), p=[0.1,0.1,0.2,0.3,0.3])

        return df

    def _generate_sample_retail_data(self):
        '''Generate realistic sample retail data'''
        np.random.seed(42)

        # Generate date range
        start_date = datetime(2020, 1, 1)
        end_date = datetime(2023, 12, 31)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')

        # Product catalog
        products = [
            ('COFFEE MUG', 'Kitchen'),
            ('WIRELESS HEADPHONES', 'Electronics'),
            ('YOGA MAT', 'Sports'),
            ('NOTEBOOK', 'Office'),
            ('SMARTPHONE CASE', 'Electronics'),
            ('WATER BOTTLE', 'Sports'),
            ('CANDLE', 'Home'),
            ('T-SHIRT', 'Clothing'),
            ('LAPTOP STAND', 'Office'),
            ('FACE MASK', 'Health')
        ]

        records = []
        invoice_no = 100000
        customer_id = 10000

        for date in dates:
            # Random number of transactions per day
            n_transactions = np.random.poisson(50)

            for _ in range(n_transactions):
                # Select random product
                product, category = products[np.random.randint(0, len(products))]

                # Generate transaction
                record = {
                    'InvoiceNo': invoice_no,
                    'StockCode': f'SKU{np.random.randint(1000, 9999)}',
                    'Description': product,
                    'Quantity': np.random.randint(1, 10),
                    'InvoiceDate': date + timedelta(hours=np.random.randint(0, 24)),
                    'UnitPrice': np.random.uniform(5.0, 100.0),
                    'CustomerID': customer_id + np.random.randint(0, 5000),
                    'Country': np.random.choice(['UK', 'Germany', 'France', 'USA']),
                    'Category': category
                }
                records.append(record)
                invoice_no += 1

        df = pd.DataFrame(records)
        df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

        print(f"✓ Generated {len(df):,} sample retail records")
        return df

    def _generate_sample_reviews(self):
        '''Generate sample product reviews'''
        np.random.seed(42)

        products = ['COFFEE MUG', 'WIRELESS HEADPHONES', 'YOGA MAT', 'NOTEBOOK', 'SMARTPHONE CASE']

        positive_reviews = [
            "Great product! Excellent quality and fast shipping.",
            "Love it! Exactly as described and works perfectly.",
            "Amazing quality for the price. Highly recommended!",
            "Perfect! Will definitely buy again.",
            "Outstanding product. Five stars!"
        ]

        negative_reviews = [
            "Poor quality. Broke after a few days.",
            "Not as described. Very disappointed.",
            "Terrible product. Would not recommend.",
            "Cheap material. Not worth the money.",
            "Bad quality control. Arrived damaged."
        ]

        neutral_reviews = [
            "It's okay. Average quality for the price.",
            "Decent product but nothing special.",
            "Works as expected. No complaints.",
            "Good value for money.",
            "Standard quality product."
        ]

        records = []
        for i in range(5000):
            sentiment = np.random.choice(['positive', 'negative', 'neutral'], p=[0.6, 0.2, 0.2])

            if sentiment == 'positive':
                review = np.random.choice(positive_reviews)
                rating = np.random.choice([4, 5], p=[0.3, 0.7])
            elif sentiment == 'negative':
                review = np.random.choice(negative_reviews)
                rating = np.random.choice([1, 2], p=[0.6, 0.4])
            else:
                review = np.random.choice(neutral_reviews)
                rating = 3

            records.append({
                'review_body': review,
                'rating': rating,
                'product': np.random.choice(products),
                'review_date': pd.Timestamp.now() - timedelta(days=np.random.randint(0, 365))
            })

        return pd.DataFrame(records)

    def _generate_sample_trends(self, keywords):
        '''Generate sample trends data'''
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='W')

        data = {'date': dates}
        for keyword in keywords:
            # Generate realistic trend data with seasonality
            trend = 50 + 30 * np.sin(2 * np.pi * np.arange(len(dates)) / 52) + np.random.normal(0, 10, len(dates))
            data[keyword] = np.clip(trend, 0, 100).astype(int)

        return pd.DataFrame(data)

# Convenience function for easy import
def load_all_data():
    '''Load all datasets'''
    loader = DataLoader()

    retail_df = loader.load_uci_retail('data/Online_Retail.xlsx')
    reviews_df = loader.load_amazon_reviews('data/amazon_reviews.csv')
    trends_df = loader.get_google_trends(['coffee', 'headphones', 'yoga'])

    return retail_df, reviews_df, trends_df

if __name__ == "__main__":
    # Test the data loader
    retail, reviews, trends = load_all_data()
    print("\n📊 Data loading complete!")
    print(f"   • Retail data: {len(retail):,} records")
    print(f"   • Reviews: {len(reviews):,} records")
    print(f"   • Trends: {len(trends):,} data points")
