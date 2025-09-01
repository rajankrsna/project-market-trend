# 5. config.py - Configuration settings
config_py = """
# Configuration settings for AI Market Trend Analysis

import os
from datetime import datetime

class Config:
    '''Application configuration'''
    
    # Data settings
    DATA_DIR = "data"
    MODELS_DIR = "models"
    OUTPUTS_DIR = "outputs"
    
    # Dataset file paths
    UCI_RETAIL_FILE = os.path.join(DATA_DIR, "Online_Retail.xlsx")
    AMAZON_REVIEWS_FILE = os.path.join(DATA_DIR, "amazon_reviews.csv")
    
    # AI model settings
    PROPHET_FORECAST_PERIODS = 90
    CUSTOMER_SEGMENTS = 4
    ANOMALY_CONTAMINATION = 0.1
    SENTIMENT_SAMPLE_SIZE = 1000
    
    # Dashboard settings
    DASHBOARD_HOST = "127.0.0.1"
    DASHBOARD_PORT = 8050
    DASHBOARD_DEBUG = True
    
    # Google Trends settings
    TRENDS_KEYWORDS = ["coffee", "headphones", "yoga", "smartphone", "laptop"]
    TRENDS_TIMEFRAME = "today 12-m"
    TRENDS_GEO = ""  # Worldwide
    
    # Performance settings
    MAX_REVIEWS_FOR_ANALYSIS = 5000
    BATCH_SIZE_SENTIMENT = 100
    
    # Visualization settings
    CHART_TEMPLATE = "plotly_dark"
    COLOR_PALETTE = {
        'primary': '#00d4ff',
        'secondary': '#ff6b6b', 
        'success': '#51cf66',
        'warning': '#ffd43b',
        'danger': '#ff6b6b',
        'info': '#339af0'
    }
    
    @classmethod
    def create_directories(cls):
        '''Create necessary directories'''
        for directory in [cls.DATA_DIR, cls.MODELS_DIR, cls.OUTPUTS_DIR]:
            os.makedirs(directory, exist_ok=True)
        print(f"✓ Directories created: {cls.DATA_DIR}, {cls.MODELS_DIR}, {cls.OUTPUTS_DIR}")
    
    @classmethod
    def get_timestamp(cls):
        '''Get current timestamp string'''
        return datetime.now().strftime("%Y%m%d_%H%M%S")

# Environment-specific configurations
class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    DASHBOARD_DEBUG = False

# Default configuration
config = DevelopmentConfig()
"""

with open("config.py", "w") as f:
    f.write(config_py)

print("✓ config.py created")