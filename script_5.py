# 6. utils.py - Utility functions
utils_py = """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import pickle
import json
import os
from config import config

def save_model(model, filename):
    '''Save ML model to disk'''
    filepath = os.path.join(config.MODELS_DIR, f"{filename}.pkl")
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved: {filepath}")

def load_model(filename):
    '''Load ML model from disk'''
    filepath = os.path.join(config.MODELS_DIR, f"{filename}.pkl")
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"✓ Model loaded: {filepath}")
        return model
    except FileNotFoundError:
        print(f"❌ Model not found: {filepath}")
        return None

def save_results(results, filename):
    '''Save analysis results to JSON'''
    filepath = os.path.join(config.OUTPUTS_DIR, f"{filename}_{config.get_timestamp()}.json")
    
    # Convert numpy arrays and pandas objects to serializable format
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return str(obj)
    
    serializable_results = {}
    for key, value in results.items():
        try:
            serializable_results[key] = convert_for_json(value)
        except:
            serializable_results[key] = str(value)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"✓ Results saved: {filepath}")

def generate_eda_report(df, output_file="eda_report.html"):
    '''Generate exploratory data analysis report'''
    print("📊 Generating EDA report...")
    
    # Basic statistics
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': df.describe().to_dict()
    }
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Exploratory Data Analysis', fontsize=16)
    
    # Plot 1: Missing values
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        axes[0, 0].bar(range(len(missing_data)), missing_data.values)
        axes[0, 0].set_xticks(range(len(missing_data)))
        axes[0, 0].set_xticklabels(missing_data.index, rotation=45)
        axes[0, 0].set_title('Missing Values by Column')
        axes[0, 0].set_ylabel('Count')
    else:
        axes[0, 0].text(0.5, 0.5, 'No Missing Values', ha='center', va='center')
        axes[0, 0].set_title('Missing Values by Column')
    
    # Plot 2: Data types distribution
    dtype_counts = df.dtypes.value_counts()
    axes[0, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
    axes[0, 1].set_title('Data Types Distribution')
    
    # Plot 3: Numeric columns distribution
    numeric_cols = df.select_dtypes(include=[np.number]).columns[:4]  # First 4 numeric columns
    if len(numeric_cols) > 0:
        df[numeric_cols].hist(bins=20, ax=axes[1, 0])
        axes[1, 0].set_title('Numeric Columns Distribution')
    else:
        axes[1, 0].text(0.5, 0.5, 'No Numeric Columns', ha='center', va='center')
        axes[1, 0].set_title('Numeric Columns Distribution')
    
    # Plot 4: Correlation matrix for numeric columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=axes[1, 1])
        axes[1, 1].set_title('Correlation Matrix')
    else:
        axes[1, 1].text(0.5, 0.5, 'Insufficient Numeric Data', ha='center', va='center')
        axes[1, 1].set_title('Correlation Matrix')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(config.OUTPUTS_DIR, f"eda_plots_{config.get_timestamp()}.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ EDA report generated: {plot_file}")
    return stats

def validate_data_quality(df, required_columns=None):
    '''Validate data quality and return quality score'''
    quality_score = 100
    issues = []
    
    # Check for required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            quality_score -= 20
            issues.append(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
    if missing_pct > 10:
        quality_score -= 15
        issues.append(f"High missing data percentage: {missing_pct:.2f}%")
    elif missing_pct > 5:
        quality_score -= 5
        issues.append(f"Moderate missing data: {missing_pct:.2f}%")
    
    # Check for duplicates
    duplicate_pct = (df.duplicated().sum() / len(df)) * 100
    if duplicate_pct > 5:
        quality_score -= 10
        issues.append(f"High duplicate percentage: {duplicate_pct:.2f}%")
    
    # Check data consistency
    for col in df.select_dtypes(include=[np.number]).columns:
        if (df[col] < 0).any():
            quality_score -= 5
            issues.append(f"Negative values found in {col}")
    
    print(f"📊 Data Quality Score: {quality_score}/100")
    if issues:
        print("⚠️  Quality Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("✅ No data quality issues found")
    
    return quality_score, issues

def calculate_business_metrics(retail_df):
    '''Calculate key business metrics'''
    metrics = {}
    
    # Revenue metrics
    metrics['total_revenue'] = retail_df['TotalPrice'].sum()
    metrics['average_order_value'] = retail_df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    metrics['revenue_per_customer'] = retail_df.groupby('CustomerID')['TotalPrice'].sum().mean()
    
    # Customer metrics
    metrics['total_customers'] = retail_df['CustomerID'].nunique()
    metrics['total_orders'] = retail_df['InvoiceNo'].nunique()
    metrics['repeat_customer_rate'] = (retail_df.groupby('CustomerID')['InvoiceNo'].nunique() > 1).mean()
    
    # Product metrics
    metrics['total_products'] = retail_df['Description'].nunique()
    metrics['avg_items_per_order'] = retail_df.groupby('InvoiceNo')['Quantity'].sum().mean()
    
    # Time-based metrics
    date_range = retail_df['InvoiceDate'].max() - retail_df['InvoiceDate'].min()
    metrics['data_span_days'] = date_range.days
    metrics['avg_daily_revenue'] = metrics['total_revenue'] / metrics['data_span_days']
    
    return metrics

def export_dashboard_data(ai_results, filename="dashboard_data"):
    '''Export data for dashboard in JSON format'''
    export_data = {}
    
    # Process AI results for JSON export
    for key, value in ai_results.items():
        if isinstance(value, pd.DataFrame):
            export_data[key] = value.to_dict('records')
        elif isinstance(value, dict):
            export_data[key] = {}
            for k, v in value.items():
                if isinstance(v, pd.DataFrame):
                    export_data[key][k] = v.to_dict('records')
                elif isinstance(v, pd.Series):
                    export_data[key][k] = v.to_dict()
                else:
                    export_data[key][k] = str(v)
        else:
            export_data[key] = str(value)
    
    # Save to file
    filepath = os.path.join(config.OUTPUTS_DIR, f"{filename}_{config.get_timestamp()}.json")
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    print(f"✓ Dashboard data exported: {filepath}")
    return filepath

def performance_timer(func):
    '''Decorator to measure function execution time'''
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        print(f"⏱️  {func.__name__} executed in {duration:.2f} seconds")
        return result
    return wrapper

class DataValidator:
    '''Class for comprehensive data validation'''
    
    @staticmethod
    def validate_retail_data(df):
        '''Validate retail/e-commerce data format'''
        required_cols = ['InvoiceNo', 'Description', 'Quantity', 'InvoiceDate', 'UnitPrice', 'CustomerID']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"❌ Missing required columns: {missing_cols}")
            return False
        
        # Validate data types and ranges
        if df['Quantity'].min() < 0:
            print("⚠️  Warning: Negative quantities found")
        
        if df['UnitPrice'].min() < 0:
            print("⚠️  Warning: Negative prices found")
        
        if df['CustomerID'].isnull().sum() > len(df) * 0.1:
            print("⚠️  Warning: High percentage of missing customer IDs")
        
        print("✅ Retail data validation passed")
        return True
    
    @staticmethod
    def validate_reviews_data(df):
        '''Validate reviews data format'''
        if 'review_body' not in df.columns:
            print("❌ Missing 'review_body' column in reviews data")
            return False
        
        if df['review_body'].isnull().sum() > 0:
            print("⚠️  Warning: Some reviews have missing text")
        
        print("✅ Reviews data validation passed")
        return True

if __name__ == "__main__":
    print("🔧 Utility functions module loaded successfully!")
    
    # Create necessary directories
    config.create_directories()
"""

with open("utils.py", "w") as f:
    f.write(utils_py)

print("✓ utils.py created")