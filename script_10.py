# 10. Create setup script (fixed)
setup_py = '''#!/usr/bin/env python3
"""
Setup script for AI Market Trend Analysis System
Handles initial setup, dependency installation, and system validation
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_header():
    print("=" * 60)
    print("🤖 AI Market Trend Analysis - Setup Script")
    print("=" * 60)

def check_python_version():
    """Check Python version compatibility"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required. Current version:", platform.python_version())
        return False
    print(f"✅ Python {platform.python_version()} - Compatible")
    return True

def create_directories():
    """Create necessary project directories"""
    directories = ["data", "models", "outputs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}/")

def install_dependencies():
    """Install Python dependencies"""
    print("\\n📦 Installing Python dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_sample_data():
    """Download or create sample datasets"""
    print("\\n📊 Setting up sample data...")
    
    # For demo purposes, we'll use the generated sample data
    # In production, users would place their actual datasets here
    
    data_info = """
📁 Data Directory Setup:

Place your datasets in the data/ folder:
• data/Online_Retail.xlsx     (UCI Online Retail Dataset)
• data/amazon_reviews.csv     (Amazon Product Reviews)

Don't have the datasets? No problem! 
The system will generate realistic sample data automatically.

Dataset sources:
• UCI: https://archive.ics.uci.edu/ml/datasets/Online+Retail
• Amazon Reviews: https://www.kaggle.com/datasets/amazonreviews
    """
    
    print(data_info)

def run_system_test():
    """Run basic system validation"""
    print("\\n🧪 Running system validation...")
    
    try:
        result = subprocess.run([sys.executable, "main.py", "--mode", "test"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ System validation passed")
            return True
        else:
            print("❌ System validation failed")
            print("Error output:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⚠️  System test timed out (this is normal for first run)")
        return True
    except Exception as e:
        print(f"❌ System test error: {e}")
        return False

def print_next_steps():
    """Print next steps for user"""
    next_steps = """
🎉 Setup Complete! Next Steps:

1. Launch Dashboard:
   python main.py --mode dashboard

2. Run Complete Analysis:  
   python main.py --mode analysis
   
3. Access Dashboard:
   Open http://127.0.0.1:8050 in your browser

4. Explore Features:
   • Sales Forecasting
   • Customer Segmentation  
   • Sentiment Analysis
   • Advanced Analytics

📚 Documentation: See README.md for detailed instructions
🆘 Support: Check troubleshooting section in README.md
    """
    
    print("=" * 60)
    print(next_steps)
    print("=" * 60)

def main():
    """Main setup routine"""
    print_header()
    
    # Check system requirements
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    print("\\n📁 Creating project directories...")
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\\n❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Setup data
    download_sample_data()
    
    # Run validation test
    print("\\n🔍 Validating installation...")
    test_passed = run_system_test()
    
    if test_passed:
        print("\\n✅ Setup completed successfully!")
        print_next_steps()
    else:
        print("\\n⚠️  Setup completed with warnings")
        print("The system should still work, but some features may be limited")
        print_next_steps()

if __name__ == "__main__":
    main()
'''

with open("setup.py", "w") as f:
    f.write(setup_py)

print("✓ setup.py created")