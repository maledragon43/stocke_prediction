# 📦 Installation Guide - Stock Prediction System

## 🚀 Quick Installation

### Option 1: Install All Packages at Once
```bash
pip install -r requirements.txt
```

### Option 2: Install Packages Individually
```bash
# Core Data Science Libraries
pip install pandas>=1.5.0 numpy>=1.21.0 scikit-learn>=1.1.0

# Machine Learning Libraries
pip install tensorflow>=2.10.0 lightgbm>=3.3.0 xgboost>=1.6.0

# Financial Data Libraries
pip install yfinance>=0.2.0 pandas-datareader>=0.10.0

# Web Interface
pip install streamlit>=1.25.0

# Visualization Libraries
pip install plotly>=5.15.0 matplotlib>=3.5.0

# HTTP Requests
pip install requests>=2.28.0

# Portfolio Optimization
pip install PyPortfolioOpt>=1.5.0

# Hyperparameter Optimization
pip install hyperopt>=0.2.7

# Additional Utilities
pip install python-dateutil>=2.8.0 pytz>=2022.1
```

## 🔧 System Requirements

### Python Version
- **Python 3.8+** (Recommended: Python 3.9 or 3.10)
- **64-bit Python** (Required for TensorFlow)

### Operating System
- **Windows 10/11** ✅
- **macOS 10.14+** ✅
- **Linux (Ubuntu 18.04+)** ✅

### Hardware Requirements
- **RAM:** 8GB+ (16GB recommended for large datasets)
- **Storage:** 2GB+ free space
- **CPU:** Multi-core processor recommended

## 📋 Package Details

### Core Data Science
| Package | Version | Purpose |
|---------|---------|---------|
| `pandas` | >=1.5.0 | Data manipulation and analysis |
| `numpy` | >=1.21.0 | Numerical computing |
| `scikit-learn` | >=1.1.0 | Machine learning utilities |

### Machine Learning
| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | >=2.10.0 | Deep learning (LSTM models) |
| `lightgbm` | >=3.3.0 | Gradient boosting (LGBM models) |
| `xgboost` | >=1.6.0 | Gradient boosting (XGBoost models) |

### Financial Data
| Package | Version | Purpose |
|---------|---------|---------|
| `yfinance` | >=0.2.0 | Yahoo Finance data |
| `pandas-datareader` | >=0.10.0 | Financial data sources |

### Web Interface
| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | >=1.25.0 | Web application framework |

### Visualization
| Package | Version | Purpose |
|---------|---------|---------|
| `plotly` | >=5.15.0 | Interactive charts |
| `matplotlib` | >=3.5.0 | Static plots |

### Utilities
| Package | Version | Purpose |
|---------|---------|---------|
| `requests` | >=2.28.0 | HTTP requests |
| `PyPortfolioOpt` | >=1.5.0 | Portfolio optimization |
| `hyperopt` | >=0.2.7 | Hyperparameter optimization |

## 🛠️ Installation Steps

### Step 1: Check Python Version
```bash
python --version
# Should be Python 3.8 or higher
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv stock_prediction_env

# Activate virtual environment
# Windows:
stock_prediction_env\Scripts\activate
# macOS/Linux:
source stock_prediction_env/bin/activate
```

### Step 3: Install Packages
```bash
# Install all packages
pip install -r requirements.txt

# Or install individually (see Option 2 above)
```

### Step 4: Verify Installation
```bash
# Test imports
python -c "import pandas, numpy, tensorflow, lightgbm, xgboost, streamlit, plotly; print('All packages installed successfully!')"
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. TensorFlow Installation Issues
```bash
# If TensorFlow fails to install:
pip install tensorflow-cpu  # For CPU-only version
# OR
pip install tensorflow-gpu  # For GPU version (requires CUDA)
```

#### 2. LightGBM Installation Issues
```bash
# If LightGBM fails on Windows:
pip install --only-binary=all lightgbm
# OR
conda install lightgbm
```

#### 3. XGBoost Installation Issues
```bash
# If XGBoost fails:
pip install xgboost --no-cache-dir
# OR
conda install xgboost
```

#### 4. Streamlit Issues
```bash
# If Streamlit has issues:
pip install streamlit --upgrade
```

### Platform-Specific Issues

#### Windows
- **Issue:** Microsoft Visual C++ 14.0 required
- **Solution:** Install Visual Studio Build Tools or use conda

#### macOS
- **Issue:** Xcode command line tools required
- **Solution:** `xcode-select --install`

#### Linux
- **Issue:** Missing system dependencies
- **Solution:** 
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-dev build-essential

# CentOS/RHEL
sudo yum install python3-devel gcc gcc-c++
```

## 🧪 Testing Installation

### Test Script
Create a file called `test_installation.py`:

```python
#!/usr/bin/env python3
"""
Test script to verify all packages are installed correctly
"""

def test_imports():
    """Test all required imports"""
    try:
        import pandas as pd
        import numpy as np
        import tensorflow as tf
        import lightgbm as lgb
        import xgboost as xgb
        import streamlit as st
        import plotly.graph_objects as go
        import yfinance as yf
        import requests
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        from pypfopt import EfficientFrontier
        from hyperopt import fmin, tpe, hp
        
        print("✅ All packages imported successfully!")
        print(f"Pandas version: {pd.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"TensorFlow version: {tf.__version__}")
        print(f"Streamlit version: {st.__version__}")
        print(f"Plotly version: {go.__version__}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

if __name__ == "__main__":
    test_imports()
```

Run the test:
```bash
python test_installation.py
```

## 🚀 Running the Applications

### 1. Data Acquisition (Streamlit)
```bash
streamlit run streamlit_alpha_av.py
```

### 2. Model Training
```bash
python stock_prediction_with_saved_models.py
```

### 3. Web Interface
```bash
streamlit run stock_prediction_web_interface.py
```

## 📞 Support

If you encounter issues:

1. **Check Python version:** Must be 3.8+
2. **Check virtual environment:** Make sure it's activated
3. **Check package versions:** Use `pip list` to verify
4. **Run test script:** Use the test script above
5. **Check error messages:** Look for specific package errors

## 🔄 Alternative Installation Methods

### Using Conda
```bash
# Create conda environment
conda create -n stock_prediction python=3.9

# Activate environment
conda activate stock_prediction

# Install packages
conda install pandas numpy scikit-learn matplotlib
pip install tensorflow lightgbm xgboost streamlit plotly yfinance
```

### Using Docker
```bash
# Create Dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["streamlit", "run", "stock_prediction_web_interface.py"]
```

## ✅ Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All packages installed successfully
- [ ] Test script runs without errors
- [ ] Streamlit applications start correctly
- [ ] No import errors in any Python files

**You're ready to use the Stock Prediction System!** 🎉
