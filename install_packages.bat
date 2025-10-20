@echo off
echo ========================================
echo Stock Prediction System - Package Installer
echo ========================================
echo.

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo.
echo Python found! Installing packages...
echo.

echo Installing core data science packages...
pip install pandas>=1.5.0 numpy>=1.21.0 scikit-learn>=1.1.0

echo.
echo Installing machine learning packages...
pip install tensorflow>=2.10.0 lightgbm>=3.3.0 xgboost>=1.6.0

echo.
echo Installing financial data packages...
pip install yfinance>=0.2.0 pandas-datareader>=0.10.0

echo.
echo Installing web interface...
pip install streamlit>=1.25.0

echo.
echo Installing visualization packages...
pip install plotly>=5.15.0 matplotlib>=3.5.0

echo.
echo Installing utility packages...
pip install requests>=2.28.0 PyPortfolioOpt>=1.5.0 hyperopt>=0.2.7 python-dateutil>=2.8.0 pytz>=2022.1

echo.
echo ========================================
echo Installation completed!
echo ========================================
echo.

echo Testing installation...
python -c "import pandas, numpy, tensorflow, lightgbm, xgboost, streamlit, plotly; print('✅ All packages installed successfully!')"

if %errorlevel% equ 0 (
    echo.
    echo 🎉 Installation successful!
    echo.
    echo You can now run:
    echo   streamlit run stock_prediction_web_interface.py
    echo   python stock_prediction_with_saved_models.py
    echo   streamlit run streamlit_alpha_av.py
) else (
    echo.
    echo ❌ Installation failed. Please check the error messages above.
)

echo.
pause
