# Stock Prediction System - How to Run the Codes

## 📊 Data Acquisition & Processing
### -streamlit_alpha_av.py
**Purpose:** Web interface for downloading financial data (Equities, Options, FX)
```bash
streamlit run streamlit_alpha_av.py
```

### -equity_processing_pipeline.py
**Purpose:** Process equity data files
```bash
python equity_processing_pipeline.py --input "input_data/COP (Conoco Phillips).csv" --output processing_output/cop_processed.csv
```

### -fx_processing_pipeline.py
**Purpose:** Process foreign exchange data files
```bash
python fx_processing_pipeline.py -i input_data/USD_EUR_daily_full.csv -o processing_output/fx_processed.csv
```

### -option_processing_pipeline.py
**Purpose:** Process options data files
```bash
python option_processing_pipeline.py -i input_data/options.csv -o processing_output/option_processed.csv
```

## 🔧 Feature Engineering
### -equity_feature_engineering.py
**Purpose:** Create engineered features for equity data
```bash
python equity_feature_engineering.py -i "input_data/COP (Conoco Phillips).csv" 
```

### -fx_feature_engining.py
**Purpose:** Create engineered features for FX data
```bash
python fx_feature_engining.py -i "input_data/USD_EUR_daily_full.csv"
```

### -option_feature_engineering.py
**Purpose:** Create engineered features for options data
```bash
python option_feature_engineering.py -i "input_data/options.csv"
```

## 🤖 Machine Learning & Prediction
### -mainM.ipynb
**Purpose:** Jupyter notebook for stock prediction analysis with LSTM, LGBM, and Hybrid models
- Open in Jupyter Notebook or JupyterLab
- Run cells sequentially for complete analysis
- Includes correlation analysis, model training, and visualization

### -stock_prediction_with_saved_models.py
**Purpose:** Enhanced prediction system with model saving/loading functionality
```bash
python stock_prediction_with_saved_models.py
```
**Features:**
- ✅ Saves models after every epoch during training
- ✅ Loads saved models to skip retraining
- ✅ Interactive command-line interface
- ✅ Automatic model versioning with timestamps

### -stock_prediction_web_interface.py
**Purpose:** Web interface to view prediction results using saved models
```bash
streamlit run stock_prediction_web_interface.py
```
**Features:**
- 🌐 Web-based interface (no training required)
- 📊 Interactive graphs and visualizations
- 🔮 Future prediction charts
- 📈 Model comparison displays
- ⚡ Fast predictions using saved models

## 🚀 Complete Workflow

### Step 1: Data Acquisition
```bash
streamlit run streamlit_alpha_av.py
```
- Download stock data, options data, or FX data
- Export CSV files for analysis

### Step 2: Model Training (First Time)
```bash
python stock_prediction_with_saved_models.py
```
- Train LSTM, LGBM, and Hybrid models
- Models are automatically saved for future use
- Choose to use saved models or train new ones

### Step 3: View Results (Any Time)
```bash
streamlit run stock_prediction_web_interface.py
```
- Load saved models instantly
- View interactive prediction results
- No training required - uses previously saved models

## 📁 File Structure
```
stocke_prediction/
├── saved_models/                    # Saved ML models (auto-created)
│   ├── models_PEP_20241215_143022/
│   │   ├── lstm_model/             # LSTM model files
│   │   ├── lgbm_model.txt          # LightGBM model
│   │   ├── xgb_model.json          # XGBoost model
│   │   ├── scaler.pkl              # Data scaler
│   │   ├── model_parameters.json   # Model parameters
│   │   └── performance_metrics.json # Performance metrics
│   └── ...
├── input_data/                     # Raw data files
├── processing_output/              # Processed data files
├── feature_output/                 # Engineered features
└── mainM.ipynb                     # Original analysis notebook
```

## 💡 Tips
- **First run:** Use `stock_prediction_with_saved_models.py` to train and save models
- **Subsequent runs:** Use `stock_prediction_web_interface.py` for instant predictions
- **Data updates:** Use `streamlit_alpha_av.py` to download fresh data
- **Analysis:** Use `mainM.ipynb` for detailed analysis and research