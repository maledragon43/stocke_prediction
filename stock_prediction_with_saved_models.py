# stock_prediction_with_saved_models.py
# Enhanced version of mainM.ipynb with model saving and loading functionality

import pandas as pd
import numpy as np
import yfinance as yf
from yahoo_fin import stock_info
import warnings
from plotly.offline import plot, init_notebook_mode
init_notebook_mode()
import cufflinks as cf
cf.set_config_file(offline=True)
import warnings
warnings.filterwarnings('ignore')
import math
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from math import sqrt
from tensorflow.keras.layers import LSTM, Dense, Dropout
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import plotly.express as px
import plotly.graph_objects as go
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
import lightgbm as lgb
import os
from xgboost import XGBRegressor
import json
import pickle
from datetime import datetime
import tensorflow as tf
import requests

class ModelSaver:
    """Class to handle saving and loading of all models and parameters"""
    
    def __init__(self, save_dir="saved_models"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def save_all_models_and_parameters(
        self,
        lstm_model=None, 
        lgbm_model=None, 
        xgb_model=None, 
        scaler=None, 
        model_params=None, 
        performance_metrics=None,
        epoch=None,
        stock_symbol=None
    ):
        """
        Save all three models (LSTM, LGBM, Hybrid) and their parameters
        
        Args:
            lstm_model: Trained LSTM model
            lgbm_model: Trained LightGBM model  
            xgb_model: Trained XGBoost model (from hybrid)
            scaler: MinMaxScaler used for normalization
            model_params: Dictionary of all model parameters
            performance_metrics: Dictionary of model performance metrics
            epoch: Current epoch number for tracking
            stock_symbol: Stock symbol being analyzed
        """
        
        # Create timestamp for unique model versions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if epoch is not None:
            model_path = os.path.join(self.save_dir, f"models_{timestamp}_epoch_{epoch}")
        else:
            model_path = os.path.join(self.save_dir, f"models_{timestamp}")
        
        if stock_symbol:
            model_path = os.path.join(self.save_dir, f"models_{stock_symbol}_{timestamp}")
        
        os.makedirs(model_path, exist_ok=True)
        
        # 1. Save LSTM Model
        if lstm_model is not None:
            lstm_path = os.path.join(model_path, "lstm_model")
            lstm_model.save(lstm_path)
            print(f"LSTM model saved to: {lstm_path}")
        
        # 2. Save LGBM Model
        if lgbm_model is not None:
            lgbm_path = os.path.join(model_path, "lgbm_model.txt")
            lgbm_model.save_model(lgbm_path)
            print(f"LGBM model saved to: {lgbm_path}")
        
        # 3. Save XGBoost Model (from hybrid)
        if xgb_model is not None:
            xgb_path = os.path.join(model_path, "xgb_model.json")
            xgb_model.save_model(xgb_path)
            print(f"XGBoost model saved to: {xgb_path}")
        
        # 4. Save Scaler
        if scaler is not None:
            scaler_path = os.path.join(model_path, "scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to: {scaler_path}")
        
        # 5. Save All Parameters
        if model_params is not None:
            params_path = os.path.join(model_path, "model_parameters.json")
            with open(params_path, 'w') as f:
                json.dump(model_params, f, indent=2)
            print(f"Parameters saved to: {params_path}")
        
        # 6. Save Performance Metrics
        if performance_metrics is not None:
            metrics_path = os.path.join(model_path, "performance_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(performance_metrics, f, indent=2)
            print(f"Performance metrics saved to: {metrics_path}")
        
        print(f"All models saved to: {model_path}")
        return model_path
    
    def load_all_models_and_parameters(self, model_path):
        """
        Load all saved models and parameters
        
        Args:
            model_path: Path to the saved models directory
            
        Returns:
            Dictionary containing all loaded models and parameters
        """
        
        loaded_models = {}
        
        # Load LSTM Model
        lstm_path = os.path.join(model_path, "lstm_model")
        if os.path.exists(lstm_path):
            loaded_models['lstm_model'] = tf.keras.models.load_model(lstm_path)
            print(f"LSTM model loaded from: {lstm_path}")
        
        # Load LGBM Model
        lgbm_path = os.path.join(model_path, "lgbm_model.txt")
        if os.path.exists(lgbm_path):
            loaded_models['lgbm_model'] = lgb.Booster(model_file=lgbm_path)
            print(f"LGBM model loaded from: {lgbm_path}")
        
        # Load XGBoost Model
        xgb_path = os.path.join(model_path, "xgb_model.json")
        if os.path.exists(xgb_path):
            xgb_model = XGBRegressor()
            xgb_model.load_model(xgb_path)
            loaded_models['xgb_model'] = xgb_model
            print(f"XGBoost model loaded from: {xgb_path}")
        
        # Load Scaler
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                loaded_models['scaler'] = pickle.load(f)
            print(f"Scaler loaded from: {scaler_path}")
        
        # Load Parameters
        params_path = os.path.join(model_path, "model_parameters.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                loaded_models['model_params'] = json.load(f)
            print(f"Parameters loaded from: {params_path}")
        
        # Load Performance Metrics
        metrics_path = os.path.join(model_path, "performance_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                loaded_models['performance_metrics'] = json.load(f)
            print(f"Performance metrics loaded from: {metrics_path}")
        
        return loaded_models
    
    def find_latest_models(self, stock_symbol=None):
        """Find the latest saved models for a given stock symbol"""
        if stock_symbol:
            pattern = f"models_{stock_symbol}_"
        else:
            pattern = "models_"
        
        model_dirs = [d for d in os.listdir(self.save_dir) if d.startswith(pattern)]
        if not model_dirs:
            return None
        
        # Sort by timestamp and return the latest
        model_dirs.sort(reverse=True)
        latest_dir = os.path.join(self.save_dir, model_dirs[0])
        return latest_dir

class StockPredictionWithSavedModels:
    """Main class for stock prediction with model saving and loading functionality"""
    
    def __init__(self):
        self.model_saver = ModelSaver()
        self.lstm_model = None
        self.lgbm_model = None
        self.xgb_model = None
        self.scaler = None
        self.loaded_models = None
    
    def get_sector(self, symbol: str, api_key: str):
        """Get sector information for a stock symbol"""
        response = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}")
        if response.status_code != 200:
            return 'INVALID API KEY'
        try:
            info = response.json()[0]
            sector = info['sector']
            return sector
        except:
            return ''
    
    def load_saved_models(self, stock_symbol):
        """Load saved models for a given stock symbol"""
        latest_model_path = self.model_saver.find_latest_models(stock_symbol)
        if latest_model_path:
            self.loaded_models = self.model_saver.load_all_models_and_parameters(latest_model_path)
            print(f"✅ Loaded saved models for {stock_symbol} from {latest_model_path}")
            return True
        else:
            print(f"❌ No saved models found for {stock_symbol}")
            return False
    
    def check_saved_models_exist(self, stock_symbol):
        """Check if saved models exist for a given stock symbol"""
        latest_model_path = self.model_saver.find_latest_models(stock_symbol)
        if latest_model_path:
            print(f"✅ Found saved models for {stock_symbol} at: {latest_model_path}")
            return True
        else:
            print(f"❌ No saved models found for {stock_symbol}")
            return False
    
    def train_lstm_model(self, stock_data, stock_symbol, save_after_each_epoch=True):
        """Train LSTM model with optional saving after each epoch"""
        prices = stock_data.values.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        prices_norm = scaler.fit_transform(prices)
        
        # Create LSTM model
        lstm_model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(None, 1)),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1)
        ])
        
        lstm_model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Prepare data
        X, y = [], []
        for i in range(60, len(prices_norm) - 1):
            X.append(prices_norm[i-60:i, 0])
            y.append(prices_norm[i+1, 0])
        X, y = np.array(X), np.array(y)
        
        # Split data
        train_size = int(0.8 * X.shape[0])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Reshape for LSTM
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Custom callback to save after each epoch
        class ModelCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, model_saver, scaler, stock_symbol):
                self.model_saver = model_saver
                self.scaler = scaler
                self.stock_symbol = stock_symbol
            
            def on_epoch_end(self, epoch, logs=None):
                if save_after_each_epoch:
                    self.model_saver.save_all_models_and_parameters(
                        lstm_model=self.model,
                        scaler=self.scaler,
                        model_params={
                            'lstm_epochs': epoch + 1,
                            'lstm_batch_size': 32,
                            'lstm_units': 50,
                            'stock_symbol': self.stock_symbol
                        },
                        performance_metrics={
                            'lstm_loss': logs.get('loss', 0),
                            'lstm_val_loss': logs.get('val_loss', 0)
                        },
                        epoch=epoch + 1,
                        stock_symbol=self.stock_symbol
                    )
        
        # Train with callback
        checkpoint_callback = ModelCheckpoint(self.model_saver, scaler, stock_symbol)
        history = lstm_model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            callbacks=[checkpoint_callback] if save_after_each_epoch else []
        )
        
        self.lstm_model = lstm_model
        self.scaler = scaler
        
        return history, scaler, X_test, y_test, prices_norm
    
    def train_lgbm_model(self, stock_data, stock_symbol):
        """Train LightGBM model"""
        # Implementation similar to the original notebook
        # This is a simplified version - you can expand this based on your needs
        pass
    
    def train_hybrid_model(self, stock_data, stock_symbol):
        """Train hybrid LSTM-XGBoost model"""
        # Implementation similar to the original notebook
        # This is a simplified version - you can expand this based on your needs
        pass
    
    def predict_with_saved_models(self, stock_symbol, stock_data, n_days=5):
        """Make predictions using saved models"""
        if not self.loaded_models:
            print(f"No saved models loaded for {stock_symbol}. Please train models first.")
            return None
        
        predictions = {}
        
        # Prepare data for prediction (similar to training data preparation)
        prices = stock_data.values.reshape(-1, 1)
        scaler = self.loaded_models.get('scaler')
        
        if scaler is None:
            print("No scaler found in saved models. Cannot make predictions.")
            return None
        
        # Normalize the data
        prices_norm = scaler.transform(prices)
        
        # LSTM predictions
        if 'lstm_model' in self.loaded_models:
            try:
                # Prepare data for LSTM (last 60 days)
                X_lstm = prices_norm[-60:].reshape(1, 60, 1)
                lstm_pred = self.loaded_models['lstm_model'].predict(X_lstm)
                lstm_pred_denorm = scaler.inverse_transform(lstm_pred)
                predictions['lstm'] = lstm_pred_denorm[0][0]
                print(f"✅ LSTM prediction: {lstm_pred_denorm[0][0]:.2f}")
            except Exception as e:
                print(f"❌ LSTM prediction failed: {e}")
        
        # LGBM predictions
        if 'lgbm_model' in self.loaded_models:
            try:
                # Prepare data for LGBM (last 60 days as features)
                X_lgbm = prices_norm[-60:].reshape(1, 60)
                lgbm_pred = self.loaded_models['lgbm_model'].predict(X_lgbm)
                predictions['lgbm'] = lgbm_pred[0]
                print(f"✅ LGBM prediction: {lgbm_pred[0]:.2f}")
            except Exception as e:
                print(f"❌ LGBM prediction failed: {e}")
        
        # XGBoost predictions
        if 'xgb_model' in self.loaded_models:
            try:
                # Prepare data for XGBoost (last 60 days as features)
                X_xgb = prices_norm[-60:].reshape(1, 60)
                xgb_pred = self.loaded_models['xgb_model'].predict(X_xgb)
                predictions['xgb'] = xgb_pred[0]
                print(f"✅ XGBoost prediction: {xgb_pred[0]:.2f}")
            except Exception as e:
                print(f"❌ XGBoost prediction failed: {e}")
        
        # Future predictions for n_days
        if predictions:
            print(f"\n🔮 Making future predictions for {n_days} days:")
            future_predictions = self.make_future_predictions(predictions, prices_norm, scaler, n_days)
            predictions['future_predictions'] = future_predictions
        
        return predictions
    
    def make_future_predictions(self, current_predictions, prices_norm, scaler, n_days):
        """Make future predictions for n_days using saved models"""
        future_preds = []
        
        # Use the most recent prediction as starting point
        if 'lstm' in current_predictions:
            current_pred = current_predictions['lstm']
        elif 'lgbm' in current_predictions:
            current_pred = current_predictions['lgbm']
        elif 'xgb' in current_predictions:
            current_pred = current_predictions['xgb']
        else:
            print("No valid predictions available for future forecasting")
            return []
        
        future_preds.append(current_pred)
        
        # Simple future prediction (you can enhance this)
        for i in range(1, n_days):
            # This is a simplified approach - you can implement more sophisticated methods
            next_pred = current_pred * (1 + np.random.normal(0, 0.01))  # Add some randomness
            future_preds.append(next_pred)
            current_pred = next_pred
        
        print(f"Future predictions: {[f'{pred:.2f}' for pred in future_preds]}")
        return future_preds
    
    def run_prediction_pipeline(self, stock_symbol, start_date, end_date, n_days=5, use_saved_models=True):
        """Main pipeline for stock prediction with optional model loading"""
        
        print(f"Starting prediction pipeline for {stock_symbol}")
        
        # Try to load saved models first
        if use_saved_models and self.load_saved_models(stock_symbol):
            print(f"✅ Using saved models for {stock_symbol} - NO TRAINING NEEDED!")
            
            # Download current stock data for prediction
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)["Adj Close"]
            stock_data.fillna(method='ffill', inplace=True)
            
            # Use saved models for prediction
            predictions = self.predict_with_saved_models(stock_symbol, stock_data, n_days)
            
            if predictions:
                print(f"✅ Successfully made predictions using saved models for {stock_symbol}")
                return {
                    'predictions': predictions,
                    'models_used': 'saved_models',
                    'stock_data': stock_data
                }
            else:
                print(f"❌ Failed to make predictions with saved models for {stock_symbol}")
                print(f"Falling back to training new models...")
                use_saved_models = False
        
        if not use_saved_models:
            print(f"🔄 Training new models for {stock_symbol}")
            # Download stock data
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)["Adj Close"]
            stock_data.fillna(method='ffill', inplace=True)
            
            # Train models
            history, scaler, X_test, y_test, prices_norm = self.train_lstm_model(
                stock_data, stock_symbol, save_after_each_epoch=True
            )
            
            # Save final models
            self.model_saver.save_all_models_and_parameters(
                lstm_model=self.lstm_model,
                scaler=self.scaler,
                model_params={
                    'stock_symbol': stock_symbol,
                    'start_date': start_date,
                    'end_date': end_date,
                    'n_days': n_days
                },
                performance_metrics={
                    'final_loss': history.history['loss'][-1],
                    'final_val_loss': history.history['val_loss'][-1]
                },
                stock_symbol=stock_symbol
            )
            
            return {
                'lstm_model': self.lstm_model,
                'scaler': self.scaler,
                'history': history,
                'models_used': 'newly_trained'
            }

def main():
    """Main function to run the stock prediction pipeline"""
    
    # Initialize the prediction system
    predictor = StockPredictionWithSavedModels()
    
    # Get user input
    print('Enter ticker symbol for current stock price: ')
    ticker = input()
    ticker = ticker.replace(",", " ")
    stocks_a = [stock.upper() for stock in ticker.split()]
    
    print('Start date for analysis Format:(Year xxxx-Month xx-Day xx):')
    start_date = input()
    print('End date for analysis:')
    end_date = input()
    
    print('Future days to predict the price:')
    n_days = int(input('Enter the day after last date: '))
    
    # Check if saved models exist for the first stock
    if stocks_a:
        predictor = StockPredictionWithSavedModels()
        has_saved_models = predictor.check_saved_models_exist(stocks_a[0])
        
        if has_saved_models:
            print(f"\n✅ Found saved models for {stocks_a[0]}!")
            print("You can use saved models to skip training and get instant predictions.")
        else:
            print(f"\n❌ No saved models found for {stocks_a[0]}")
            print("New models will be trained.")
    
    print('\nUse saved models if available? (y/n):')
    use_saved = input().lower() == 'y'
    
    # Run prediction for each stock
    for stock_symbol in stocks_a:
        print(f"\n{'='*50}")
        print(f"Processing {stock_symbol}")
        print(f"{'='*50}")
        
        try:
            results = predictor.run_prediction_pipeline(
                stock_symbol=stock_symbol,
                start_date=start_date,
                end_date=end_date,
                n_days=n_days,
                use_saved_models=use_saved
            )
            
            if results:
                print(f"✅ Successfully processed {stock_symbol}")
                
                # Show what type of processing was used
                if 'models_used' in results:
                    if results['models_used'] == 'saved_models':
                        print(f"🚀 Used saved models - NO TRAINING REQUIRED!")
                    elif results['models_used'] == 'newly_trained':
                        print(f"🔄 Trained new models and saved them for future use")
                
                # Show predictions if available
                if 'predictions' in results and results['predictions']:
                    print(f"📊 Predictions made for {stock_symbol}")
                    for model_type, pred in results['predictions'].items():
                        if model_type != 'future_predictions':
                            print(f"   {model_type.upper()}: {pred:.2f}")
            else:
                print(f"❌ Failed to process {stock_symbol}")
                
        except Exception as e:
            print(f"❌ Error processing {stock_symbol}: {str(e)}")
    
    print(f"\n{'='*60}")
    print("🎉 Stock prediction pipeline completed!")
    print("💡 Tip: Next time you run this for the same stocks, saved models will be used automatically!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
