# stock_prediction_web_interface.py
# Web interface to load saved models and display prediction results with graphs

import os
import json
import pickle
import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
import requests
from datetime import datetime, timedelta

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Prediction Results Viewer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class ModelLoader:
    """Class to load saved models and make predictions"""
    
    def __init__(self, save_dir="saved_models"):
        self.save_dir = save_dir
        self.loaded_models = {}
        self.scaler = None
        self.model_params = {}
        self.performance_metrics = {}
    
    def find_latest_models(self, stock_symbol):
        """Find the latest saved models for a given stock symbol"""
        if not os.path.exists(self.save_dir):
            return None
        
        pattern = f"models_{stock_symbol}_"
        model_dirs = [d for d in os.listdir(self.save_dir) if d.startswith(pattern)]
        if not model_dirs:
            return None
        
        # Sort by timestamp and return the latest
        model_dirs.sort(reverse=True)
        latest_dir = os.path.join(self.save_dir, model_dirs[0])
        return latest_dir
    
    def load_all_models_and_parameters(self, model_path):
        """Load all saved models and parameters"""
        loaded_models = {}
        
        # Load LSTM Model
        lstm_path = os.path.join(model_path, "lstm_model")
        if os.path.exists(lstm_path):
            loaded_models['lstm_model'] = tf.keras.models.load_model(lstm_path)
            st.success(f"✅ LSTM model loaded from: {lstm_path}")
        
        # Load LGBM Model
        lgbm_path = os.path.join(model_path, "lgbm_model.txt")
        if os.path.exists(lgbm_path):
            loaded_models['lgbm_model'] = lgb.Booster(model_file=lgbm_path)
            st.success(f"✅ LGBM model loaded from: {lgbm_path}")
        
        # Load XGBoost Model
        xgb_path = os.path.join(model_path, "xgb_model.json")
        if os.path.exists(xgb_path):
            xgb_model = XGBRegressor()
            xgb_model.load_model(xgb_path)
            loaded_models['xgb_model'] = xgb_model
            st.success(f"✅ XGBoost model loaded from: {xgb_path}")
        
        # Load Scaler
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                loaded_models['scaler'] = pickle.load(f)
            st.success(f"✅ Scaler loaded from: {scaler_path}")
        
        # Load Parameters
        params_path = os.path.join(model_path, "model_parameters.json")
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                loaded_models['model_params'] = json.load(f)
            st.success(f"✅ Parameters loaded from: {params_path}")
        
        # Load Performance Metrics
        metrics_path = os.path.join(model_path, "performance_metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                loaded_models['performance_metrics'] = json.load(f)
            st.success(f"✅ Performance metrics loaded from: {metrics_path}")
        
        return loaded_models
    
    def load_models_for_stock(self, stock_symbol):
        """Load models for a specific stock"""
        latest_model_path = self.find_latest_models(stock_symbol)
        if not latest_model_path:
            st.error(f"❌ No saved models found for {stock_symbol}")
            return False
        
        st.info(f"📁 Loading models from: {latest_model_path}")
        self.loaded_models = self.load_all_models_and_parameters(latest_model_path)
        
        if not self.loaded_models:
            st.error(f"❌ Failed to load models for {stock_symbol}")
            return False
        
        self.scaler = self.loaded_models.get('scaler')
        self.model_params = self.loaded_models.get('model_params', {})
        self.performance_metrics = self.loaded_models.get('performance_metrics', {})
        
        return True
    
    def get_sector(self, symbol: str, api_key: str):
        """Get sector information for a stock symbol"""
        try:
            response = requests.get(f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={api_key}")
            if response.status_code != 200:
                return 'API Error'
            info = response.json()[0]
            sector = info['sector']
            return sector
        except:
            return 'Unknown'

class PredictionVisualizer:
    """Class to create visualizations for prediction results"""
    
    def __init__(self, model_loader):
        self.model_loader = model_loader
    
    def create_training_loss_plot(self, history_data=None):
        """Create training loss visualization"""
        if not history_data:
            return None
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history_data.get('loss', []),
            name="Training Loss",
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            y=history_data.get('val_loss', []),
            name="Validation Loss",
            line=dict(color='red')
        ))
        fig.update_layout(
            title='LSTM Model Training Progress',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified'
        )
        return fig
    
    def create_price_prediction_plot(self, actual_prices, predictions, model_name):
        """Create actual vs predicted prices plot"""
        fig = go.Figure()
        
        # Create time index
        time_index = list(range(len(actual_prices)))
        
        fig.add_trace(go.Scatter(
            x=time_index,
            y=actual_prices,
            name="Actual Prices",
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=time_index,
            y=predictions,
            name=f"{model_name} Predictions",
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title=f'{model_name} Model: Actual vs Predicted Prices',
            xaxis_title='Time Step',
            yaxis_title='Price ($)',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    
    def create_future_prediction_plot(self, historical_prices, future_predictions, stock_symbol, n_days):
        """Create future prediction visualization"""
        # Create date range
        last_date = datetime.now()
        historical_dates = [last_date - timedelta(days=i) for i in range(len(historical_prices), 0, -1)]
        future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical_dates,
            y=historical_prices,
            name="Historical Prices",
            line=dict(color='blue', width=2)
        ))
        
        # Future predictions
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=future_predictions,
            name="Future Predictions",
            line=dict(color='orange', width=3, dash='dash')
        ))
        
        # Add vertical line to separate historical and future
        fig.add_vline(
            x=last_date,
            line_dash="dash",
            line_color="gray",
            annotation_text="Today"
        )
        
        fig.update_layout(
            title=f'{stock_symbol} Future Price Predictions ({n_days} days)',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        return fig
    
    def create_model_comparison_plot(self, predictions_dict, stock_symbol):
        """Create comparison plot for different models"""
        fig = go.Figure()
        
        colors = ['red', 'green', 'purple', 'orange']
        for i, (model_name, pred_value) in enumerate(predictions_dict.items()):
            if model_name != 'future_predictions' and isinstance(pred_value, (int, float)):
                fig.add_trace(go.Bar(
                    x=[model_name.upper()],
                    y=[pred_value],
                    name=model_name.upper(),
                    marker_color=colors[i % len(colors)]
                ))
        
        fig.update_layout(
            title=f'{stock_symbol} Model Predictions Comparison',
            xaxis_title='Model',
            yaxis_title='Predicted Price ($)',
            showlegend=False
        )
        return fig

class StockPredictionWebApp:
    """Main web application class"""
    
    def __init__(self):
        self.model_loader = ModelLoader()
        self.visualizer = PredictionVisualizer(self.model_loader)
    
    def get_available_stocks(self):
        """Get list of stocks with saved models"""
        if not os.path.exists(self.model_loader.save_dir):
            return []
        
        stocks = set()
        for dir_name in os.listdir(self.model_loader.save_dir):
            if dir_name.startswith("models_") and "_" in dir_name:
                # Extract stock symbol from directory name
                parts = dir_name.split("_")
                if len(parts) >= 2:
                    stock_symbol = parts[1]
                    stocks.add(stock_symbol)
        
        return sorted(list(stocks))
    
    def make_predictions(self, stock_symbol, start_date, end_date, n_days=5):
        """Make predictions using loaded models"""
        if not self.model_loader.loaded_models:
            st.error("No models loaded. Please load models first.")
            return None
        
        # Download stock data
        try:
            stock_data = yf.download(stock_symbol, start=start_date, end=end_date)["Adj Close"]
            stock_data.fillna(method='ffill', inplace=True)
        except Exception as e:
            st.error(f"Failed to download data for {stock_symbol}: {e}")
            return None
        
        predictions = {}
        
        # Prepare data for prediction
        prices = stock_data.values.reshape(-1, 1)
        if self.model_loader.scaler:
            prices_norm = self.model_loader.scaler.transform(prices)
        else:
            st.error("No scaler found in loaded models")
            return None
        
        # LSTM predictions
        if 'lstm_model' in self.model_loader.loaded_models:
            try:
                X_lstm = prices_norm[-60:].reshape(1, 60, 1)
                lstm_pred = self.model_loader.loaded_models['lstm_model'].predict(X_lstm)
                lstm_pred_denorm = self.model_loader.scaler.inverse_transform(lstm_pred)
                predictions['lstm'] = lstm_pred_denorm[0][0]
            except Exception as e:
                st.warning(f"LSTM prediction failed: {e}")
        
        # LGBM predictions
        if 'lgbm_model' in self.model_loader.loaded_models:
            try:
                X_lgbm = prices_norm[-60:].reshape(1, 60)
                lgbm_pred = self.model_loader.loaded_models['lgbm_model'].predict(X_lgbm)
                predictions['lgbm'] = lgbm_pred[0]
            except Exception as e:
                st.warning(f"LGBM prediction failed: {e}")
        
        # XGBoost predictions
        if 'xgb_model' in self.model_loader.loaded_models:
            try:
                X_xgb = prices_norm[-60:].reshape(1, 60)
                xgb_pred = self.model_loader.loaded_models['xgb_model'].predict(X_xgb)
                predictions['xgb'] = xgb_pred[0]
            except Exception as e:
                st.warning(f"XGBoost prediction failed: {e}")
        
        # Future predictions
        if predictions:
            future_predictions = self.make_future_predictions(predictions, n_days)
            predictions['future_predictions'] = future_predictions
        
        return {
            'predictions': predictions,
            'stock_data': stock_data,
            'prices': prices.flatten()
        }
    
    def make_future_predictions(self, current_predictions, n_days):
        """Make future predictions for n_days"""
        future_preds = []
        
        # Use the most recent prediction as starting point
        if 'lstm' in current_predictions:
            current_pred = current_predictions['lstm']
        elif 'lgbm' in current_predictions:
            current_pred = current_predictions['lgbm']
        elif 'xgb' in current_predictions:
            current_pred = current_predictions['xgb']
        else:
            return []
        
        future_preds.append(current_pred)
        
        # Simple future prediction (you can enhance this)
        for i in range(1, n_days):
            next_pred = current_pred * (1 + np.random.normal(0, 0.01))
            future_preds.append(next_pred)
            current_pred = next_pred
        
        return future_preds
    
    def run(self):
        """Main application runner"""
        st.title("📈 Stock Prediction Results Viewer")
        st.markdown("Load saved models and view prediction results with interactive graphs")
        
        # Sidebar
        with st.sidebar:
            st.header("🔧 Configuration")
            
            # Get available stocks
            available_stocks = self.get_available_stocks()
            
            if not available_stocks:
                st.error("❌ No saved models found!")
                st.info("Please run the training script first to create models.")
                return
            
            # Stock selection
            selected_stock = st.selectbox(
                "Select Stock Symbol",
                available_stocks,
                help="Choose a stock that has saved models"
            )
            
            # Date range
            st.subheader("📅 Date Range")
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                help="Start date for historical data"
            )
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                help="End date for historical data"
            )
            
            # Future prediction days
            n_days = st.number_input(
                "Future Prediction Days",
                min_value=1,
                max_value=30,
                value=5,
                help="Number of days to predict into the future"
            )
            
            # Load models button
            if st.button("🔄 Load Models", type="primary"):
                with st.spinner("Loading models..."):
                    success = self.model_loader.load_models_for_stock(selected_stock)
                    if success:
                        st.session_state.models_loaded = True
                        st.session_state.selected_stock = selected_stock
                    else:
                        st.session_state.models_loaded = False
        
        # Main content
        if st.session_state.get('models_loaded', False):
            st.success(f"✅ Models loaded for {st.session_state.get('selected_stock', 'Unknown')}")
            
            # Make predictions
            if st.button("🔮 Make Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    results = self.make_predictions(
                        st.session_state.get('selected_stock'),
                        start_date,
                        end_date,
                        n_days
                    )
                    
                    if results:
                        st.session_state.prediction_results = results
                        st.success("✅ Predictions completed!")
                    else:
                        st.error("❌ Failed to make predictions")
            
            # Display results
            if st.session_state.get('prediction_results'):
                self.display_results(st.session_state.prediction_results, n_days)
        else:
            st.info("👈 Please select a stock and load models from the sidebar")
    
    def display_results(self, results, n_days):
        """Display prediction results with visualizations"""
        predictions = results['predictions']
        stock_data = results['stock_data']
        prices = results['prices']
        
        st.header("📊 Prediction Results")
        
        # Model predictions comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Model Predictions")
            pred_df = pd.DataFrame([
                {"Model": model.upper(), "Predicted Price": f"${pred:.2f}"}
                for model, pred in predictions.items()
                if model != 'future_predictions' and isinstance(pred, (int, float))
            ])
            st.dataframe(pred_df, use_container_width=True)
        
        with col2:
            st.subheader("📈 Model Comparison")
            comparison_fig = self.visualizer.create_model_comparison_plot(
                predictions, st.session_state.get('selected_stock', 'Unknown')
            )
            st.plotly_chart(comparison_fig, use_container_width=True)
        
        # Future predictions
        if 'future_predictions' in predictions:
            st.subheader("🔮 Future Predictions")
            future_preds = predictions['future_predictions']
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Next {n_days} days predictions:**")
                for i, pred in enumerate(future_preds, 1):
                    st.write(f"Day {i}: ${pred:.2f}")
            
            with col2:
                future_fig = self.visualizer.create_future_prediction_plot(
                    prices[-30:], future_preds, 
                    st.session_state.get('selected_stock', 'Unknown'), n_days
                )
                st.plotly_chart(future_fig, use_container_width=True)
        
        # Performance metrics
        if self.model_loader.performance_metrics:
            st.subheader("📊 Model Performance Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": metric, "Value": f"{value:.4f}"}
                for metric, value in self.model_loader.performance_metrics.items()
            ])
            st.dataframe(metrics_df, use_container_width=True)
        
        # Model parameters
        if self.model_loader.model_params:
            st.subheader("⚙️ Model Parameters")
            with st.expander("View Model Parameters"):
                st.json(self.model_loader.model_params)
        
        # Historical data plot
        st.subheader("📈 Historical Price Data")
        historical_fig = go.Figure()
        historical_fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data.values,
            name="Historical Prices",
            line=dict(color='blue', width=2)
        ))
        historical_fig.update_layout(
            title=f"{st.session_state.get('selected_stock', 'Unknown')} Historical Prices",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified'
        )
        st.plotly_chart(historical_fig, use_container_width=True)

def main():
    """Main function to run the web application"""
    app = StockPredictionWebApp()
    app.run()

if __name__ == "__main__":
    main()
