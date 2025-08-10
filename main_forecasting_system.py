"""
Main Forecasting System
Integrates all components: data collection, factor extraction, VAR modeling, and ML enhancement
Provides comprehensive stock price and return forecasting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import pickle
import os

# Import our custom modules
from data_collector import DataCollector
from technical_factors import TechnicalFactorExtractor
from fundamental_factors import FundamentalFactorExtractor
from sentiment_factors import SentimentFactorExtractor
from var_model import ChincariniVARModel
from ml_models import MLEnhancedForecasting
import config

warnings.filterwarnings('ignore')

class ComprehensiveStockForecaster:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data_collector = DataCollector()
        self.technical_extractor = TechnicalFactorExtractor()
        self.fundamental_extractor = FundamentalFactorExtractor()
        self.sentiment_extractor = SentimentFactorExtractor()
        self.var_model = ChincariniVARModel()
        self.ml_model = MLEnhancedForecasting()
        
        # Storage for processed data and models
        self.raw_data = {}
        self.processed_factors = {}
        self.predictions = {}
        self.model_performance = {}
        
    def collect_all_data(self, force_refresh: bool = False) -> Dict:
        """
        Collect all required data for the ticker
        """
        print(f"Collecting comprehensive data for {self.ticker}...")
        
        # Check if we have cached data
        cache_file = f"data_cache_{self.ticker}.pkl"
        
        if not force_refresh and os.path.exists(cache_file):
            print("Loading cached data...")
            with open(cache_file, 'rb') as f:
                self.raw_data = pickle.load(f)
            return self.raw_data
        
        # Collect fresh data
        ticker_data = self.data_collector.collect_all_data([self.ticker])
        
        if self.ticker in ticker_data:
            self.raw_data = {
                'price_data': ticker_data[self.ticker]['price_data'],
                'fundamental_data': ticker_data[self.ticker]['fundamental_data'],
                'market_data': ticker_data['market_data']
            }
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(self.raw_data, f)
                
            print("Data collection completed and cached!")
        else:
            raise ValueError(f"Failed to collect data for {self.ticker}")
        
        return self.raw_data
    
    def extract_all_factors(self) -> Dict[str, pd.DataFrame]:
        """
        Extract all factors from the collected data
        """
        print("Extracting all factors...")
        
        if not self.raw_data:
            raise ValueError("No data available. Run collect_all_data() first.")
        
        price_data = self.raw_data['price_data']
        fundamental_data = self.raw_data['fundamental_data']
        market_data = self.raw_data['market_data']
        
        # Extract technical factors
        print("Extracting technical factors...")
        technical_indicators = self.technical_extractor.create_technical_factor(price_data)
        technical_factors = self.technical_extractor.create_aggregated_factors(technical_indicators)
        
        # Extract fundamental factors
        print("Extracting fundamental factors...")
        fundamental_indicators = self.fundamental_extractor.create_fundamental_factor(fundamental_data)
        fundamental_factors = self.fundamental_extractor.create_aggregated_fundamental_factors(fundamental_indicators)
        
        # Extract sentiment factors
        print("Extracting sentiment factors...")
        sentiment_indicators = self.sentiment_extractor.create_comprehensive_sentiment_factor(
            self.ticker, price_data, market_data['sentiment_data']
        )
        sentiment_factors = self.sentiment_extractor.create_aggregated_sentiment_factors(sentiment_indicators)
        
        # Store processed factors
        self.processed_factors = {
            'technical_factors': technical_factors,
            'fundamental_factors': fundamental_factors,
            'sentiment_factors': sentiment_factors,
            'technical_indicators': technical_indicators,
            'fundamental_indicators': fundamental_indicators,
            'sentiment_indicators': sentiment_indicators
        }
        
        print("Factor extraction completed!")
        print(f"Technical factors shape: {technical_factors.shape}")
        print(f"Fundamental factors shape: {fundamental_factors.shape}")
        print(f"Sentiment factors shape: {sentiment_factors.shape}")
        
        return self.processed_factors
    
    def build_var_model(self) -> Dict:
        """
        Build and fit the VAR model using extracted factors
        """
        print("Building VAR model...")
        
        if not self.processed_factors:
            raise ValueError("No factors available. Run extract_all_factors() first.")
        
        # Prepare factor data for VAR model
        factor_data = self.var_model.prepare_factor_data(
            technical_factors=self.processed_factors['technical_factors'],
            fundamental_factors=self.processed_factors['fundamental_factors'],
            sentiment_factors=self.processed_factors['sentiment_factors'],
            economic_indicators=self.raw_data['market_data']['economic_indicators']
        )
        
        # Test and ensure stationarity
        stationarity_results = self.var_model.test_stationarity()
        stationary_data = self.var_model.make_stationary()
        
        # Select optimal lags and fit model
        optimal_lags = self.var_model.select_optimal_lags()
        self.var_model.fit_var_model(lags=optimal_lags)
        
        # Model diagnostics
        diagnostics = self.var_model.model_diagnostics()
        
        # Generate forecasts
        var_forecasts = self.var_model.forecast_factors(steps=config.FORECAST_HORIZON)
        
        # Analyze factor interactions
        interactions = self.var_model.analyze_factor_interactions()
        
        var_results = {
            'factor_data': factor_data,
            'stationarity_results': stationarity_results,
            'optimal_lags': optimal_lags,
            'diagnostics': diagnostics,
            'forecasts': var_forecasts,
            'interactions': interactions,
            'model': self.var_model
        }
        
        self.predictions['var'] = var_results
        
        print("VAR model building completed!")
        return var_results
    
    def train_ml_models(self) -> Dict:
        """
        Train ML models using factor data and VAR forecasts
        """
        print("Training ML enhancement models...")
        
        if 'var' not in self.predictions:
            raise ValueError("VAR model not built. Run build_var_model() first.")
        
        # Combine all factors for ML training
        combined_factors = pd.concat([
            self.processed_factors['technical_factors'],
            self.processed_factors['fundamental_factors'], 
            self.processed_factors['sentiment_factors']
        ], axis=1)
        
        # Prepare ML features
        ml_features, target_returns = self.ml_model.prepare_ml_features(
            factor_data=combined_factors,
            price_data=self.raw_data['price_data'],
            var_forecasts=self.predictions['var']['forecasts']
        )
        
        # Train individual models
        rf_metrics = self.ml_model.train_random_forest(ml_features, target_returns)
        xgb_metrics = self.ml_model.train_xgboost(ml_features, target_returns)
        lstm_metrics = self.ml_model.train_lstm(ml_features, target_returns)
        
        # Get model performance summary
        performance_summary = self.ml_model.get_model_performance_summary()
        feature_importance = self.ml_model.get_feature_importance_summary()
        
        ml_results = {
            'ml_features': ml_features,
            'target_returns': target_returns,
            'rf_metrics': rf_metrics,
            'xgb_metrics': xgb_metrics,
            'lstm_metrics': lstm_metrics,
            'performance_summary': performance_summary,
            'feature_importance': feature_importance,
            'model': self.ml_model
        }
        
        self.predictions['ml'] = ml_results
        self.model_performance = performance_summary
        
        print("ML model training completed!")
        return ml_results
    
    def generate_comprehensive_forecast(self, forecast_days: int = 22) -> Dict:
        """
        Generate comprehensive forecasts combining VAR and ML models
        """
        print(f"Generating comprehensive {forecast_days}-day forecast...")
        
        if 'var' not in self.predictions or 'ml' not in self.predictions:
            raise ValueError("Both VAR and ML models must be trained first.")
        
        # Get VAR factor forecasts
        var_forecasts = self.predictions['var']['forecasts']
        
        # Generate ML predictions for each forecast day
        ml_predictions = []
        current_price = self.raw_data['price_data']['Close'].iloc[-1]
        
        # Get the most recent factor values
        recent_factors = pd.concat([
            self.processed_factors['technical_factors'].iloc[-1:],
            self.processed_factors['fundamental_factors'].iloc[-1:],
            self.processed_factors['sentiment_factors'].iloc[-1:]
        ], axis=1)
        
        # Simple forecast generation (in practice, you'd be more sophisticated)
        forecast_dates = pd.date_range(
            start=self.raw_data['price_data'].index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )
        
        # Create forecast DataFrame
        comprehensive_forecast = pd.DataFrame(index=forecast_dates)
        
        # VAR-based predictions (using factor forecasts to predict returns)
        # This is a simplified approach - you'd typically have a more sophisticated mapping
        var_return_signal = var_forecasts.mean(axis=1).rolling(window=5).mean()
        normalized_signal = (var_return_signal - var_return_signal.mean()) / var_return_signal.std()
        
        # Convert to expected returns (simplified)
        var_expected_returns = normalized_signal * 0.02  # Scale to reasonable return range
        
        # ML ensemble predictions
        try:
            # Use the latest features for prediction
            latest_features = self.predictions['ml']['ml_features'].iloc[-1:].fillna(method='ffill')
            ml_prediction = self.ml_model.predict_next_return(latest_features)
            daily_ml_return = ml_prediction.get('ensemble', 0)
        except:
            daily_ml_return = 0
        
        # Combine forecasts
        combined_returns = []
        price_forecasts = []
        confidence_bands = []
        
        current_price_forecast = current_price
        
        for i, date in enumerate(forecast_dates):
            # Combine VAR and ML signals
            if i < len(var_expected_returns):
                var_return = var_expected_returns.iloc[i]
            else:
                var_return = 0
            
            # Weight the predictions (you can optimize these weights)
            combined_return = 0.6 * var_return + 0.4 * daily_ml_return
            
            # Add some decay to the ML prediction over time
            daily_ml_return *= 0.95
            
            combined_returns.append(combined_return)
            
            # Calculate price forecast
            current_price_forecast = current_price_forecast * (1 + combined_return)
            price_forecasts.append(current_price_forecast)
            
            # Simple confidence bands (±2 standard deviations)
            returns_std = self.raw_data['price_data']['Close'].pct_change().std()
            lower_bound = current_price_forecast * (1 - 2 * returns_std)
            upper_bound = current_price_forecast * (1 + 2 * returns_std)
            confidence_bands.append((lower_bound, upper_bound))
        
        # Populate forecast DataFrame
        comprehensive_forecast['Predicted_Price'] = price_forecasts
        comprehensive_forecast['Predicted_Return'] = combined_returns
        comprehensive_forecast['Lower_Bound'] = [cb[0] for cb in confidence_bands]
        comprehensive_forecast['Upper_Bound'] = [cb[1] for cb in confidence_bands]
        
        # Calculate prediction confidence based on model performance
        avg_r2 = self.model_performance['Test_R2'].mean()
        comprehensive_forecast['Confidence_Score'] = avg_r2
        
        # Add buy/sell signals
        current_price = self.raw_data['price_data']['Close'].iloc[-1]
        price_change_pct = (comprehensive_forecast['Predicted_Price'] - current_price) / current_price
        
        comprehensive_forecast['Signal'] = np.where(
            price_change_pct > 0.05, 'Strong Buy',
            np.where(price_change_pct > 0.02, 'Buy',
                    np.where(price_change_pct < -0.05, 'Strong Sell',
                            np.where(price_change_pct < -0.02, 'Sell', 'Hold')))
        )
        
        forecast_results = {
            'forecast_df': comprehensive_forecast,
            'var_contribution': var_expected_returns,
            'ml_contribution': daily_ml_return,
            'current_price': current_price,
            'forecast_horizon': forecast_days,
            'model_confidence': avg_r2
        }
        
        self.predictions['comprehensive'] = forecast_results
        
        print("Comprehensive forecast generated!")
        return forecast_results
    
    def generate_multi_horizon_forecast(self) -> Dict:
        """
        Generate forecasts for multiple time horizons
        """
        horizons = {
            'short_term': 5,    # 1 week
            'mid_term': 22,     # 1 month  
            'long_term': 66     # 3 months
        }
        
        multi_forecasts = {}
        
        for horizon_name, days in horizons.items():
            forecast = self.generate_comprehensive_forecast(forecast_days=days)
            multi_forecasts[horizon_name] = forecast
            
        return multi_forecasts
    
    def get_multi_horizon_recommendations(self) -> Dict:
        """
        Get investment recommendations for different time horizons
        """
        if not hasattr(self, 'multi_forecasts'):
            self.multi_forecasts = self.generate_multi_horizon_forecast()
        
        recommendations = {}
        current_price = self.raw_data['price_data']['Close'].iloc[-1]
        
        for horizon, forecast_data in self.multi_forecasts.items():
            forecast_df = forecast_data['forecast_df']
            
            # Calculate metrics for each horizon
            target_price = forecast_df['Predicted_Price'].iloc[-1]
            expected_return = (target_price - current_price) / current_price
            volatility = forecast_df['Predicted_Return'].std()
            confidence = forecast_df['Confidence_Score'].mean()
            
            # Determine signal
            if expected_return > 0.1:
                signal = "STRONG BUY"
            elif expected_return > 0.05:
                signal = "BUY"
            elif expected_return < -0.1:
                signal = "STRONG SELL"
            elif expected_return < -0.05:
                signal = "SELL"
            else:
                signal = "HOLD"
            
            recommendations[horizon] = {
                'days': len(forecast_df),
                'target_price': target_price,
                'expected_return': expected_return,
                'volatility': volatility,
                'confidence': confidence,
                'signal': signal,
                'forecast_df': forecast_df
            }
        
        return recommendations
    
    def get_investment_recommendation(self) -> Dict:
        """
        Generate investment recommendation based on all analyses
        """
        if 'comprehensive' not in self.predictions:
            raise ValueError("Comprehensive forecast not generated. Run generate_comprehensive_forecast() first.")
        
        forecast = self.predictions['comprehensive']['forecast_df']
        current_price = self.predictions['comprehensive']['current_price']
        
        # Calculate key metrics
        one_week_return = (forecast['Predicted_Price'].iloc[4] - current_price) / current_price
        one_month_return = (forecast['Predicted_Price'].iloc[-1] - current_price) / current_price
        
        avg_confidence = forecast['Confidence_Score'].mean()
        volatility = forecast['Predicted_Return'].std()
        
        # Risk-adjusted return (Sharpe-like ratio)
        risk_adjusted_return = one_month_return / volatility if volatility > 0 else 0
        
        # Overall recommendation logic
        if one_month_return > 0.1 and avg_confidence > 0.6:
            recommendation = "STRONG BUY"
        elif one_month_return > 0.05 and avg_confidence > 0.4:
            recommendation = "BUY"
        elif one_month_return < -0.1 and avg_confidence > 0.6:
            recommendation = "STRONG SELL"
        elif one_month_return < -0.05 and avg_confidence > 0.4:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        investment_rec = {
            'ticker': self.ticker,
            'current_price': current_price,
            'target_price_1w': forecast['Predicted_Price'].iloc[4],
            'target_price_1m': forecast['Predicted_Price'].iloc[-1],
            'expected_return_1w': one_week_return,
            'expected_return_1m': one_month_return,
            'risk_adjusted_return': risk_adjusted_return,
            'volatility': volatility,
            'confidence_score': avg_confidence,
            'recommendation': recommendation,
            'timestamp': datetime.now()
        }
        
        return investment_rec
