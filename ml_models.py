"""
Machine Learning Enhancement Module
Combines VAR forecasts with ML models for improved prediction accuracy
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
from typing import Dict, List, Tuple, Optional
import warnings
import config

warnings.filterwarnings('ignore')

class MLEnhancedForecasting:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # ML settings from config
        self.test_size = config.ML_SETTINGS['test_size']
        self.random_state = config.ML_SETTINGS['random_state']
        self.lstm_epochs = config.ML_SETTINGS['lstm_epochs']
        self.lstm_batch_size = config.ML_SETTINGS['lstm_batch_size']
        self.xgb_n_estimators = config.ML_SETTINGS['xgb_n_estimators']
        
    def prepare_ml_features(self, factor_data: pd.DataFrame, 
                          price_data: pd.DataFrame,
                          var_forecasts: pd.DataFrame = None,
                          lookback_window: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for ML models including VAR forecasts, factor data, and technical indicators
        """
        print("Preparing ML features...")
        
        # Calculate target variable (future returns)
        returns = price_data['Close'].pct_change().shift(-1)  # Next day return
        
        # Prepare feature matrix
        features = pd.DataFrame(index=factor_data.index)
        
        # Add factor data
        for col in factor_data.columns:
            features[col] = factor_data[col]
        
        # Add lagged features
        for lag in [1, 2, 3, 5, 10]:
            for col in factor_data.columns:
                features[f'{col}_lag_{lag}'] = factor_data[col].shift(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            for col in factor_data.columns:
                features[f'{col}_ma_{window}'] = factor_data[col].rolling(window=window).mean()
                features[f'{col}_std_{window}'] = factor_data[col].rolling(window=window).std()
        
        # Add price-based features
        if 'Close' in price_data.columns:
            # Price momentum features
            features['Price_Return_1D'] = price_data['Close'].pct_change()
            features['Price_Return_5D'] = price_data['Close'].pct_change(5)
            features['Price_Return_20D'] = price_data['Close'].pct_change(20)
            
            # Volatility features
            features['Price_Vol_5D'] = price_data['Close'].pct_change().rolling(5).std()
            features['Price_Vol_20D'] = price_data['Close'].pct_change().rolling(20).std()
            
            # Price relative to moving averages
            features['Price_vs_MA20'] = price_data['Close'] / price_data['Close'].rolling(20).mean() - 1
            features['Price_vs_MA50'] = price_data['Close'] / price_data['Close'].rolling(50).mean() - 1
        
        # Add VAR forecasts as features if available
        if var_forecasts is not None:
            # Use the most recent VAR forecast as features
            latest_forecast = var_forecasts.iloc[0]  # First forecast period
            for col in var_forecasts.columns:
                if not col.endswith('_lower') and not col.endswith('_upper'):
                    features[f'VAR_forecast_{col}'] = latest_forecast[col]
        
        # Add time-based features
        features['Month'] = features.index.month
        features['Quarter'] = features.index.quarter
        features['DayOfWeek'] = features.index.dayofweek
        features['IsMonthEnd'] = features.index.is_month_end.astype(int)
        features['IsQuarterEnd'] = features.index.is_quarter_end.astype(int)
        
        # Align features with returns
        aligned_features = features.reindex(returns.index)
        
        # Remove NaN values
        valid_idx = aligned_features.dropna().index.intersection(returns.dropna().index)
        final_features = aligned_features.loc[valid_idx]
        final_returns = returns.loc[valid_idx]
        
        print(f"ML features prepared: {final_features.shape[1]} features, {len(final_features)} samples")
        
        return final_features, final_returns
    
    def prepare_lstm_data(self, features: pd.DataFrame, target: pd.Series, 
                         sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for LSTM model with sequences
        """
        # Scale the features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        self.scalers['lstm_features'] = scaler
        
        # Scale the target
        target_scaler = MinMaxScaler()
        scaled_target = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        self.scalers['lstm_target'] = target_scaler
        
        # Create sequences
        X, y = [], []
        for i in range(sequence_length, len(scaled_features)):
            X.append(scaled_features[i-sequence_length:i])
            y.append(scaled_target[i])
        
        return np.array(X), np.array(y)
    
    def train_random_forest(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Train Random Forest model
        """
        print("Training Random Forest model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.test_size, 
            random_state=self.random_state, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['rf'] = scaler
        
        # Grid search for hyperparameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=self.random_state)
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_rf = grid_search.best_estimator_
        self.models['random_forest'] = best_rf
        
        # Predictions
        y_pred_train = best_rf.predict(X_train_scaled)
        y_pred_test = best_rf.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        self.feature_importance['random_forest'] = pd.Series(
            best_rf.feature_importances_, 
            index=features.columns
        ).sort_values(ascending=False)
        
        self.performance_metrics['random_forest'] = metrics
        
        print(f"Random Forest - Test R²: {metrics['test_r2']:.4f}, Test MAE: {metrics['test_mae']:.6f}")
        
        return metrics
    
    def train_xgboost(self, features: pd.DataFrame, target: pd.Series) -> Dict:
        """
        Train XGBoost model
        """
        print("Training XGBoost model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=self.test_size, 
            random_state=self.random_state, shuffle=False
        )
        
        # XGBoost doesn't require scaling, but we'll do it for consistency
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers['xgb'] = scaler
        
        # Grid search for hyperparameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        xgb_model = xgb.XGBRegressor(
            random_state=self.random_state,
            objective='reg:squarederror'
        )
        
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=tscv, scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Best model
        best_xgb = grid_search.best_estimator_
        self.models['xgboost'] = best_xgb
        
        # Predictions
        y_pred_train = best_xgb.predict(X_train_scaled)
        y_pred_test = best_xgb.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'best_params': grid_search.best_params_
        }
        
        # Feature importance
        self.feature_importance['xgboost'] = pd.Series(
            best_xgb.feature_importances_, 
            index=features.columns
        ).sort_values(ascending=False)
        
        self.performance_metrics['xgboost'] = metrics
        
        print(f"XGBoost - Test R²: {metrics['test_r2']:.4f}, Test MAE: {metrics['test_mae']:.6f}")
        
        return metrics
    
    def train_lstm(self, features: pd.DataFrame, target: pd.Series, 
                  sequence_length: int = 30) -> Dict:
        """
        Train LSTM model
        """
        print("Training LSTM model...")
        
        # Prepare LSTM data
        X, y = self.prepare_lstm_data(features, target, sequence_length)
        
        # Split data
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(sequence_length, features.shape[1])),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            batch_size=self.lstm_batch_size,
            epochs=self.lstm_epochs,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )
        
        self.models['lstm'] = model
        
        # Predictions
        y_pred_train = model.predict(X_train).flatten()
        y_pred_test = model.predict(X_test).flatten()
        
        # Inverse transform predictions
        target_scaler = self.scalers['lstm_target']
        y_train_orig = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
        y_test_orig = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        y_pred_train_orig = target_scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
        y_pred_test_orig = target_scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        metrics = {
            'train_mse': mean_squared_error(y_train_orig, y_pred_train_orig),
            'test_mse': mean_squared_error(y_test_orig, y_pred_test_orig),
            'train_mae': mean_absolute_error(y_train_orig, y_pred_train_orig),
            'test_mae': mean_absolute_error(y_test_orig, y_pred_test_orig),
            'train_r2': r2_score(y_train_orig, y_pred_train_orig),
            'test_r2': r2_score(y_test_orig, y_pred_test_orig),
            'training_history': history.history
        }
        
        self.performance_metrics['lstm'] = metrics
        
        print(f"LSTM - Test R²: {metrics['test_r2']:.4f}, Test MAE: {metrics['test_mae']:.6f}")
        
        return metrics
    
    def ensemble_predictions(self, features: pd.DataFrame, 
                           weights: Dict[str, float] = None) -> np.ndarray:
        """
        Create ensemble predictions from all trained models
        """
        if not self.models:
            raise ValueError("No models trained. Train models first.")
        
        predictions = {}
        
        # Get predictions from each model
        if 'random_forest' in self.models:
            scaler = self.scalers['rf']
            features_scaled = scaler.transform(features)
            predictions['random_forest'] = self.models['random_forest'].predict(features_scaled)
        
        if 'xgboost' in self.models:
            scaler = self.scalers['xgb']
            features_scaled = scaler.transform(features)
            predictions['xgboost'] = self.models['xgboost'].predict(features_scaled)
        
        if 'lstm' in self.models:
            # LSTM requires sequence data
            X_lstm = []
            sequence_length = 30
            
            # Prepare features for LSTM
            feature_scaler = self.scalers['lstm_features']
            scaled_features = feature_scaler.transform(features)
            
            for i in range(sequence_length, len(scaled_features)):
                X_lstm.append(scaled_features[i-sequence_length:i])
            
            if X_lstm:
                X_lstm = np.array(X_lstm)
                lstm_pred_scaled = self.models['lstm'].predict(X_lstm).flatten()
                
                # Inverse transform
                target_scaler = self.scalers['lstm_target']
                lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
                
                # Pad with NaN for alignment
                lstm_pred_full = np.full(len(features), np.nan)
                lstm_pred_full[sequence_length:] = lstm_pred
                predictions['lstm'] = lstm_pred_full
        
        # Default equal weights if not specified
        if weights is None:
            weights = {model: 1.0/len(self.models) for model in self.models.keys()}
        
        # Create ensemble prediction
        ensemble_pred = np.zeros(len(features))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            if model_name in weights:
                weight = weights[model_name]
                # Handle NaN values
                valid_mask = ~np.isnan(pred)
                ensemble_pred[valid_mask] += weight * pred[valid_mask]
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    def predict_next_return(self, current_features: pd.DataFrame) -> Dict[str, float]:
        """
        Predict next period return using all trained models
        """
        predictions = {}
        
        # Individual model predictions
        if 'random_forest' in self.models:
            scaler = self.scalers['rf']
            features_scaled = scaler.transform(current_features)
            predictions['random_forest'] = self.models['random_forest'].predict(features_scaled)[0]
        
        if 'xgboost' in self.models:
            scaler = self.scalers['xgb']
            features_scaled = scaler.transform(current_features)
            predictions['xgboost'] = self.models['xgboost'].predict(features_scaled)[0]
        
        # Ensemble prediction
        if len(predictions) > 1:
            predictions['ensemble'] = np.mean(list(predictions.values()))
        
        return predictions
    
    def get_model_performance_summary(self) -> pd.DataFrame:
        """
        Get performance summary for all trained models
        """
        if not self.performance_metrics:
            return pd.DataFrame()
        
        summary_data = []
        
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'Model': model_name,
                'Train_R2': metrics.get('train_r2', np.nan),
                'Test_R2': metrics.get('test_r2', np.nan),
                'Train_MAE': metrics.get('train_mae', np.nan),
                'Test_MAE': metrics.get('test_mae', np.nan),
                'Train_MSE': metrics.get('train_mse', np.nan),
                'Test_MSE': metrics.get('test_mse', np.nan)
            })
        
        return pd.DataFrame(summary_data)
    
    def get_feature_importance_summary(self, top_n: int = 20) -> Dict[str, pd.Series]:
        """
        Get top feature importances for tree-based models
        """
        importance_summary = {}
        
        for model_name in ['random_forest', 'xgboost']:
            if model_name in self.feature_importance:
                importance_summary[model_name] = self.feature_importance[model_name].head(top_n)
        
        return importance_summary

# Example usage
if __name__ == "__main__":
    print("ML Enhanced Forecasting module ready!")
    print("Use with factor data and VAR forecasts to train ensemble models.") 