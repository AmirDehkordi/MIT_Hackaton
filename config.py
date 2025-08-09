"""
Configuration file for the forecasting system
"""

# API Keys
ALPHA_VANTAGE_API_KEY = "JWE9E91AMPNQRQX8"
QUANDL_API_KEY = "3gUQnKPkaJRZD_PFaBa6"
ADDITIONAL_KEY = "AdGnFS7zUxQar34BNmee"

# Data settings
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'JPM', 'JNJ', 'V']
LOOKBACK_PERIOD = 252 * 5  # 5 years of trading days
FORECAST_HORIZON = 22  # 1 month ahead

# Technical indicator settings
TECHNICAL_INDICATORS = {
    'sma_periods': [5, 10, 20, 50, 200],
    'ema_periods': [12, 26],
    'rsi_period': 14,
    'bb_period': 20,
    'bb_std': 2,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

# VAR model settings
VAR_SETTINGS = {
    'max_lags': 10,
    'trend': 'ct',  # constant and trend
    'method': 'ols'
}

# ML model settings
ML_SETTINGS = {
    'test_size': 0.2,
    'random_state': 42,
    'lstm_epochs': 100,
    'lstm_batch_size': 32,
    'xgb_n_estimators': 100
} 