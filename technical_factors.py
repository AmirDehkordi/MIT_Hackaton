"""
Technical Factor Extraction Module
Implements various technical indicators and aggregates them into meaningful factors
"""

import pandas as pd
import numpy as np
import ta
from typing import Dict, List, Tuple
import config

class TechnicalFactorExtractor:
    def __init__(self):
        self.config = config.TECHNICAL_INDICATORS
        
    def calculate_moving_averages(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Simple and Exponential Moving Averages
        """
        ma_df = pd.DataFrame(index=data.index)
        
        # Simple Moving Averages
        for period in self.config['sma_periods']:
            ma_df[f'SMA_{period}'] = ta.trend.sma_indicator(data['Close'], window=period)
            ma_df[f'SMA_{period}_ratio'] = data['Close'] / ma_df[f'SMA_{period}'] - 1
        
        # Exponential Moving Averages
        for period in self.config['ema_periods']:
            ma_df[f'EMA_{period}'] = ta.trend.ema_indicator(data['Close'], window=period)
            ma_df[f'EMA_{period}_ratio'] = data['Close'] / ma_df[f'EMA_{period}'] - 1
            
        return ma_df
    
    def calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum-based technical indicators
        """
        momentum_df = pd.DataFrame(index=data.index)
        
        # RSI (Relative Strength Index)
        momentum_df['RSI'] = ta.momentum.rsi(data['Close'], window=self.config['rsi_period'])
        momentum_df['RSI_normalized'] = (momentum_df['RSI'] - 50) / 50  # Normalize around 0
        
        # MACD (Moving Average Convergence Divergence)
        macd_line = ta.trend.macd(data['Close'], 
                                  window_fast=self.config['macd_fast'],
                                  window_slow=self.config['macd_slow'])
        macd_signal = ta.trend.macd_signal(data['Close'],
                                          window_fast=self.config['macd_fast'],
                                          window_slow=self.config['macd_slow'],
                                          window_sign=self.config['macd_signal'])
        momentum_df['MACD'] = macd_line
        momentum_df['MACD_signal'] = macd_signal
        momentum_df['MACD_histogram'] = macd_line - macd_signal
        
        # Stochastic Oscillator
        momentum_df['Stoch_K'] = ta.momentum.stoch(data['High'], data['Low'], data['Close'])
        momentum_df['Stoch_D'] = ta.momentum.stoch_signal(data['High'], data['Low'], data['Close'])
        
        # Williams %R
        momentum_df['Williams_R'] = ta.momentum.williams_r(data['High'], data['Low'], data['Close'])
        
        # Rate of Change
        momentum_df['ROC_10'] = ta.momentum.roc(data['Close'], window=10)
        momentum_df['ROC_21'] = ta.momentum.roc(data['Close'], window=21)
        
        return momentum_df
    
    def calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based indicators
        """
        volatility_df = pd.DataFrame(index=data.index)
        
        # Bollinger Bands
        bb_high = ta.volatility.bollinger_hband(data['Close'], 
                                               window=self.config['bb_period'],
                                               window_dev=self.config['bb_std'])
        bb_low = ta.volatility.bollinger_lband(data['Close'],
                                              window=self.config['bb_period'],
                                              window_dev=self.config['bb_std'])
        bb_mid = ta.volatility.bollinger_mavg(data['Close'], window=self.config['bb_period'])
        
        volatility_df['BB_high'] = bb_high
        volatility_df['BB_low'] = bb_low
        volatility_df['BB_mid'] = bb_mid
        volatility_df['BB_width'] = (bb_high - bb_low) / bb_mid
        volatility_df['BB_position'] = (data['Close'] - bb_low) / (bb_high - bb_low)
        
        # Average True Range (ATR)
        volatility_df['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
        volatility_df['ATR_ratio'] = volatility_df['ATR'] / data['Close']
        
        # Historical Volatility
        returns = data['Close'].pct_change()
        volatility_df['HV_10'] = returns.rolling(window=10).std() * np.sqrt(252)
        volatility_df['HV_21'] = returns.rolling(window=21).std() * np.sqrt(252)
        volatility_df['HV_63'] = returns.rolling(window=63).std() * np.sqrt(252)
        
        return volatility_df
    
    def calculate_volume_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volume-based indicators
        """
        volume_df = pd.DataFrame(index=data.index)
        
        # Volume Moving Averages (simple rolling means)
        volume_df['Volume_SMA_10'] = data['Volume'].rolling(window=10).mean()
        volume_df['Volume_SMA_21'] = data['Volume'].rolling(window=21).mean()
        
        # On-Balance Volume (OBV)
        volume_df['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
        volume_df['OBV_SMA_10'] = volume_df['OBV'].rolling(window=10).mean()
        
        # Volume Price Trend (VPT)
        volume_df['VPT'] = ta.volume.volume_price_trend(data['Close'], data['Volume'])
        
        # Ease of Movement
        volume_df['EOM'] = ta.volume.ease_of_movement(data['High'], data['Low'], data['Volume'])
        volume_df['EOM_SMA_14'] = volume_df['EOM'].rolling(window=14).mean()
        
        # Volume Ratio
        volume_df['Volume_ratio'] = data['Volume'] / data['Volume'].rolling(window=21).mean()
        
        return volume_df
    
    def calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend-following indicators
        """
        trend_df = pd.DataFrame(index=data.index)
        
        # Average Directional Index (ADX)
        trend_df['ADX'] = ta.trend.adx(data['High'], data['Low'], data['Close'])
        trend_df['ADX_pos'] = ta.trend.adx_pos(data['High'], data['Low'], data['Close'])
        trend_df['ADX_neg'] = ta.trend.adx_neg(data['High'], data['Low'], data['Close'])
        
        # Commodity Channel Index (CCI)
        trend_df['CCI'] = ta.trend.cci(data['High'], data['Low'], data['Close'])
        
        # Aroon Indicator (with explicit window parameter)
        try:
            trend_df['Aroon_up'] = ta.trend.aroon_up(data['High'], data['Low'], window=14)
            trend_df['Aroon_down'] = ta.trend.aroon_down(data['High'], data['Low'], window=14)
            trend_df['Aroon_indicator'] = trend_df['Aroon_up'] - trend_df['Aroon_down']
        except:
            # Fallback: simple calculation
            trend_df['Aroon_up'] = 0
            trend_df['Aroon_down'] = 0
            trend_df['Aroon_indicator'] = 0
        
        # Parabolic SAR
        try:
            trend_df['PSAR'] = ta.trend.psar_down(data['High'], data['Low'], data['Close'])
            trend_df['PSAR_signal'] = np.where(data['Close'] > trend_df['PSAR'], 1, -1)
        except:
            # Fallback: use simple moving average as proxy
            trend_df['PSAR'] = data['Close'].rolling(window=10).mean()
            trend_df['PSAR_signal'] = np.where(data['Close'] > trend_df['PSAR'], 1, -1)
        
        return trend_df
    
    def calculate_price_action_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price action and pattern features
        """
        price_df = pd.DataFrame(index=data.index)
        
        # Basic price features
        price_df['Returns'] = data['Close'].pct_change()
        price_df['Log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # High-Low ratio
        price_df['HL_ratio'] = (data['High'] - data['Low']) / data['Close']
        
        # Open-Close ratio
        price_df['OC_ratio'] = (data['Close'] - data['Open']) / data['Open']
        
        # Gap features
        price_df['Gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
        
        # Price position within the day's range
        price_df['Price_position'] = (data['Close'] - data['Low']) / (data['High'] - data['Low'])
        
        # Rolling statistics
        for window in [5, 10, 21, 63]:
            price_df[f'Returns_mean_{window}'] = price_df['Returns'].rolling(window=window).mean()
            price_df[f'Returns_std_{window}'] = price_df['Returns'].rolling(window=window).std()
            price_df[f'Returns_skew_{window}'] = price_df['Returns'].rolling(window=window).skew()
            price_df[f'Returns_kurt_{window}'] = price_df['Returns'].rolling(window=window).kurt()
        
        return price_df
    
    def create_technical_factor(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate all technical indicators into a comprehensive technical factor
        """
        # Check if data is empty or insufficient
        if data.empty or len(data) < 50:
            print(f"⚠️ Insufficient data for technical analysis: {len(data)} rows")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(index=data.index if not data.empty else pd.DatetimeIndex([]))
        
        try:
            # Calculate all individual indicators
            ma_indicators = self.calculate_moving_averages(data)
            momentum_indicators = self.calculate_momentum_indicators(data)
            volatility_indicators = self.calculate_volatility_indicators(data)
            volume_indicators = self.calculate_volume_indicators(data)
            trend_indicators = self.calculate_trend_indicators(data)
            price_indicators = self.calculate_price_action_features(data)
            
            # Combine all indicators
            technical_data = pd.concat([
                ma_indicators,
                momentum_indicators,
                volatility_indicators,
                volume_indicators,
                trend_indicators,
                price_indicators
            ], axis=1)
            
            # Forward fill missing values and drop remaining NaNs
            technical_data = technical_data.ffill().dropna()
            
            return technical_data
            
        except Exception as e:
            print(f"❌ Error in technical factor calculation: {e}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(index=data.index if not data.empty else pd.DatetimeIndex([]))
    
    def create_aggregated_factors(self, technical_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated factors from technical indicators using PCA or factor analysis
        """
        # Check if technical data is empty or insufficient
        if technical_data.empty or technical_data.shape[0] < 10 or technical_data.shape[1] < 2:
            print(f"⚠️ Insufficient technical data for PCA: shape {technical_data.shape}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(index=technical_data.index if not technical_data.empty else pd.DatetimeIndex([]))
        
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(technical_data.fillna(0))
            
            # Apply PCA to create factors
            n_components = min(10, scaled_data.shape[1])  # Limit to 10 factors
            if n_components < 1:
                print(f"⚠️ No components available for PCA")
                return pd.DataFrame(index=technical_data.index)
                
            pca = PCA(n_components=n_components)
            factors = pca.fit_transform(scaled_data)
            
            # Create factor DataFrame
            factor_columns = [f'Tech_Factor_{i+1}' for i in range(n_components)]
            factor_df = pd.DataFrame(factors, 
                                    index=technical_data.index, 
                                    columns=factor_columns)
            
            # Store the explained variance ratio
            self.tech_factor_variance = pca.explained_variance_ratio_
            self.tech_factor_components = pca.components_
            
            return factor_df
            
        except Exception as e:
            print(f"❌ Error in PCA factor creation: {e}")
            # Return empty DataFrame with proper structure
            return pd.DataFrame(index=technical_data.index if not technical_data.empty else pd.DatetimeIndex([]))

# Example usage
if __name__ == "__main__":
    # This would typically be called with real price data
    print("Technical Factor Extractor module ready!")
    print("Use with stock price data to generate technical factors.") 