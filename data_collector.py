"""
Data collection module for financial forecasting
Supports Alpha Vantage, Yahoo Finance, and Quandl APIs
"""

import pandas as pd
import numpy as np
import yfinance as yf
import quandl
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from alpha_vantage.techindicators import TechIndicators
import requests
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import config

class DataCollector:
    def __init__(self):
        self.alpha_vantage_key = config.ALPHA_VANTAGE_API_KEY
        self.quandl_key = config.QUANDL_API_KEY
        
        # Initialize API clients
        self.ts = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        self.fd = FundamentalData(key=self.alpha_vantage_key, output_format='pandas')
        self.ti = TechIndicators(key=self.alpha_vantage_key, output_format='pandas')
        
        # Set Quandl API key
        quandl.ApiConfig.api_key = self.quandl_key
        
    def get_stock_data_yfinance(self, ticker: str, period: str = "5y") -> pd.DataFrame:
        """
        Get stock price data from Yahoo Finance
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period)
            
            if not data.empty:
                data.index = pd.to_datetime(data.index)
                print(f"✅ Successfully fetched {len(data)} days of data for {ticker}")
                return data
            else:
                print(f"⚠️ No data available for {ticker}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {e}")
            return pd.DataFrame()
        
        # If all periods fail, try Alpha Vantage as fallback
        print(f"🔄 Trying Alpha Vantage as fallback for {ticker}...")
        alpha_data = self.get_stock_data_alpha_vantage(ticker)
        
        if not alpha_data.empty:
            return alpha_data
        
        # If all data sources fail, return empty DataFrame
        print(f"❌ All data sources failed for {ticker}. No data available.")
        return pd.DataFrame()
        
    
    def get_stock_data_alpha_vantage(self, ticker: str, outputsize: str = "full") -> pd.DataFrame:
        """
        Get stock price data from Alpha Vantage
        """
        try:
            data, meta_data = self.ts.get_daily_adjusted(symbol=ticker, outputsize=outputsize)
            data.index = pd.to_datetime(data.index)
            data = data.sort_index()
            
            # Rename columns to match yfinance format
            column_mapping = {
                '1. open': 'Open',
                '2. high': 'High',
                '3. low': 'Low',
                '4. close': 'Close',
                '5. adjusted close': 'Adj Close',
                '6. volume': 'Volume'
            }
            data = data.rename(columns=column_mapping)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker} from Alpha Vantage: {e}")
            return pd.DataFrame()



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
    
    def get_economic_indicators(self) -> Dict[str, pd.DataFrame]:
        """
        Get economic indicators from Quandl and FRED
        """
        indicators = {}
        
        try:
            # GDP Growth Rate
            indicators['gdp_growth'] = quandl.get("FRED/GDPC1", start_date="2010-01-01")
            
            # Federal Funds Rate
            indicators['fed_rate'] = quandl.get("FRED/FEDFUNDS", start_date="2010-01-01")
            
            # Unemployment Rate
            indicators['unemployment'] = quandl.get("FRED/UNRATE", start_date="2010-01-01")
            
            # VIX (Volatility Index)
            indicators['vix'] = quandl.get("CBOE/VIX", start_date="2010-01-01")
            
            # Treasury Yield Curve
            indicators['10y_treasury'] = quandl.get("FRED/DGS10", start_date="2010-01-01")
            indicators['2y_treasury'] = quandl.get("FRED/DGS2", start_date="2010-01-01")
            
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            
        return indicators
    
    def get_sector_data(self, sector_tickers: List[str]) -> pd.DataFrame:
        """
        Get sector ETF data for sector analysis
        """
        sector_data = {}
        
        for ticker in sector_tickers:
            try:
                data = self.get_stock_data_yfinance(ticker, period="5y")
                if not data.empty:
                    sector_data[ticker] = data['Adj Close']

            except Exception as e:
                print(f"Error fetching sector data for {ticker}: {e}")
        
        if sector_data:
            return pd.DataFrame(sector_data)
        return pd.DataFrame()
    
    def get_market_sentiment_data(self) -> Dict[str, pd.DataFrame]:
        """
        Get market sentiment indicators
        """
        sentiment_data = {}
        
        try:
            # Put/Call Ratio
            sentiment_data['put_call_ratio'] = quandl.get("CBOE/EQUITY_PC", start_date="2010-01-01")
            
            # AAII Sentiment Survey
            sentiment_data['aaii_bull'] = quandl.get("AAII/AAII_SENTIMENT", start_date="2010-01-01")
            
        except Exception as e:
            print(f"Error fetching sentiment data: {e}")
            
        return sentiment_data
    
    def collect_all_data(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Comprehensive data collection for multiple tickers
        """
        all_data = {}
        
        # Sector ETFs for sector analysis
        sector_etfs = ['XLK', 'XLF', 'XLV', 'XLE', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB', 'XLRE', 'XLC']
        
        for ticker in tickers:
            print(f"Collecting data for {ticker}...")
            
            ticker_data = {
                'price_data': self.get_stock_data_yfinance(ticker),
                # Removed fundamental_data to fix the error
            }
            
            all_data[ticker] = ticker_data
            

        
        # Collect market-wide data once
        print("Collecting market-wide data...")
        all_data['market_data'] = {
            'economic_indicators': self.get_economic_indicators(),
            'sector_data': self.get_sector_data(sector_etfs),
            'sentiment_data': self.get_market_sentiment_data()
        }
        
        return all_data

# Example usage
if __name__ == "__main__":
    collector = DataCollector()
    
    # Test with a few tickers
    test_tickers = ['AAPL', 'GOOGL']
    data = collector.collect_all_data(test_tickers)
    
    print("Data collection completed!")
    for ticker in test_tickers:
        if ticker in data:
            print(f"{ticker} price data shape: {data[ticker]['price_data'].shape}") 
