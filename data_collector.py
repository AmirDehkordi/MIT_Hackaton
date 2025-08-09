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
            data.index = pd.to_datetime(data.index)
            return data
        except Exception as e:
            print(f"Error fetching data for {ticker} from Yahoo Finance: {e}")
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
    
    def get_fundamental_data(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get fundamental data from Alpha Vantage
        """
        fundamental_data = {}
        
        try:
            # Income Statement
            income_statement, meta_data = self.fd.get_income_statement_annual(symbol=ticker)
            fundamental_data['income_statement'] = income_statement
            time.sleep(12)  # Alpha Vantage rate limit
            
            # Balance Sheet
            balance_sheet, meta_data = self.fd.get_balance_sheet_annual(symbol=ticker)
            fundamental_data['balance_sheet'] = balance_sheet
            time.sleep(12)
            
            # Cash Flow
            cash_flow, meta_data = self.fd.get_cash_flow_annual(symbol=ticker)
            fundamental_data['cash_flow'] = cash_flow
            time.sleep(12)
            
            # Company Overview
            overview, meta_data = self.fd.get_company_overview(symbol=ticker)
            fundamental_data['overview'] = overview
            
        except Exception as e:
            print(f"Error fetching fundamental data for {ticker}: {e}")
            
        return fundamental_data
    
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
                time.sleep(1)  # Rate limiting
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
                'fundamental_data': self.get_fundamental_data(ticker),
            }
            
            all_data[ticker] = ticker_data
            
            # Add delays to respect API rate limits
            time.sleep(2)
        
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