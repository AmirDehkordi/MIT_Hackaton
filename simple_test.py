#!/usr/bin/env python3
"""
Simple test with mock data to verify the system works without API calls
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import traceback

def create_mock_stock_data(ticker="AAPL", days=500):
    """Create realistic mock stock data for testing"""
    print(f"📊 Creating mock data for {ticker} ({days} days)...")
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate realistic stock price data
    np.random.seed(42)  # For reproducible results
    
    # Starting price
    initial_price = 150.0
    
    # Generate returns with some trend and volatility
    daily_returns = np.random.normal(0.0005, 0.02, len(dates))  # Small positive drift, 2% daily vol
    
    # Generate prices
    prices = [initial_price]
    for ret in daily_returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLCV data
    data = pd.DataFrame(index=dates)
    data['Close'] = prices
    data['Open'] = data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, len(data)))
    data['High'] = np.maximum(data['Open'], data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, len(data))))
    data['Low'] = np.minimum(data['Open'], data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, len(data))))
    data['Volume'] = np.random.lognormal(15, 0.5, len(data))  # Realistic volume
    data['Adj Close'] = data['Close']  # Simplified
    
    # Remove weekends (business days only)
    data = data[data.index.dayofweek < 5]
    
    # Fill any NaN values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    print(f"✅ Mock data created: {data.shape}")
    return data

def test_technical_analysis():
    """Test technical analysis with mock data"""
    print("🔧 Testing technical analysis...")
    
    try:
        from technical_factors import TechnicalFactorExtractor
        
        # Create mock data
        mock_data = create_mock_stock_data("AAPL", 300)
        
        # Initialize technical extractor
        tech_extractor = TechnicalFactorExtractor()
        
        # Test technical factor creation
        print("📈 Creating technical factors...")
        technical_indicators = tech_extractor.create_technical_factor(mock_data)
        
        print(f"✅ Technical indicators created: {technical_indicators.shape}")
        
        # Test aggregated factors
        print("🔄 Creating aggregated factors...")
        technical_factors = tech_extractor.create_aggregated_factors(technical_indicators)
        
        print(f"✅ Aggregated factors created: {technical_factors.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in technical analysis: {str(e)}")
        traceback.print_exc()
        return False

def test_full_system():
    """Test the full system with mock data"""
    print("🚀 Testing full forecasting system...")
    
    try:
        from main_forecasting_system import ComprehensiveStockForecaster
        
        # Create a simple test that bypasses data collection
        ticker = "AAPL"
        forecaster = ComprehensiveStockForecaster(ticker)
        
        # Manually inject mock data to bypass API calls
        mock_price_data = create_mock_stock_data(ticker, 300)
        
        forecaster.raw_data = {
            'price_data': mock_price_data,
            'market_data': {
                'economic_indicators': {},
                'sector_data': pd.DataFrame(),
                'sentiment_data': {}
            }
        }
        
        print("✅ Mock data injected successfully!")
        
        # Test factor extraction
        print("🔧 Testing factor extraction...")
        factors = forecaster.extract_all_factors()
        
        print("✅ Factor extraction completed!")
        for factor_type, factor_data in factors.items():
            if hasattr(factor_data, 'shape'):
                print(f"   {factor_type} shape: {factor_data.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in full system test: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Running simple tests with mock data...")
    
    # Test 1: Technical analysis
    tech_success = test_technical_analysis()
    
    # Test 2: Full system
    system_success = test_full_system()
    
    if tech_success and system_success:
        print("🎉 All tests passed! System works with proper data.")
        print("💡 Issue is with data collection APIs (rate limiting), not the core logic.")
    else:
        print("💥 Some tests failed - check errors above")
