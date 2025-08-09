"""
Basic functionality test script
Tests individual components with sample data when APIs are not available
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf

def test_data_collection():
    """Test basic data collection without API dependencies"""
    print("🔍 Testing basic data collection...")
    
    try:
        # Test Yahoo Finance directly
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="1y")
        
        if not data.empty:
            print(f"✅ Yahoo Finance working! Got {len(data)} days of data for AAPL")
            print(f"   Latest price: ${data['Close'].iloc[-1]:.2f}")
            print(f"   Date range: {data.index[0].date()} to {data.index[-1].date()}")
            return data
        else:
            print("❌ Yahoo Finance returned empty data")
            return None
            
    except Exception as e:
        print(f"❌ Error with Yahoo Finance: {e}")
        return None

def create_sample_data():
    """Create sample stock data for testing"""
    print("\n🏗️  Creating sample data for testing...")
    
    # Generate 2 years of sample daily stock data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='B')  # Business days
    n_days = len(dates)
    
    # Generate realistic stock price movement
    np.random.seed(42)  # For reproducible results
    
    # Start with a base price
    initial_price = 150.0
    
    # Generate random returns with some trend and volatility
    daily_returns = np.random.normal(0.0005, 0.02, n_days)  # Small positive drift, 2% daily volatility
    
    # Calculate cumulative prices
    cumulative_returns = np.cumprod(1 + daily_returns)
    prices = initial_price * cumulative_returns
    
    # Create OHLCV data
    sample_data = pd.DataFrame(index=dates)
    
    # Close prices from our calculation
    sample_data['Close'] = prices
    
    # Generate Open, High, Low based on Close
    sample_data['Open'] = sample_data['Close'].shift(1) * (1 + np.random.normal(0, 0.005, n_days))
    sample_data['Open'].iloc[0] = initial_price
    
    # High is the maximum of Open/Close plus some random uptick
    sample_data['High'] = np.maximum(sample_data['Open'], sample_data['Close']) * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    
    # Low is the minimum of Open/Close minus some random downtick
    sample_data['Low'] = np.minimum(sample_data['Open'], sample_data['Close']) * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    
    # Generate realistic volume (millions of shares)
    sample_data['Volume'] = np.random.lognormal(15, 0.5, n_days).astype(int)  # Log-normal distribution for volume
    
    # Adjusted close (same as close for simplicity)
    sample_data['Adj Close'] = sample_data['Close']
    
    print(f"✅ Sample data created!")
    print(f"   📊 Shape: {sample_data.shape}")
    print(f"   📅 Date range: {sample_data.index[0].date()} to {sample_data.index[-1].date()}")
    print(f"   💰 Price range: ${sample_data['Low'].min():.2f} - ${sample_data['High'].max():.2f}")
    print(f"   📈 Final price: ${sample_data['Close'].iloc[-1]:.2f}")
    
    return sample_data

def test_technical_factors(price_data):
    """Test technical factor extraction"""
    print("\n🔧 Testing technical factor extraction...")
    
    try:
        from technical_factors import TechnicalFactorExtractor
        
        extractor = TechnicalFactorExtractor()
        
        # Test individual indicator calculations
        print("   📊 Calculating moving averages...")
        ma_data = extractor.calculate_moving_averages(price_data)
        print(f"      ✅ Moving averages: {ma_data.shape[1]} indicators")
        
        print("   📈 Calculating momentum indicators...")
        momentum_data = extractor.calculate_momentum_indicators(price_data)
        print(f"      ✅ Momentum indicators: {momentum_data.shape[1]} indicators")
        
        print("   📊 Calculating volatility indicators...")
        volatility_data = extractor.calculate_volatility_indicators(price_data)
        print(f"      ✅ Volatility indicators: {volatility_data.shape[1]} indicators")
        
        print("   📦 Creating aggregated technical factors...")
        technical_indicators = extractor.create_technical_factor(price_data)
        technical_factors = extractor.create_aggregated_factors(technical_indicators)
        
        print(f"   ✅ Technical factors created!")
        print(f"      📊 Raw indicators: {technical_indicators.shape}")
        print(f"      🎯 Aggregated factors: {technical_factors.shape}")
        
        return technical_factors
        
    except Exception as e:
        print(f"   ❌ Error in technical factor extraction: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_fundamental_factors():
    """Test fundamental factor extraction with sample data"""
    print("\n💰 Testing fundamental factor extraction...")
    
    try:
        from fundamental_factors import FundamentalFactorExtractor
        
        extractor = FundamentalFactorExtractor()
        
        # Create sample fundamental data
        sample_fundamental = {
            'income_statement': pd.DataFrame({
                'totalRevenue': [100000, 110000, 120000],
                'grossProfit': [40000, 45000, 50000],
                'operatingIncome': [20000, 23000, 26000],
                'netIncome': [15000, 17000, 19000]
            }, index=pd.date_range('2022-01-01', periods=3, freq='A')),
            
            'balance_sheet': pd.DataFrame({
                'totalAssets': [200000, 220000, 240000],
                'totalLiabilities': [100000, 105000, 110000],
                'totalStockholdersEquity': [100000, 115000, 130000]
            }, index=pd.date_range('2022-01-01', periods=3, freq='A'))
        }
        
        # Test fundamental factor creation
        fundamental_factors = extractor.create_fundamental_factor(sample_fundamental)
        
        if not fundamental_factors.empty:
            print(f"   ✅ Fundamental factors created: {fundamental_factors.shape}")
            print(f"      📊 Metrics: {list(fundamental_factors.columns)[:5]}...")
            return fundamental_factors
        else:
            print("   ⚠️  Empty fundamental factors")
            return None
            
    except Exception as e:
        print(f"   ❌ Error in fundamental factor extraction: {e}")
        return None

def test_sentiment_factors(price_data):
    """Test sentiment factor extraction with sample data"""
    print("\n😊 Testing sentiment factor extraction...")
    
    try:
        from sentiment_factors import SentimentFactorExtractor
        
        extractor = SentimentFactorExtractor()
        
        # Create sample market sentiment data
        sample_sentiment_data = {
            'vix': pd.DataFrame({
                'VIX': np.random.normal(20, 5, len(price_data))
            }, index=price_data.index),
        }
        
        # Test sentiment factor creation
        sentiment_factors = extractor.create_comprehensive_sentiment_factor(
            "AAPL", price_data, sample_sentiment_data
        )
        
        if not sentiment_factors.empty:
            print(f"   ✅ Sentiment factors created: {sentiment_factors.shape}")
            return sentiment_factors
        else:
            print("   ⚠️  Empty sentiment factors")
            return None
            
    except Exception as e:
        print(f"   ❌ Error in sentiment factor extraction: {e}")
        return None

def test_var_model(technical_factors, fundamental_factors, sentiment_factors):
    """Test VAR model with sample factors"""
    print("\n🔮 Testing VAR model...")
    
    try:
        from var_model import ChincariniVARModel
        
        var_model = ChincariniVARModel()
        
        # Prepare factor data
        print("   📊 Preparing factor data...")
        factor_data = var_model.prepare_factor_data(
            technical_factors if technical_factors is not None else pd.DataFrame(),
            fundamental_factors if fundamental_factors is not None else pd.DataFrame(),
            sentiment_factors if sentiment_factors is not None else pd.DataFrame()
        )
        
        if not factor_data.empty and factor_data.shape[0] > 50:  # Need enough data for VAR
            print("   🧪 Testing stationarity...")
            stationarity_results = var_model.test_stationarity()
            
            print("   📈 Making data stationary...")
            stationary_data = var_model.make_stationary()
            
            print("   🎯 Selecting optimal lags...")
            optimal_lags = var_model.select_optimal_lags()
            
            print("   🏗️  Fitting VAR model...")
            var_model.fit_var_model(lags=min(optimal_lags, 3))  # Limit lags for demo
            
            print("   🔮 Generating forecasts...")
            forecasts = var_model.forecast_factors(steps=10)
            
            print(f"   ✅ VAR model completed!")
            print(f"      📊 Factors: {factor_data.shape[1]}")
            print(f"      🎯 Optimal lags: {optimal_lags}")
            print(f"      🔮 Forecast shape: {forecasts.shape}")
            
            return forecasts
        else:
            print("   ⚠️  Insufficient data for VAR model")
            return None
            
    except Exception as e:
        print(f"   ❌ Error in VAR model: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run basic functionality tests"""
    print("🧪 BASIC FUNCTIONALITY TEST")
    print("="*60)
    
    # Test 1: Data Collection
    real_data = test_data_collection()
    
    # Use real data if available, otherwise create sample data
    if real_data is not None and not real_data.empty:
        price_data = real_data
        print("   📈 Using real market data")
    else:
        price_data = create_sample_data()
        print("   🏗️  Using sample data")
    
    # Test 2: Technical Factors
    technical_factors = test_technical_factors(price_data)
    
    # Test 3: Fundamental Factors
    fundamental_factors = test_fundamental_factors()
    
    # Test 4: Sentiment Factors
    sentiment_factors = test_sentiment_factors(price_data)
    
    # Test 5: VAR Model (if we have enough factors)
    if technical_factors is not None:
        var_forecasts = test_var_model(technical_factors, fundamental_factors, sentiment_factors)
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Data Collection: {'✓' if price_data is not None else '✗'}")
    print(f"✅ Technical Factors: {'✓' if technical_factors is not None else '✗'}")
    print(f"✅ Fundamental Factors: {'✓' if fundamental_factors is not None else '✗'}")
    print(f"✅ Sentiment Factors: {'✓' if sentiment_factors is not None else '✗'}")
    print(f"✅ VAR Model: {'✓' if 'var_forecasts' in locals() and var_forecasts is not None else '✗'}")
    
    if technical_factors is not None:
        print("\n🎉 Core functionality is working!")
        print("💡 The system can process data and generate factors successfully.")
        print("📈 Technical analysis pipeline: OPERATIONAL")
        
        if 'var_forecasts' in locals() and var_forecasts is not None:
            print("🔮 Forecasting pipeline: OPERATIONAL")
            print("🚀 System ready for comprehensive analysis!")
        else:
            print("⚠️  VAR forecasting needs more data or debugging")
    else:
        print("❌ Some core components need attention")
    
    print("\n🔗 Next steps:")
    print("   1. Verify API keys if using real data")
    print("   2. Check network connectivity for data sources")
    print("   3. Run full system with working data sources")

if __name__ == "__main__":
    main() 