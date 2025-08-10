#!/usr/bin/env python3
"""
Debug test script to isolate the data processing error
"""

import sys
import traceback
from main_forecasting_system import ComprehensiveStockForecaster

def test_basic_functionality():
    """Test basic forecasting functionality with error handling"""
    print("🔍 Starting debug test...")
    
    try:
        # Use AAPL for testing - our improved data collector will handle fallbacks
        ticker = "AAPL"
        print(f"📈 Testing with ticker: {ticker} (with fallback to sample data)")
        
        # Initialize the forecasting system
        print("📊 Initializing forecasting system...")
        forecaster = ComprehensiveStockForecaster(ticker)
        
        # Collect data
        print("📥 Collecting data...")
        forecaster.collect_all_data(ticker)
        
        # Check if data was collected successfully
        if not forecaster.raw_data:
            print("❌ No data collected!")
            return False
            
        print(f"✅ Data collected successfully!")
        print(f"   Price data shape: {forecaster.raw_data['price_data'].shape}")
        
        # Extract factors
        print("🔧 Extracting factors...")
        factors = forecaster.extract_all_factors()
        
        print(f"✅ Factors extracted successfully!")
        for factor_type, factor_data in factors.items():
            if hasattr(factor_data, 'shape'):
                print(f"   {factor_type} shape: {factor_data.shape}")
            else:
                print(f"   {factor_type}: {type(factor_data)}")
        
        # Train models
        print("🤖 Training ML models...")
        forecaster.train_ml_models()
        
        print("✅ Models trained successfully!")
        
        # Generate forecast
        print("🔮 Generating forecast...")
        forecast = forecaster.generate_comprehensive_forecast(forecast_days=22)
        
        print(f"✅ Forecast generated successfully!")
        print(f"   Forecast shape: {forecast.shape}")
        print(f"   Forecast columns: {list(forecast.columns)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        print(f"📋 Error type: {type(e).__name__}")
        print("📋 Full traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Running debug test for forecasting system...")
    success = test_basic_functionality()
    
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Test failed - check errors above")
