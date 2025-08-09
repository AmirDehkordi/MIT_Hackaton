"""
Example Usage Script for the AI-Powered Stock Forecasting System
Demonstrates basic usage and showcases key features
"""

import sys
import traceback
from datetime import datetime
from main_forecasting_system import ComprehensiveStockForecaster
from backtesting import ForecastingBacktester

def demo_single_stock_analysis(ticker="AAPL"):
    """
    Demonstrate complete analysis for a single stock
    """
    print(f"\n{'='*60}")
    print(f"🔮 COMPREHENSIVE STOCK ANALYSIS DEMO")
    print(f"📊 Analyzing {ticker}")
    print(f"{'='*60}")
    
    try:
        # Initialize the forecaster
        print(f"\n1️⃣  Initializing forecaster for {ticker}...")
        forecaster = ComprehensiveStockForecaster(ticker)
        
        # Step 1: Data Collection
        print(f"\n2️⃣  Collecting comprehensive data...")
        data = forecaster.collect_all_data()
        print(f"✅ Data collected successfully!")
        print(f"   📈 Price data shape: {data['price_data'].shape}")
        
        # Step 2: Factor Extraction
        print(f"\n3️⃣  Extracting multi-factor features...")
        factors = forecaster.extract_all_factors()
        print(f"✅ Factors extracted successfully!")
        for factor_type, factor_data in factors.items():
            if isinstance(factor_data, type(data['price_data'])) and not factor_data.empty:
                print(f"   🔧 {factor_type}: {factor_data.shape}")
        
        # Step 3: VAR Model Building
        print(f"\n4️⃣  Building Chincarini VAR model...")
        var_results = forecaster.build_var_model()
        print(f"✅ VAR model built successfully!")
        print(f"   📊 Optimal lags: {var_results['optimal_lags']}")
        print(f"   🎯 Forecast horizon: {var_results['forecasts'].shape[0]} days")
        
        # Step 4: ML Model Training
        print(f"\n5️⃣  Training ML enhancement models...")
        ml_results = forecaster.train_ml_models()
        print(f"✅ ML models trained successfully!")
        
        # Display model performance
        performance = ml_results['performance_summary']
        if not performance.empty:
            print(f"\n   🏆 Model Performance Summary:")
            for _, row in performance.iterrows():
                print(f"      {row['Model']}: R² = {row['Test_R2']:.4f}, MAE = {row['Test_MAE']:.6f}")
        
        # Step 5: Generate Forecast
        print(f"\n6️⃣  Generating comprehensive forecast...")
        forecast = forecaster.generate_comprehensive_forecast()
        print(f"✅ Forecast generated successfully!")
        
        # Step 6: Investment Recommendation
        print(f"\n7️⃣  Generating investment recommendation...")
        recommendation = forecaster.get_investment_recommendation()
        
        # Display results
        print(f"\n{'='*60}")
        print(f"🎯 INVESTMENT RECOMMENDATION FOR {ticker}")
        print(f"{'='*60}")
        print(f"📊 Current Price: ${recommendation['current_price']:.2f}")
        print(f"🎯 1-Week Target: ${recommendation['target_price_1w']:.2f}")
        print(f"🎯 1-Month Target: ${recommendation['target_price_1m']:.2f}")
        print(f"📈 Expected 1-Week Return: {recommendation['expected_return_1w']:.2%}")
        print(f"📈 Expected 1-Month Return: {recommendation['expected_return_1m']:.2%}")
        print(f"⚖️  Risk-Adjusted Return: {recommendation['risk_adjusted_return']:.3f}")
        print(f"📊 Volatility: {recommendation['volatility']:.4f}")
        print(f"🎲 Confidence Score: {recommendation['confidence_score']:.3f}")
        print(f"💡 RECOMMENDATION: {recommendation['recommendation']}")
        print(f"{'='*60}")
        
        # Step 7: Create Visualization
        print(f"\n8️⃣  Creating forecast visualization...")
        forecaster.visualize_forecast(save_path=f"demo_forecast_{ticker}.png")
        print(f"✅ Visualization saved as demo_forecast_{ticker}.png")
        
        # Step 8: Save Results
        print(f"\n9️⃣  Saving analysis results...")
        forecaster.save_complete_analysis(directory=f"demo_analysis_{ticker}")
        print(f"✅ Complete analysis saved to demo_analysis_{ticker}/")
        
        return recommendation
        
    except Exception as e:
        print(f"❌ Error in single stock analysis: {e}")
        traceback.print_exc()
        return None

def demo_backtesting(tickers=["AAPL"], quick_test=True):
    """
    Demonstrate backtesting functionality
    """
    print(f"\n{'='*60}")
    print(f"📊 BACKTESTING DEMO")
    print(f"🔍 Testing tickers: {', '.join(tickers)}")
    print(f"{'='*60}")
    
    try:
        # Initialize backtester
        print(f"\n1️⃣  Initializing backtester...")
        start_date = "2022-01-01" if quick_test else "2020-01-01"
        backtester = ForecastingBacktester(tickers, start_date=start_date)
        print(f"✅ Backtester initialized for period starting {start_date}")
        
        # Run limited backtest for demo
        if quick_test:
            print(f"\n2️⃣  Running quick backtest (limited scope for demo)...")
            print(f"⚠️  Note: This is a simplified demo. Full backtesting requires more data and time.")
            
            # For demo purposes, we'll just show the framework
            print(f"✅ Backtest framework ready!")
            print(f"   📊 Tickers to test: {len(tickers)}")
            print(f"   📅 Start date: {start_date}")
            print(f"   🔧 Walk-forward validation: Enabled")
            
            return {"demo": "completed", "tickers": tickers}
        else:
            # Run full backtest
            print(f"\n2️⃣  Running comprehensive backtest...")
            results = backtester.run_comprehensive_backtest()
            
            # Generate performance report
            print(f"\n3️⃣  Creating performance report...")
            backtester.create_performance_report(save_path="demo_backtest_report.png")
            
            # Save results
            print(f"\n4️⃣  Saving backtest results...")
            backtester.save_backtest_results(directory="demo_backtest_results")
            
            return results
        
    except Exception as e:
        print(f"❌ Error in backtesting demo: {e}")
        traceback.print_exc()
        return None

def demo_system_overview():
    """
    Print system overview and capabilities
    """
    print(f"\n{'='*80}")
    print(f"🚀 AI-POWERED STOCK FORECASTING SYSTEM")
    print(f"{'='*80}")
    print(f"""
🎯 SYSTEM CAPABILITIES:
   • Multi-factor data integration (Technical + Fundamental + Sentiment)
   • Chincarini-style VAR modeling for factor forecasting
   • ML enhancement with Random Forest, XGBoost, and LSTM
   • Comprehensive backtesting with walk-forward validation
   • Investment recommendations with confidence scoring
   • Risk assessment and performance analytics

📊 SUPPORTED DATA SOURCES:
   • Alpha Vantage API (technical & fundamental data)
   • Yahoo Finance (price data & news)
   • Quandl (economic indicators & market data)
   • Market sentiment indicators (VIX, Put/Call ratio, etc.)

🔮 FORECASTING MODELS:
   • Vector Autoregression (VAR) - Chincarini methodology
   • Random Forest - Tree-based ensemble learning
   • XGBoost - Gradient boosting framework
   • LSTM Neural Networks - Deep learning for sequences
   • Ensemble Methods - Combined model predictions

📈 OUTPUT FEATURES:
   • Price forecasts with confidence intervals
   • Buy/Sell/Hold investment signals
   • Risk-adjusted return calculations
   • Performance backtesting reports
   • Interactive visualizations
    """)

def main():
    """
    Main demo function
    """
    # System overview
    demo_system_overview()
    
    # Demo configuration
    DEMO_TICKER = "AAPL"  # Apple Inc. - good for demo due to data availability
    QUICK_DEMO = True     # Set to False for full backtesting (takes much longer)
    
    print(f"\n{'='*80}")
    print(f"🎮 STARTING DEMO SESSION")
    print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Demo Ticker: {DEMO_TICKER}")
    print(f"⚡ Quick Demo Mode: {QUICK_DEMO}")
    print(f"{'='*80}")
    
    # Part 1: Single Stock Analysis
    print(f"\n🔥 PART 1: COMPREHENSIVE STOCK ANALYSIS")
    recommendation = demo_single_stock_analysis(DEMO_TICKER)
    
    if recommendation:
        # Part 2: Backtesting Demo
        print(f"\n🔥 PART 2: BACKTESTING FRAMEWORK")
        backtest_results = demo_backtesting([DEMO_TICKER], quick_test=QUICK_DEMO)
        
        # Summary
        print(f"\n{'='*80}")
        print(f"✅ DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print(f"""
📁 Generated Files:
   • demo_forecast_{DEMO_TICKER}.png - Forecast visualization
   • demo_analysis_{DEMO_TICKER}/ - Complete analysis results
   • Data cache files for faster future runs

🚀 Next Steps:
   1. Modify config.py with your API keys
   2. Explore different tickers and time periods
   3. Customize factor extraction and model parameters
   4. Run full backtesting for comprehensive evaluation
   5. Integrate with your trading/analysis workflow

💡 Tips:
   • Start with liquid, well-covered stocks (AAPL, GOOGL, MSFT)
   • Allow time for data collection on first runs
   • Review model diagnostics for statistical validity
   • Consider transaction costs in trading strategies
        """)
        
        if recommendation['recommendation'] in ['BUY', 'STRONG BUY']:
            print(f"🎯 The model suggests {recommendation['recommendation']} for {DEMO_TICKER}")
            print(f"   Target: ${recommendation['target_price_1m']:.2f} (1-month)")
            print(f"   Expected return: {recommendation['expected_return_1m']:.2%}")
        
    else:
        print(f"\n❌ Demo encountered issues. Please check your API keys and connection.")
        
    print(f"\n🎉 Thank you for trying the AI-Powered Stock Forecasting System!")
    print(f"📧 Questions? Issues? Check the README.md for support information.")

if __name__ == "__main__":
    # Run the demo
    main() 