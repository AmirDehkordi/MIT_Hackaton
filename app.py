import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
import time

# Import your custom modules
from main_forecasting_system import ComprehensiveStockForecaster
from ticker_utils import get_all_tickers

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Helper Functions ---
@st.cache_data(show_spinner=False, ttl=3600)
def run_forecasting_pipeline(ticker):
    """Runs the full analysis with multi-horizon forecasts."""
    try:
        forecaster = ComprehensiveStockForecaster(ticker)
        
        # Collect data with error handling
        data = forecaster.collect_all_data()
        if data is None or 'price_data' not in data:
            return None, None, None, "Failed to collect data"
            
        # Extract factors
        factors = forecaster.extract_all_factors()
        
        # Build models
        forecaster.build_var_model()
        forecaster.train_ml_models()
        
        # Generate multi-horizon forecasts
        multi_recommendations = forecaster.get_multi_horizon_recommendations()
        
        # Get single recommendation for backward compatibility
        single_rec = forecaster.get_investment_recommendation()
        
        return forecaster, single_rec, multi_recommendations, None
        
    except Exception as e:
        return None, None, None, str(e)

def create_forecast_chart(price_data, forecast_data, ticker_symbol, horizon_name):
    """Creates a forecast chart for a specific time horizon."""
    
    fig = go.Figure()
    
    # Historical prices (last 60 days)
    historical = price_data.tail(60)
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical['Close'],
        mode='lines',
        name='Historical',
        line=dict(color='lightblue', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Predicted_Price'],
        mode='lines',
        name='Forecast',
        line=dict(color='orange', width=2)
    ))
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Upper_Bound'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,165,0,0.2)'),
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Lower_Bound'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,165,0,0.2)'),
        name='Confidence Band'
    ))
    
    fig.update_layout(
        title=f"{ticker_symbol} - {horizon_name.replace('_', ' ').title()} Forecast",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        template='plotly_dark',
        height=400
    )
    
    return fig

def create_technical_chart(price_data, ticker_symbol):
    """Creates a technical analysis chart."""
    
    # Calculate indicators
    price_data['SMA_20'] = ta.trend.sma_indicator(price_data['Close'], window=20)
    price_data['SMA_50'] = ta.trend.sma_indicator(price_data['Close'], window=50)
    price_data['RSI'] = ta.momentum.rsi(price_data['Close'], window=14)
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=('Price & Moving Averages', 'Volume', 'RSI'),
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Price and MAs
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['Open'],
        high=price_data['High'],
        low=price_data['Low'],
        close=price_data['Close'],
        name='Price'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='yellow', width=1)
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='orange', width=1)
    ), row=1, col=1)
    
    # Volume
    fig.add_trace(go.Bar(
        x=price_data.index,
        y=price_data['Volume'],
        name='Volume',
        marker_color='lightblue'
    ), row=2, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='purple', width=1.5)
    ), row=3, col=1)
    
    # RSI levels
    fig.add_hline(y=70, row=3, col=1, line=dict(color='red', dash='dash', width=1))
    fig.add_hline(y=30, row=3, col=1, line=dict(color='green', dash='dash', width=1))
    
    fig.update_layout(
        title_text=f"{ticker_symbol} Technical Analysis",
        template='plotly_dark',
        height=700,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

# --- Main App ---
st.title('🚀 AI-Powered Stock Forecaster')
st.markdown("### Chincarini VAR + ML Ensemble Forecasting System")

# Ticker selection
col1, col2 = st.columns([3, 1])

with col1:
    # Get ticker list with search functionality
    search_query = st.text_input("🔍 Search for a stock (e.g., 'GOOG' or 'Apple')", "")
    try:
        ticker_data = get_all_tickers(search_query)
        display_options = list(ticker_data.keys())
        if not display_options and search_query:
            st.info("No matches found. Try a different search term.")
            # Show some popular stocks as suggestions
            ticker_data = {
                "AAPL - Apple Inc.": "AAPL",
                "MSFT - Microsoft Corporation": "MSFT",
                "GOOGL - Alphabet Inc.": "GOOGL",
                "AMZN - Amazon.com Inc.": "AMZN",
                "NVDA - NVIDIA Corporation": "NVDA",
                "TSLA - Tesla Inc.": "TSLA",
                "META - Meta Platforms Inc.": "META"
            }
            display_options = list(ticker_data.keys())
    except Exception as e:
        st.warning(f"Could not fetch full ticker list: {e}")
        # Fallback options
        ticker_data = {
            "AAPL - Apple Inc.": "AAPL",
            "MSFT - Microsoft Corporation": "MSFT",
            "GOOGL - Alphabet Inc.": "GOOGL",
            "AMZN - Amazon.com Inc.": "AMZN",
            "NVDA - NVIDIA Corporation": "NVDA",
            "TSLA - Tesla Inc.": "TSLA",
            "META - Meta Platforms Inc.": "META"
        }
        display_options = list(ticker_data.keys())
    
    selected_display = st.selectbox(
        "🔍 Search for a stock",
        options=display_options,
        index=0
    )
    
    ticker_symbol = ticker_data.get(selected_display)

with col2:
    analyze_button = st.button("🔮 Analyze", type="primary", use_container_width=True)

# Analysis section
if analyze_button and ticker_symbol:
    st.divider()
    
    # Progress indicator
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Run analysis
    status_text.text(f"📊 Analyzing {ticker_symbol}...")
    progress_bar.progress(25)
    
    forecaster, single_rec, multi_rec, error = run_forecasting_pipeline(ticker_symbol)
    
    if error:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ Error analyzing {ticker_symbol}: {error}")
        st.info("💡 Tips: Check if the ticker is valid, ensure you have API access, or wait if rate-limited.")
    
    elif forecaster and multi_rec:
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        time.sleep(1)
        progress_bar.empty()
        status_text.empty()
        
        # Display current metrics
        st.markdown("### 📊 Current Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Current Price", f"${single_rec['current_price']:.2f}")
        col2.metric("Confidence", f"{single_rec['confidence_score']:.1%}")
        col3.metric("Volatility", f"{single_rec['volatility']:.1%}")
        col4.metric("Overall Signal", single_rec['recommendation'])
        
        st.divider()
        
        # Multi-horizon forecasts
        st.markdown("### 📈 Multi-Horizon Forecasts")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📈 Short-Term", "📉 Mid-Term", "📊 Long-Term"])
        
        with tab1:
            # Summary metrics for all horizons
            st.markdown("#### Forecast Summary")
            
            summary_cols = st.columns(3)
            
            for idx, (horizon, data) in enumerate(multi_rec.items()):
                with summary_cols[idx]:
                    horizon_display = horizon.replace('_', ' ').title()
                    st.markdown(f"**{horizon_display}** ({data['days']} days)")
                    
                    delta_color = "normal" if data['expected_return'] > 0 else "inverse"
                    st.metric(
                        "Target Price",
                        f"${data['target_price']:.2f}",
                        delta=f"{data['expected_return']:.1%}",
                        delta_color=delta_color
                    )
                    st.metric("Signal", data['signal'])
                    st.metric("Confidence", f"{data['confidence']:.1%}")
            
            # Technical chart
            st.markdown("#### Technical Analysis")
            tech_chart = create_technical_chart(
                forecaster.raw_data['price_data'].tail(100),
                ticker_symbol
            )
            st.plotly_chart(tech_chart, use_container_width=True)
        
        with tab2:
            # Short-term forecast
            st.markdown("#### Short-Term Forecast (1 Week)")
            short_data = multi_rec['short_term']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", f"${short_data['target_price']:.2f}")
            col2.metric("Return", f"{short_data['expected_return']:.2%}")
            col3.metric("Signal", short_data['signal'])
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                short_data['forecast_df'],
                ticker_symbol,
                "short_term"
            )
            st.plotly_chart(chart, use_container_width=True)
        
        with tab3:
            # Mid-term forecast
            st.markdown("#### Mid-Term Forecast (1 Month)")
            mid_data = multi_rec['mid_term']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", f"${mid_data['target_price']:.2f}")
            col2.metric("Return", f"{mid_data['expected_return']:.2%}")
            col3.metric("Signal", mid_data['signal'])
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                mid_data['forecast_df'],
                ticker_symbol,
                "mid_term"
            )
            st.plotly_chart(chart, use_container_width=True)
        
        with tab4:
            # Long-term forecast
            st.markdown("#### Long-Term Forecast (3 Months)")
            long_data = multi_rec['long_term']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Target", f"${long_data['target_price']:.2f}")
            col2.metric("Return", f"{long_data['expected_return']:.2%}")
            col3.metric("Signal", long_data['signal'])
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                long_data['forecast_df'],
                ticker_symbol,
                "long_term"
            )
            st.plotly_chart(chart, use_container_width=True)
        
        # Model performance section
        with st.expander("📊 Model Performance Details"):
            if hasattr(forecaster, 'model_performance') and not forecaster.model_performance.empty:
                st.dataframe(forecaster.model_performance)
            else:
                st.info("Model performance metrics will appear here after analysis.")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center'>
    <p>Built with Chincarini VAR methodology + ML ensemble (RF, XGBoost, LSTM)</p>
    <p>📊 Technical • 💰 Fundamental • 😊 Sentiment Factors</p>
</div>
""", unsafe_allow_html=True)

