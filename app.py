# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta  # Make sure 'ta' is in your requirements.txt

# Import your main forecaster class
from main_forecasting_system import ComprehensiveStockForecaster

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Caching Functions ---
# Use Streamlit's caching to avoid re-running the whole pipeline on every interaction.
@st.cache_data(show_spinner=False)
def run_forecasting_pipeline(_ticker):
    """
    Runs the full analysis and returns the forecaster object.
    The _ticker argument is used to create a unique cache key.
    """
    try:
        forecaster = ComprehensiveStockForecaster(_ticker)
        forecaster.collect_all_data()
        forecaster.extract_all_factors()
        forecaster.build_var_model()
        forecaster.train_ml_models()
        forecaster.generate_comprehensive_forecast()
        recommendation = forecaster.get_investment_recommendation()
        return forecaster, recommendation, None  # Return forecaster, recommendation, and no error
    except Exception as e:
        return None, None, str(e) # Return None, None, and the error message


# --- Charting Function ---
def create_trading_view_chart(price_data):
    """Creates a TradingView-style chart with Price, MAs, and RSI."""
    # Calculate indicators
    price_data['SMA_20'] = ta.trend.sma_indicator(price_data['Close'], window=20)
    price_data['SMA_50'] = ta.trend.sma_indicator(price_data['Close'], window=50)
    price_data['RSI'] = ta.momentum.rsi(price_data['Close'], window=14)

    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, subplot_titles=('Price Chart', 'RSI'),
                        row_heights=[0.8, 0.2])

    # Plot Price Candlestick
    fig.add_trace(go.Candlestick(x=price_data.index,
                                 open=price_data['Open'],
                                 high=price_data['High'],
                                 low=price_data['Low'],
                                 close=price_data['Close'],
                                 name='Price'), row=1, col=1)

    # Plot Moving Averages
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_20'],
                             mode='lines', name='SMA 20', line=dict(color='yellow', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_50'],
                             mode='lines', name='SMA 50', line=dict(color='orange', width=1)), row=1, col=1)

    # Plot RSI
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'],
                             mode='lines', name='RSI', line=dict(color='cyan', width=1)), row=2, col=1)
    # Add RSI horizontal lines
    fig.add_hline(y=70, col=1, row=2, line_color="#ff4444", line_width=1, line_dash="dash")
    fig.add_hline(y=30, col=1, row=2, line_color="#44ff44", line_width=1, line_dash="dash")


    # Update layout for a professional look
    fig.update_layout(
        title_text=f"{ticker.upper()} Price Analysis",
        height=600,
        showlegend=True,
        template='plotly_dark',  # Dark theme like TradingView
        xaxis_rangeslider_visible=False,
        yaxis1_title="Price ($)",
        yaxis2_title="RSI"
    )

    return fig


# --- Main Application UI ---
st.title('AI-Powered Stock Forecaster 📈')

# Central search bar
st.markdown("<br>", unsafe_allow_html=True) # A little vertical space
col1, col2, col3 = st.columns([2,3,2])
with col2:
    ticker = st.text_input("Enter a stock ticker to analyze (e.g., AAPL, NVDA, TSLA)",
                           value="AAPL",
                           key="ticker_input").upper()
    analyze_button = st.button("Analyze Stock", use_container_width=True, type="primary")

st.markdown("---")


# --- Analysis and Display Area ---
if analyze_button:
    with st.spinner(f'Running full analysis for **{ticker}**... This may take a moment.'):
        forecaster, recommendation, error = run_forecasting_pipeline(ticker)

    if error:
        st.error(f"An error occurred while analyzing {ticker}: {error}")
        st.warning("Please ensure the ticker is correct and data is available. Try another ticker like 'MSFT' or 'GOOGL'.")
    else:
        st.success(f"Analysis for **{ticker}** complete!")

        # Display key metrics
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Recommendation", recommendation['recommendation'])
        m_col2.metric("Current Price", f"${recommendation['current_price']:.2f}")
        m_col3.metric(
            "1-Month Forecast",
            f"${recommendation['target_price_1m']:.2f}",
            delta=f"{recommendation['expected_return_1m']:.2%}"
        )

        # Display the chart
        price_data = forecaster.raw_data['price_data']
        fig = create_trading_view_chart(price_data)
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Enter a stock ticker above and click 'Analyze Stock' to see the forecast and analysis.")
