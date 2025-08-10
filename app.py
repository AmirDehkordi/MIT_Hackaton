# app.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta

# Import your custom modules
from main_forecasting_system import ComprehensiveStockForecaster
from ticker_utils import get_sp500_tickers # Import the new utility

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Stock Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Caching Functions ---
@st.cache_data(show_spinner=False)
def run_forecasting_pipeline(ticker):
    """Runs the full analysis and returns the forecaster object and recommendation."""
    try:
        forecaster = ComprehensiveStockForecaster(ticker)
        forecaster.collect_all_data()
        forecaster.extract_all_factors()
        forecaster.build_var_model()
        forecaster.train_ml_models()
        forecaster.generate_comprehensive_forecast()
        recommendation = forecaster.get_investment_recommendation()
        return forecaster, recommendation, None
    except Exception as e:
        return None, None, str(e)

# --- Charting Function ---
def create_trading_view_chart(price_data, ticker_symbol):
    """Creates a TradingView-style chart with Price, MAs, and RSI."""
    price_data['SMA_20'] = ta.trend.sma_indicator(price_data['Close'], window=20)
    price_data['SMA_50'] = ta.trend.sma_indicator(price_data['Close'], window=50)
    price_data['RSI'] = ta.momentum.rsi(price_data['Close'], window=14)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05, subplot_titles=('Price Chart', 'RSI'),
                        row_heights=[0.8, 0.2])

    fig.add_trace(go.Candlestick(x=price_data.index,
                                 open=price_data['Open'], high=price_data['High'],
                                 low=price_data['Low'], close=price_data['Close'],
                                 name='Price'), row=1, col=1)

    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_20'],
                             mode='lines', name='SMA 20', line=dict(color='yellow', width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['SMA_50'],
                             mode='lines', name='SMA 50', line=dict(color='orange', width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(x=price_data.index, y=price_data['RSI'],
                             mode='lines', name='RSI', line=dict(color='cyan', width=1.5)), row=2, col=1)
    fig.add_hline(y=70, col=1, row=2, line_color="#ff4444", line_width=1, line_dash="dash")
    fig.add_hline(y=30, col=1, row=2, line_color="#44ff44", line_width=1, line_dash="dash")

    fig.update_layout(
        title_text=f"{ticker_symbol} Price Analysis",
        height=600,
        showlegend=True,
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        yaxis1_title="Price ($)",
        yaxis2_title="RSI",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# --- Main Application UI ---
st.title('AI-Powered Stock Forecaster 📈')

# --- NEW SEARCH BOX ---
# Load the ticker data
ticker_data = get_sp500_tickers()
display_options = list(ticker_data.keys())

# Use a selectbox for a searchable dropdown
selected_display = st.selectbox(
    "Search for a stock from the S&P 500",
    options=display_options,
    index=display_options.index("AAPL - Apple Inc.") # Default to Apple
)

# Extract the ticker symbol from the selected option
ticker_symbol = ticker_data[selected_display]

# --- Analysis and Display Area ---
if ticker_symbol:
    with st.spinner(f'Running full analysis for **{ticker_symbol}**... This may take a moment.'):
        forecaster, recommendation, error = run_forecasting_pipeline(ticker_symbol)

    if error:
        st.error(f"An error occurred while analyzing {ticker_symbol}: {error}")
        st.warning("Please ensure the ticker is correct and data is available. The API may be rate-limiting. Try another ticker.")
    elif forecaster:
        st.success(f"Analysis for **{ticker_symbol}** complete!")

        col1, col2, col3 = st.columns(3)
        col1.metric("Recommendation", recommendation['recommendation'])
        col2.metric("Current Price", f"${recommendation['current_price']:.2f}")
        col3.metric(
            "1-Month Forecast",
            f"${recommendation['target_price_1m']:.2f}",
            delta=f"{recommendation['expected_return_1m']:.2%}"
        )

        price_data = forecaster.raw_data['price_data']
        fig = create_trading_view_chart(price_data, ticker_symbol)
        st.plotly_chart(fig, use_container_width=True)
