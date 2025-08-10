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
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for TradingView-like styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #1a1d29 100%);
    }
    .search-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(45deg, #ff6b35, #f7931e);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 53, 0.3);
    }
    .title-container {
        text-align: center;
        margin-bottom: 30px;
    }
    .subtitle {
        color: #8b949e;
        font-size: 18px;
        margin-top: -10px;
    }
</style>
""", unsafe_allow_html=True)

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
    """Creates a TradingView-style forecast chart for a specific time horizon."""
    
    fig = go.Figure()
    
    # Historical prices (last 60 days)
    historical = price_data.tail(60)
    fig.add_trace(go.Scatter(
        x=historical.index,
        y=historical['Close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#00d4aa', width=3),
        hovertemplate='<b>%{x}</b><br>Price: $%{y:.2f}<extra></extra>'
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Predicted_Price'],
        mode='lines',
        name='AI Forecast',
        line=dict(color='#ff6b35', width=3, dash='dot'),
        hovertemplate='<b>%{x}</b><br>Forecast: $%{y:.2f}<extra></extra>'
    ))
    
    # Confidence bands
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Upper_Bound'],
        fill=None,
        mode='lines',
        line=dict(color='rgba(255,107,53,0.1)', width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_data.index,
        y=forecast_data['Lower_Bound'],
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255,107,53,0.1)', width=0),
        name='Confidence Band',
        fillcolor='rgba(255,107,53,0.15)',
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title={
            'text': f"{ticker_symbol} - {horizon_name.replace('_', ' ').title()} Forecast",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'white'}
        },
        xaxis=dict(
            title="Date",
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title="Price ($)",
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True,
            zeroline=False
        ),
        template='plotly_dark',
        height=450,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def create_technical_chart(price_data, ticker_symbol):
    """Creates a TradingView-style technical analysis chart."""
    
    # Calculate technical indicators
    price_data = price_data.copy()
    price_data['SMA_20'] = ta.trend.sma_indicator(price_data['Close'], window=20)
    price_data['SMA_50'] = ta.trend.sma_indicator(price_data['Close'], window=50)
    price_data['RSI'] = ta.momentum.rsi(price_data['Close'], window=14)
    price_data['MACD'] = ta.trend.macd_diff(price_data['Close'])
    
    # Create subplots with TradingView-like styling
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(
            f'{ticker_symbol} - Price & Moving Averages', 
            'Relative Strength Index (RSI)', 
            'MACD Histogram'
        ),
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Price and moving averages with enhanced styling
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='#00d4aa', width=3),
        hovertemplate='<b>%{x}</b><br>Close: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['SMA_20'],
        mode='lines',
        name='SMA 20',
        line=dict(color='#f7931e', width=2),
        hovertemplate='<b>%{x}</b><br>SMA 20: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['SMA_50'],
        mode='lines',
        name='SMA 50',
        line=dict(color='#ff6b35', width=2),
        hovertemplate='<b>%{x}</b><br>SMA 50: $%{y:.2f}<extra></extra>'
    ), row=1, col=1)
    
    # RSI with enhanced styling
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['RSI'],
        mode='lines',
        name='RSI',
        line=dict(color='#9c88ff', width=2),
        hovertemplate='<b>%{x}</b><br>RSI: %{y:.1f}<extra></extra>'
    ), row=2, col=1)
    
    # RSI levels with better colors
    fig.add_hline(y=70, line_dash="dash", line_color="#ff4757", line_width=1, row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="#2ed573", line_width=1, row=2, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="#747d8c", line_width=1, row=2, col=1)
    
    # MACD with enhanced styling
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['MACD'],
        mode='lines',
        name='MACD',
        line=dict(color='#3742fa', width=2),
        hovertemplate='<b>%{x}</b><br>MACD: %{y:.3f}<extra></extra>'
    ), row=3, col=1)
    
    fig.update_layout(
        title={
            'text': f"{ticker_symbol} Technical Analysis",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 22, 'color': 'white'}
        },
        template='plotly_dark',
        height=650,
        showlegend=True,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update axes styling
    fig.update_xaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    fig.update_yaxes(
        gridcolor='rgba(255,255,255,0.1)',
        showgrid=True,
        zeroline=False
    )
    
    return fig

# --- Main App ---
st.markdown("""
<div class="title-container">
    <h1 style="font-size: 3rem; margin-bottom: 10px;">🚀 AI-Powered Stock Forecaster</h1>
    <p class="subtitle">Advanced ML Ensemble Forecasting System</p>
</div>
""", unsafe_allow_html=True)

# Stock Search Section
col1, col2 = st.columns([4, 1])

with col1:
    # Get all available tickers
    all_tickers = get_all_tickers()
    display_options = list(all_tickers.keys())
    
    # Single search selectbox with all stocks
    selected_display = st.selectbox(
        "🔍 Search for a stock:",
        options=display_options,
        index=0,
        help="Type to search by ticker symbol or company name"
    )
    
    ticker_symbol = all_tickers.get(selected_display)

with col2:
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    analyze_button = st.button(
        "📊 Analyze Stock", 
        type="primary", 
        use_container_width=True,
        disabled=not ticker_symbol
    )

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
        
        # Display current metrics with enhanced styling
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; margin: 20px 0;">
            <h3 style="margin-bottom: 20px; text-align: center;">📊 Current Analysis</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("💰 Current Price", f"${single_rec['current_price']:.2f}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Confidence", f"{single_rec['confidence_score']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📈 Volatility", f"{single_rec['volatility']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            signal_color = "🟢" if single_rec['recommendation'] == "BUY" else "🔴" if single_rec['recommendation'] == "SELL" else "🟡"
            st.metric(f"{signal_color} Signal", single_rec['recommendation'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.divider()
        
        # Multi-horizon forecasts with enhanced styling
        st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; margin: 20px 0;">
            <h3 style="margin-bottom: 20px; text-align: center;">📈 Multi-Horizon Forecasts</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "⚡ Short-Term (1W)", "📈 Mid-Term (1M)", "🎯 Long-Term (3M)"])
        
        with tab1:
            # Enhanced summary metrics for all horizons
            st.markdown("#### 🎯 Forecast Overview")
            
            summary_cols = st.columns(3)
            
            for idx, (horizon, data) in enumerate(multi_rec.items()):
                with summary_cols[idx]:
                    horizon_display = horizon.replace('_', ' ').title()
                    
                    # Enhanced card styling
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**{horizon_display}** ({data['days']} days)")
                    
                    delta_color = "normal" if data['expected_return'] > 0 else "inverse"
                    st.metric(
                        "🎯 Target Price",
                        f"${data['target_price']:.2f}",
                        delta=f"{data['expected_return']:.1%}",
                        delta_color=delta_color
                    )
                    
                    signal_emoji = "🟢" if data['signal'] == "BUY" else "🔴" if data['signal'] == "SELL" else "🟡"
                    st.markdown(f"**Signal:** {signal_emoji} {data['signal']}")
                    st.markdown(f"**Confidence:** {data['confidence']:.1%}")
                    st.markdown('</div>', unsafe_allow_html=True)
            
            # Technical chart with enhanced styling
            st.markdown("#### 📊 Technical Analysis")
            tech_chart = create_technical_chart(
                forecaster.raw_data['price_data'].tail(100),
                ticker_symbol
            )
            st.plotly_chart(tech_chart, use_container_width=True, config={'displayModeBar': False})
        
        with tab2:
            # Short-term forecast with enhanced styling
            st.markdown("#### ⚡ Short-Term Forecast (1 Week)")
            short_data = multi_rec['short_term']
            
            # Enhanced metrics layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Target Price", f"${short_data['target_price']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                delta_color = "normal" if short_data['expected_return'] > 0 else "inverse"
                st.metric("📈 Expected Return", f"{short_data['expected_return']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                signal_emoji = "🟢" if short_data['signal'] == "BUY" else "🔴" if short_data['signal'] == "SELL" else "🟡"
                st.markdown(f"**{signal_emoji} Signal:** {short_data['signal']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                short_data['forecast_df'],
                ticker_symbol,
                "short_term"
            )
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        with tab3:
            # Mid-term forecast with enhanced styling
            st.markdown("#### 📈 Mid-Term Forecast (1 Month)")
            mid_data = multi_rec['mid_term']
            
            # Enhanced metrics layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Target Price", f"${mid_data['target_price']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 Expected Return", f"{mid_data['expected_return']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                signal_emoji = "🟢" if mid_data['signal'] == "BUY" else "🔴" if mid_data['signal'] == "SELL" else "🟡"
                st.markdown(f"**{signal_emoji} Signal:** {mid_data['signal']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                mid_data['forecast_df'],
                ticker_symbol,
                "mid_term"
            )
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        with tab4:
            # Long-term forecast with enhanced styling
            st.markdown("#### 🎯 Long-Term Forecast (3 Months)")
            long_data = multi_rec['long_term']
            
            # Enhanced metrics layout
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("🎯 Target Price", f"${long_data['target_price']:.2f}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("📈 Expected Return", f"{long_data['expected_return']:.2%}")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                signal_emoji = "🟢" if long_data['signal'] == "BUY" else "🔴" if long_data['signal'] == "SELL" else "🟡"
                st.markdown(f"**{signal_emoji} Signal:** {long_data['signal']}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            chart = create_forecast_chart(
                forecaster.raw_data['price_data'],
                long_data['forecast_df'],
                ticker_symbol,
                "long_term"
            )
            st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})
        
        # Model performance section
        with st.expander("📊 Model Performance Details"):
            if hasattr(forecaster, 'model_performance') and not forecaster.model_performance.empty:
                st.dataframe(forecaster.model_performance)
            else:
                st.info("Model performance metrics will appear here after analysis.")

# Enhanced Footer
st.divider()
st.markdown("""
<div style='text-align: center; padding: 20px; background: rgba(255, 255, 255, 0.02); border-radius: 8px; margin-top: 30px;'>
    <p style='font-size: 16px; margin-bottom: 10px;'>Built with Advanced ML Ensemble Forecasting</p>
    <p style='color: #8b949e;'>📊 Technical Analysis • 📈 VAR Methodology • 🤖 ML Models (RF, XGBoost, LSTM)</p>
    <p style='color: #6b7280; font-size: 14px; margin-top: 15px;'>⚠️ For educational purposes only. Not financial advice.</p>
</div>
""", unsafe_allow_html=True)

