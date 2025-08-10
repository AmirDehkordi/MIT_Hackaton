
import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def get_sp500_tickers():
    """
    Scrapes the S&P 500 tickers and company names from Wikipedia.
    Returns a dictionary mapping 'Ticker - Company Name' to 'Ticker'.
    """
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        sp500_df = tables[0]
        
        # Create the formatted string for display and the ticker for backend use
        sp500_df['display'] = sp500_df['Symbol'] + ' - ' + sp500_df['Security']
        
        # Create a dictionary for easy lookup
        ticker_dict = pd.Series(sp500_df.Symbol.values, index=sp500_df.display).to_dict()
        return ticker_dict
    except Exception as e:
        # Fallback list in case Wikipedia scraping fails
        st.error(f"Could not fetch S&P 500 list: {e}. Using a fallback list.")
        return {
            "AAPL - Apple Inc.": "AAPL",
            "MSFT - Microsoft Corporation": "MSFT",
            "GOOGL - Alphabet Inc.": "GOOGL",
            "AMZN - Amazon.com, Inc.": "AMZN",
            "NVDA - NVIDIA Corporation": "NVDA",
            "TSLA - Tesla, Inc.": "TSLA"
        }
