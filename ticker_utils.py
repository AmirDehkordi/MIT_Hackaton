import pandas as pd
import streamlit as st

@st.cache_data(ttl=3600)  # Cache the data for 1 hour
def get_nasdaq_tickers():
    """
    Scrapes all NASDAQ-listed tickers and company names.
    Returns a dictionary mapping 'Ticker - Company Name' to 'Ticker'.
    """
    try:
        # Using a reliable source for NASDAQ-listed stocks
        url = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=2500&exchange=nasdaq&download=true'
        headers = {'User-Agent': 'Mozilla/5.0'} # NASDAQ API requires a user-agent header
        
        # The request returns a JSON object which we can directly process
        import requests
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Navigate the JSON structure to get the list of stocks
        rows = data.get('data', {}).get('rows', [])
        if not rows:
            raise ValueError("Could not find stock rows in NASDAQ API response.")
            
        nasdaq_df = pd.DataFrame(rows)

        # Clean up the data
        nasdaq_df = nasdaq_df[['symbol', 'name']].dropna()
        nasdaq_df['name'] = nasdaq_df['name'].str.replace(' Common Stock', '').str.strip()
        
        # Create the formatted string for display
        nasdaq_df['display'] = nasdaq_df['symbol'] + ' - ' + nasdaq_df['name']
        
        # Create a dictionary for easy lookup
        ticker_dict = pd.Series(nasdaq_df.symbol.values, index=nasdaq_df.display).to_dict()
        return ticker_dict
        
    except Exception as e:
        # Fallback list in case API scraping fails
        st.error(f"Could not fetch NASDAQ list: {e}. Using a fallback list.")
        return {
            "AAPL - Apple Inc.": "AAPL",
            "MSFT - Microsoft Corporation": "MSFT",
            "GOOGL - Alphabet Inc.": "GOOGL",
            "AMZN - Amazon.com, Inc.": "AMZN",
            "NVDA - NVIDIA Corporation": "NVDA",
            "TSLA - Tesla, Inc.": "TSLA"
        }

# We can rename the main function for clarity
get_all_tickers = get_nasdaq_tickers
