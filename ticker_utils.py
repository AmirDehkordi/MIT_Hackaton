import pandas as pd
import streamlit as st
import requests

@st.cache_data(ttl=3600)
def get_all_tickers(search_query=None):
    """
    Get NASDAQ tickers with robust fallback and improved search
    """
    try:
        # Try to get from NASDAQ API
        url = 'https://api.nasdaq.com/api/screener/stocks?tableonly=true&limit=2500&exchange=nasdaq&download=true'
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        rows = data.get('data', {}).get('rows', [])
        
        if rows:
            nasdaq_df = pd.DataFrame(rows)
            nasdaq_df = nasdaq_df[['symbol', 'name']].dropna()
            nasdaq_df['display'] = nasdaq_df['symbol'] + ' - ' + nasdaq_df['name'].str[:50]
            
            if search_query:
                # Case-insensitive search in both symbol and name
                search_query = search_query.lower()
                mask = (nasdaq_df['symbol'].str.lower().str.contains(search_query)) | \
                      (nasdaq_df['name'].str.lower().str.contains(search_query))
                nasdaq_df = nasdaq_df[mask]
                
                # Sort by relevance: exact symbol matches first, then symbol starts-with,
                # then name matches
                exact_symbol_match = nasdaq_df['symbol'].str.lower() == search_query
                symbol_startswith = nasdaq_df['symbol'].str.lower().str.startswith(search_query)
                
                nasdaq_df['sort_key'] = (exact_symbol_match.astype(int) * 3 + 
                                       symbol_startswith.astype(int) * 2)
                nasdaq_df = nasdaq_df.sort_values('sort_key', ascending=False)
                
            return pd.Series(nasdaq_df.symbol.values, index=nasdaq_df.display).to_dict()
    
    except Exception as e:
        st.warning(f"Using fallback ticker list: {e}")
    
    # Comprehensive fallback list
    return {
        "AAPL - Apple Inc.": "AAPL",
        "MSFT - Microsoft Corporation": "MSFT",
        "GOOGL - Alphabet Inc.": "GOOGL",
        "AMZN - Amazon.com Inc.": "AMZN",
        "NVDA - NVIDIA Corporation": "NVDA",
        "TSLA - Tesla Inc.": "TSLA",
        "META - Meta Platforms Inc.": "META",
        "BRK.B - Berkshire Hathaway": "BRK-B",
        "JNJ - Johnson & Johnson": "JNJ",
        "V - Visa Inc.": "V",
        "JPM - JPMorgan Chase": "JPM",
        "WMT - Walmart Inc.": "WMT",
        "PG - Procter & Gamble": "PG",
        "MA - Mastercard": "MA",
        "AVGO - Broadcom Inc.": "AVGO",
        "HD - Home Depot": "HD",
        "DIS - Walt Disney": "DIS",
        "ADBE - Adobe Inc.": "ADBE",
        "NFLX - Netflix Inc.": "NFLX",
        "CRM - Salesforce": "CRM"
    }
