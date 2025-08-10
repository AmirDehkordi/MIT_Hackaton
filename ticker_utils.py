import pandas as pd
import streamlit as st
import requests

@st.cache_data(ttl=3600)
def get_all_tickers(search_query=None):
    """
    Get comprehensive stock tickers with robust fallback
    """
    # Comprehensive stock list - always return this for reliability
    comprehensive_tickers = {
        "AAPL - Apple Inc.": "AAPL",
        "MSFT - Microsoft Corporation": "MSFT",
        "GOOGL - Alphabet Inc.": "GOOGL",
        "GOOG - Alphabet Inc. Class C": "GOOG",
        "AMZN - Amazon.com Inc.": "AMZN",
        "NVDA - NVIDIA Corporation": "NVDA",
        "TSLA - Tesla Inc.": "TSLA",
        "META - Meta Platforms Inc.": "META",
        "NFLX - Netflix Inc.": "NFLX",
        "BRK.B - Berkshire Hathaway": "BRK-B",
        "JNJ - Johnson & Johnson": "JNJ",
        "V - Visa Inc.": "V",
        "JPM - JPMorgan Chase & Co.": "JPM",
        "WMT - Walmart Inc.": "WMT",
        "PG - Procter & Gamble Co.": "PG",
        "MA - Mastercard Inc.": "MA",
        "AVGO - Broadcom Inc.": "AVGO",
        "HD - Home Depot Inc.": "HD",
        "DIS - Walt Disney Co.": "DIS",
        "ADBE - Adobe Inc.": "ADBE",
        "CRM - Salesforce Inc.": "CRM",
        "PYPL - PayPal Holdings Inc.": "PYPL",
        "INTC - Intel Corporation": "INTC",
        "AMD - Advanced Micro Devices": "AMD",
        "ORCL - Oracle Corporation": "ORCL",
        "IBM - International Business Machines": "IBM",
        "CSCO - Cisco Systems Inc.": "CSCO",
        "QCOM - QUALCOMM Inc.": "QCOM",
        "TXN - Texas Instruments Inc.": "TXN",
        "COST - Costco Wholesale Corp.": "COST",
        "ABBV - AbbVie Inc.": "ABBV",
        "PEP - PepsiCo Inc.": "PEP",
        "KO - Coca-Cola Co.": "KO",
        "TMO - Thermo Fisher Scientific": "TMO",
        "MRK - Merck & Co. Inc.": "MRK",
        "CVX - Chevron Corporation": "CVX",
        "LLY - Eli Lilly and Co.": "LLY",
        "ACN - Accenture plc": "ACN",
        "AVGO - Broadcom Inc.": "AVGO",
        "NOW - ServiceNow Inc.": "NOW",
        "CRM - Salesforce Inc.": "CRM",
        "INTU - Intuit Inc.": "INTU",
        "ISRG - Intuitive Surgical": "ISRG",
        "BKNG - Booking Holdings Inc.": "BKNG",
        "GILD - Gilead Sciences Inc.": "GILD",
        "MDLZ - Mondelez International": "MDLZ",
        "ADP - Automatic Data Processing": "ADP",
        "CME - CME Group Inc.": "CME",
        "LRCX - Lam Research Corp.": "LRCX",
        "AMAT - Applied Materials Inc.": "AMAT",
        "SBUX - Starbucks Corporation": "SBUX",
        "REGN - Regeneron Pharmaceuticals": "REGN",
        "KLAC - KLA Corporation": "KLAC",
        "MELI - MercadoLibre Inc.": "MELI",
        "SNPS - Synopsys Inc.": "SNPS",
        "CDNS - Cadence Design Systems": "CDNS",
        "MRVL - Marvell Technology Inc.": "MRVL",
        "ORLY - O'Reilly Automotive Inc.": "ORLY",
        "FTNT - Fortinet Inc.": "FTNT",
        "ADSK - Autodesk Inc.": "ADSK",
        "CHTR - Charter Communications": "CHTR",
        "NXPI - NXP Semiconductors": "NXPI",
        "PCAR - PACCAR Inc.": "PCAR",
        "MNST - Monster Beverage Corp.": "MNST",
        "PAYX - Paychex Inc.": "PAYX",
        "FAST - Fastenal Co.": "FAST",
        "ODFL - Old Dominion Freight Line": "ODFL",
        "ROST - Ross Stores Inc.": "ROST",
        "BZ - KANZHUN LIMITED": "BZ",
        "CTSH - Cognizant Technology Solutions": "CTSH",
        "DDOG - Datadog Inc.": "DDOG",
        "TEAM - Atlassian Corporation": "TEAM",
        "IDXX - IDEXX Laboratories Inc.": "IDXX",
        "FANG - Diamondback Energy Inc.": "FANG",
        "CSGP - CoStar Group Inc.": "CSGP",
        "ANSS - ANSYS Inc.": "ANSS",
        "ON - ON Semiconductor Corp.": "ON",
        "DXCM - DexCom Inc.": "DXCM",
        "BIIB - Biogen Inc.": "BIIB",
        "GFS - GLOBALFOUNDRIES Inc.": "GFS",
        "ILMN - Illumina Inc.": "ILMN",
        "WBD - Warner Bros. Discovery": "WBD",
        "GEHC - GE HealthCare Technologies": "GEHC",
        "EXC - Exelon Corporation": "EXC",
        "KDP - Keurig Dr Pepper Inc.": "KDP",
        "LULU - Lululemon Athletica Inc.": "LULU",
        "VRSK - Verisk Analytics Inc.": "VRSK",
        "CCEP - Coca-Cola Europacific Partners": "CCEP",
        "CRWD - CrowdStrike Holdings Inc.": "CRWD",
        "SMCI - Super Micro Computer Inc.": "SMCI",
        "ARM - Arm Holdings plc": "ARM"
    }
    
    # If search query provided, filter results
    if search_query:
        search_query = search_query.lower().strip()
        filtered_tickers = {}
        
        for display, symbol in comprehensive_tickers.items():
            # Search in both display name and symbol
            if (search_query in display.lower() or 
                search_query in symbol.lower()):
                filtered_tickers[display] = symbol
        
        # Sort by relevance - exact symbol matches first
        sorted_items = sorted(filtered_tickers.items(), 
                            key=lambda x: (
                                x[1].lower() == search_query,  # Exact symbol match
                                x[1].lower().startswith(search_query),  # Symbol starts with
                                search_query in x[0].lower()  # Name contains
                            ), reverse=True)
        
        return dict(sorted_items) if sorted_items else comprehensive_tickers
    
    return comprehensive_tickers

