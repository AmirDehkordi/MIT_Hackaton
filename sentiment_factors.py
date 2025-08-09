"""
Sentiment Factor Extraction Module
Processes market sentiment data, news sentiment, and social media sentiment
"""

import pandas as pd
import numpy as np
import requests
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
import config

class SentimentFactorExtractor:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def process_market_sentiment_indicators(self, sentiment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Process traditional market sentiment indicators
        """
        sentiment_df = pd.DataFrame()
        
        try:
            # VIX (Fear Index) - from data collector
            if 'vix' in sentiment_data and not sentiment_data['vix'].empty:
                vix_data = sentiment_data['vix']
                sentiment_df['VIX'] = vix_data.iloc[:, 0]  # First column usually contains the values
                sentiment_df['VIX_MA_10'] = sentiment_df['VIX'].rolling(window=10).mean()
                sentiment_df['VIX_Spike'] = (sentiment_df['VIX'] > sentiment_df['VIX_MA_10'] * 1.2).astype(int)
                sentiment_df['VIX_Normalized'] = (sentiment_df['VIX'] - sentiment_df['VIX'].rolling(window=252).mean()) / sentiment_df['VIX'].rolling(window=252).std()
            
            # Put/Call Ratio
            if 'put_call_ratio' in sentiment_data and not sentiment_data['put_call_ratio'].empty:
                pc_data = sentiment_data['put_call_ratio']
                sentiment_df['Put_Call_Ratio'] = pc_data.iloc[:, 0]
                sentiment_df['PC_MA_10'] = sentiment_df['Put_Call_Ratio'].rolling(window=10).mean()
                sentiment_df['PC_Extreme'] = np.where(sentiment_df['Put_Call_Ratio'] > 1.2, 1,  # High fear
                                                    np.where(sentiment_df['Put_Call_Ratio'] < 0.8, -1, 0))  # Low fear/high greed
            
            # AAII Sentiment Survey
            if 'aaii_bull' in sentiment_data and not sentiment_data['aaii_bull'].empty:
                aaii_data = sentiment_data['aaii_bull']
                if 'Bull' in aaii_data.columns:
                    sentiment_df['AAII_Bull'] = aaii_data['Bull']
                    sentiment_df['AAII_Bull_MA'] = sentiment_df['AAII_Bull'].rolling(window=4).mean()  # 4-week average
                    sentiment_df['AAII_Extreme_Bull'] = (sentiment_df['AAII_Bull'] > 50).astype(int)  # Contrarian indicator
                
                if 'Bear' in aaii_data.columns:
                    sentiment_df['AAII_Bear'] = aaii_data['Bear']
                    sentiment_df['AAII_Bear_MA'] = sentiment_df['AAII_Bear'].rolling(window=4).mean()
                    sentiment_df['AAII_Extreme_Bear'] = (sentiment_df['AAII_Bear'] > 50).astype(int)
                
                # Bull-Bear Spread
                if 'AAII_Bull' in sentiment_df.columns and 'AAII_Bear' in sentiment_df.columns:
                    sentiment_df['AAII_Bull_Bear_Spread'] = sentiment_df['AAII_Bull'] - sentiment_df['AAII_Bear']
                    sentiment_df['AAII_Normalized_Spread'] = (sentiment_df['AAII_Bull_Bear_Spread'] - 
                                                             sentiment_df['AAII_Bull_Bear_Spread'].rolling(window=52).mean()) / \
                                                            sentiment_df['AAII_Bull_Bear_Spread'].rolling(window=52).std()
            
        except Exception as e:
            print(f"Error processing market sentiment indicators: {e}")
        
        return sentiment_df
    
    def get_news_sentiment_for_ticker(self, ticker: str, days_back: int = 30) -> Dict:
        """
        Get news sentiment for a specific ticker using various sources
        """
        sentiment_scores = {
            'compound_scores': [],
            'positive_scores': [],
            'negative_scores': [],
            'neutral_scores': [],
            'dates': [],
            'article_count': 0
        }
        
        try:
            # Get news from yfinance (limited but free)
            stock = yf.Ticker(ticker)
            news = stock.news
            
            for article in news[:min(20, len(news))]:  # Limit to recent 20 articles
                try:
                    # Extract title and summary
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    text = f"{title} {summary}"
                    
                    if text.strip():
                        # VADER sentiment analysis
                        scores = self.vader_analyzer.polarity_scores(text)
                        
                        sentiment_scores['compound_scores'].append(scores['compound'])
                        sentiment_scores['positive_scores'].append(scores['pos'])
                        sentiment_scores['negative_scores'].append(scores['neg'])
                        sentiment_scores['neutral_scores'].append(scores['neu'])
                        
                        # Try to extract date
                        pub_date = article.get('providerPublishTime')
                        if pub_date:
                            sentiment_scores['dates'].append(datetime.fromtimestamp(pub_date))
                        else:
                            sentiment_scores['dates'].append(datetime.now())
                            
                except Exception as e:
                    print(f"Error processing article: {e}")
                    continue
            
            sentiment_scores['article_count'] = len(sentiment_scores['compound_scores'])
            
        except Exception as e:
            print(f"Error getting news sentiment for {ticker}: {e}")
        
        return sentiment_scores
    
    def calculate_sector_sentiment(self, sector_tickers: List[str]) -> pd.DataFrame:
        """
        Calculate sentiment for sector ETFs to gauge sector-specific sentiment
        """
        sector_sentiment = pd.DataFrame()
        
        sector_mapping = {
            'XLK': 'Technology',
            'XLF': 'Financial',
            'XLV': 'Healthcare',
            'XLE': 'Energy',
            'XLI': 'Industrial',
            'XLY': 'Consumer_Discretionary',
            'XLP': 'Consumer_Staples',
            'XLU': 'Utilities',
            'XLB': 'Materials',
            'XLRE': 'Real_Estate',
            'XLC': 'Communication'
        }
        
        for ticker in sector_tickers:
            try:
                sentiment_data = self.get_news_sentiment_for_ticker(ticker, days_back=7)
                
                if sentiment_data['article_count'] > 0:
                    sector_name = sector_mapping.get(ticker, ticker)
                    avg_sentiment = np.mean(sentiment_data['compound_scores'])
                    sentiment_strength = np.std(sentiment_data['compound_scores'])
                    
                    # Create a simple time series (this would be more sophisticated in practice)
                    current_date = datetime.now().date()
                    
                    if sector_sentiment.empty:
                        sector_sentiment = pd.DataFrame(index=[current_date])
                    
                    sector_sentiment[f'{sector_name}_Sentiment'] = avg_sentiment
                    sector_sentiment[f'{sector_name}_Sentiment_Strength'] = sentiment_strength
                    
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                print(f"Error calculating sentiment for {ticker}: {e}")
                continue
        
        return sector_sentiment
    
    def create_news_sentiment_factor(self, ticker: str, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create a news sentiment factor aligned with price data timeline
        """
        sentiment_df = pd.DataFrame(index=price_data.index)
        
        try:
            # Get recent news sentiment
            news_sentiment = self.get_news_sentiment_for_ticker(ticker, days_back=90)
            
            if news_sentiment['article_count'] > 0:
                # Create sentiment time series
                sentiment_dates = pd.to_datetime(news_sentiment['dates'])
                sentiment_scores = pd.DataFrame({
                    'Date': sentiment_dates,
                    'Compound': news_sentiment['compound_scores'],
                    'Positive': news_sentiment['positive_scores'],
                    'Negative': news_sentiment['negative_scores'],
                    'Neutral': news_sentiment['neutral_scores']
                })
                
                # Group by date and take average if multiple articles per day
                daily_sentiment = sentiment_scores.groupby(sentiment_scores['Date'].dt.date).mean()
                daily_sentiment.index = pd.to_datetime(daily_sentiment.index)
                
                # Align with price data and forward fill
                for col in daily_sentiment.columns:
                    sentiment_df[f'News_{col}'] = daily_sentiment[col].reindex(sentiment_df.index, method='ffill')
                
                # Calculate rolling averages and trends
                sentiment_df['News_Sentiment_MA_5'] = sentiment_df['News_Compound'].rolling(window=5).mean()
                sentiment_df['News_Sentiment_MA_20'] = sentiment_df['News_Compound'].rolling(window=20).mean()
                sentiment_df['News_Sentiment_Trend'] = sentiment_df['News_Compound'] - sentiment_df['News_Sentiment_MA_20']
                
                # Sentiment momentum
                sentiment_df['News_Sentiment_Momentum'] = sentiment_df['News_Compound'].diff()
                
                # Extreme sentiment flags
                sentiment_df['News_Extreme_Positive'] = (sentiment_df['News_Compound'] > 0.5).astype(int)
                sentiment_df['News_Extreme_Negative'] = (sentiment_df['News_Compound'] < -0.5).astype(int)
            
        except Exception as e:
            print(f"Error creating news sentiment factor for {ticker}: {e}")
        
        return sentiment_df.fillna(method='ffill').fillna(0)
    
    def calculate_social_sentiment_proxy(self, ticker: str) -> Dict:
        """
        Calculate social media sentiment proxy using available data
        Note: This is a simplified version. In practice, you'd use APIs like Twitter, Reddit, etc.
        """
        # This is a placeholder - in practice you'd integrate with social media APIs
        # For now, we'll use some proxy indicators
        
        social_metrics = {
            'social_volume_proxy': 0,
            'social_sentiment_proxy': 0,
            'reddit_mentions_proxy': 0,
            'twitter_sentiment_proxy': 0
        }
        
        try:
            # You could integrate with:
            # - Reddit API for r/wallstreetbets mentions
            # - Twitter API for tweet sentiment
            # - Google Trends for search volume
            # - StockTwits API for financial social sentiment
            
            # For now, we'll create some synthetic proxy data
            # based on stock volatility and volume as sentiment proxies
            stock = yf.Ticker(ticker)
            recent_data = stock.history(period="30d")
            
            if not recent_data.empty:
                # Use volume and volatility as proxies for social interest
                avg_volume = recent_data['Volume'].mean()
                recent_volatility = recent_data['Close'].pct_change().std()
                
                # Normalize to sentiment-like scores
                social_metrics['social_volume_proxy'] = min(avg_volume / 1000000, 10) / 10  # Normalize to 0-1
                social_metrics['social_sentiment_proxy'] = max(-1, min(1, (recent_volatility - 0.02) * 10))  # Convert volatility to sentiment
                
        except Exception as e:
            print(f"Error calculating social sentiment proxy for {ticker}: {e}")
        
        return social_metrics
    
    def create_comprehensive_sentiment_factor(self, ticker: str, price_data: pd.DataFrame, 
                                            market_sentiment_data: Dict) -> pd.DataFrame:
        """
        Create a comprehensive sentiment factor combining all sentiment sources
        """
        # Start with market sentiment indicators
        market_sentiment_df = self.process_market_sentiment_indicators(market_sentiment_data)
        
        # Add news sentiment
        news_sentiment_df = self.create_news_sentiment_factor(ticker, price_data)
        
        # Combine all sentiment data
        if not market_sentiment_df.empty and not news_sentiment_df.empty:
            # Align indices
            combined_index = price_data.index
            
            # Reindex market sentiment to match price data
            market_aligned = market_sentiment_df.reindex(combined_index, method='ffill')
            news_aligned = news_sentiment_df.reindex(combined_index, method='ffill')
            
            # Combine
            sentiment_factor = pd.concat([market_aligned, news_aligned], axis=1)
        elif not news_sentiment_df.empty:
            sentiment_factor = news_sentiment_df
        elif not market_sentiment_df.empty:
            sentiment_factor = market_sentiment_df.reindex(price_data.index, method='ffill')
        else:
            sentiment_factor = pd.DataFrame(index=price_data.index)
        
        # Add social sentiment proxy
        social_metrics = self.calculate_social_sentiment_proxy(ticker)
        for key, value in social_metrics.items():
            sentiment_factor[key] = value
        
        return sentiment_factor.fillna(method='ffill').fillna(0)
    
    def create_aggregated_sentiment_factors(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Create aggregated sentiment factors using PCA
        """
        if sentiment_data.empty:
            return pd.DataFrame()
        
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        try:
            # Clean the data
            sentiment_clean = sentiment_data.replace([np.inf, -np.inf], np.nan)
            sentiment_clean = sentiment_clean.fillna(method='ffill').fillna(0)
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(sentiment_clean)
            
            # Apply PCA
            n_components = min(3, scaled_data.shape[1])  # Limit to 3 sentiment factors
            pca = PCA(n_components=n_components)
            factors = pca.fit_transform(scaled_data)
            
            # Create factor DataFrame
            factor_columns = [f'Sentiment_Factor_{i+1}' for i in range(n_components)]
            factor_df = pd.DataFrame(factors, 
                                    index=sentiment_data.index, 
                                    columns=factor_columns)
            
            # Store variance explained
            self.sentiment_factor_variance = pca.explained_variance_ratio_
            self.sentiment_factor_components = pca.components_
            
            return factor_df
            
        except Exception as e:
            print(f"Error creating aggregated sentiment factors: {e}")
            return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    print("Sentiment Factor Extractor module ready!")
    print("Use with market data and news to generate sentiment factors.") 