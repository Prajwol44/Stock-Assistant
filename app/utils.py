# utils.py
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st
from typing import Dict, List, Tuple, Optional
import plotly.graph_objs as go
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NewsDataFetcher:
    """Fetch and analyze news sentiment for stocks"""
    
    def __init__(self, api_key: str = None):
        """
        Initialize with NewsAPI key
        Get free API key from: https://newsapi.org/register
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2/everything"
        self.analyzer = SentimentIntensityAnalyzer()
        
    def get_company_name(self, ticker: str) -> str:
        """Get company name from ticker symbol"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            return info.get('shortName', ticker)
        except:
            return ticker
    
    def fetch_stock_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Fetch news for a stock ticker
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            days_back: Days to look back for news
            
        Returns:
            List of news articles with sentiment scores
        """
        if not self.api_key:
            # Return mock data if no API key
            return self._get_mock_news_data(ticker)
        
        company_name = self.get_company_name(ticker)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Search queries
        queries = [
            f'"{company_name}"',
            f'{ticker} stock',
            f'{company_name} earnings',
            f'{ticker} financial'
        ]
        
        all_articles = []
        
        for query in queries:
            try:
                articles = self._fetch_articles(
                    query=query,
                    from_date=start_date.strftime('%Y-%m-%d'),
                    to_date=end_date.strftime('%Y-%m-%d')
                )
                all_articles.extend(articles)
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching news for {query}: {e}")
                continue
        
        # Remove duplicates and analyze sentiment
        unique_articles = self._remove_duplicates(all_articles)
        analyzed_articles = self._analyze_sentiment(unique_articles)
        
        return analyzed_articles[:10]  # Return top 10 articles
    
    def _fetch_articles(self, query: str, from_date: str, to_date: str) -> List[Dict]:
        """Fetch articles from NewsAPI"""
        params = {
            'q': query,
            'from': from_date,
            'to': to_date,
            'sortBy': 'publishedAt',
            'pageSize': 50,
            'language': 'en',
            'apiKey': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('articles', [])
        else:
            raise Exception(f"NewsAPI request failed: {response.status_code}")
    
    def _remove_duplicates(self, articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on URL"""
        unique_articles = []
        seen_urls = set()
        
        for article in articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                unique_articles.append(article)
                seen_urls.add(url)
                
        return unique_articles
    
    def _analyze_sentiment(self, articles: List[Dict]) -> List[Dict]:
        """Analyze sentiment of articles"""
        analyzed_articles = []
        
        for article in articles:
            # Combine title and description for analysis
            text = f"{article.get('title', '')} {article.get('description', '')}"
            
            if not text.strip():
                continue
            
            # TextBlob sentiment
            blob = TextBlob(text)
            textblob_score = blob.sentiment.polarity
            
            # VADER sentiment
            vader_scores = self.analyzer.polarity_scores(text)
            vader_score = vader_scores['compound']
            
            # Combined sentiment score
            combined_sentiment = (textblob_score + vader_score) / 2
            
            # Classify sentiment
            if combined_sentiment > 0.1:
                sentiment_label = "Positive"
            elif combined_sentiment < -0.1:
                sentiment_label = "Negative"
            else:
                sentiment_label = "Neutral"
            
            analyzed_article = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', ''),
                'sentiment_score': round(combined_sentiment, 3),
                'sentiment_label': sentiment_label,
                'textblob_score': round(textblob_score, 3),
                'vader_score': round(vader_score, 3)
            }
            
            analyzed_articles.append(analyzed_article)
        
        return analyzed_articles
    
    def _get_mock_news_data(self, ticker: str) -> List[Dict]:
        """Return mock news data when API key is not available"""
        mock_articles = [
            {
                'title': f'{ticker} Shows Strong Performance in Latest Quarter',
                'description': f'{ticker} reported better than expected earnings with strong revenue growth.',
                'url': f'https://example.com/news/{ticker}-earnings',
                'published_at': (datetime.now() - timedelta(days=1)).isoformat(),
                'source': 'Financial Times',
                'sentiment_score': 0.6,
                'sentiment_label': 'Positive',
                'textblob_score': 0.65,
                'vader_score': 0.55
            },
            {
                'title': f'Market Volatility Affects {ticker} Trading',
                'description': f'{ticker} experiences fluctuations amid broader market uncertainty.',
                'url': f'https://example.com/news/{ticker}-volatility',
                'published_at': (datetime.now() - timedelta(days=2)).isoformat(),
                'source': 'Reuters',
                'sentiment_score': -0.2,
                'sentiment_label': 'Negative',
                'textblob_score': -0.1,
                'vader_score': -0.3
            },
            {
                'title': f'{ticker} Announces New Product Line',
                'description': f'{ticker} unveils innovative products expected to drive future growth.',
                'url': f'https://example.com/news/{ticker}-products',
                'published_at': (datetime.now() - timedelta(days=3)).isoformat(),
                'source': 'TechCrunch',
                'sentiment_score': 0.4,
                'sentiment_label': 'Positive',
                'textblob_score': 0.45,
                'vader_score': 0.35
            }
        ]
        return mock_articles
    
    def calculate_overall_sentiment(self, articles: List[Dict]) -> Dict:
        """Calculate overall sentiment metrics"""
        if not articles:
            return {
                'overall_score': 0,
                'positive_percentage': 0,
                'negative_percentage': 0,
                'neutral_percentage': 0,
                'total_articles': 0
            }
        
        scores = [article['sentiment_score'] for article in articles]
        labels = [article['sentiment_label'] for article in articles]
        
        # Calculate percentages
        positive_count = labels.count('Positive')
        negative_count = labels.count('Negative')
        neutral_count = labels.count('Neutral')
        total = len(labels)
        
        return {
            'overall_score': round(np.mean(scores), 3),
            'positive_percentage': round((positive_count / total) * 100, 1),
            'negative_percentage': round((negative_count / total) * 100, 1),
            'neutral_percentage': round((neutral_count / total) * 100, 1),
            'total_articles': total
        }

class TechnicalAnalyzer:
    """Technical analysis utilities"""
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate multiple moving averages"""
        ma_df = pd.DataFrame(index=prices.index)
        for window in windows:
            ma_df[f'MA_{window}'] = prices.rolling(window=window).mean()
        return ma_df
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        bb_df = pd.DataFrame(index=prices.index)
        bb_df['Middle'] = sma
        bb_df['Upper'] = sma + (std * std_dev)
        bb_df['Lower'] = sma - (std * std_dev)
        
        return bb_df
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd_df = pd.DataFrame(index=prices.index)
        macd_df['MACD'] = ema_fast - ema_slow
        macd_df['Signal'] = macd_df['MACD'].ewm(span=signal).mean()
        macd_df['Histogram'] = macd_df['MACD'] - macd_df['Signal']
        
        return macd_df

class PricePredictionModel:
    """Simple price prediction utilities"""
    
    @staticmethod
    def simple_linear_trend(prices: pd.Series, days_ahead: int = 7) -> Dict:
        """Simple linear trend prediction"""
        # Use last 30 days for trend calculation
        recent_prices = prices.tail(30)
        x = np.arange(len(recent_prices))
        coeffs = np.polyfit(x, recent_prices.values, 1)
        
        # Predict future prices
        future_x = np.arange(len(recent_prices), len(recent_prices) + days_ahead)
        predicted_prices = np.polyval(coeffs, future_x)
        
        current_price = prices.iloc[-1]
        predicted_price = predicted_prices[-1]
        change_percent = ((predicted_price - current_price) / current_price) * 100
        
        return {
            'predicted_price': round(predicted_price, 2),
            'current_price': round(current_price, 2),
            'change_percent': round(change_percent, 2),
            'trend': 'Bullish' if change_percent > 0 else 'Bearish',
            'confidence': min(85, max(15, 50 + abs(change_percent) * 2))  # Mock confidence
        }
    
    @staticmethod
    def calculate_support_resistance(prices: pd.Series, window: int = 20) -> Dict:
        """Calculate basic support and resistance levels"""
        recent_prices = prices.tail(window * 3)
        
        # Find local minima and maxima
        highs = recent_prices.rolling(window=window, center=True).max()
        lows = recent_prices.rolling(window=window, center=True).min()
        
        resistance = highs[highs == recent_prices].mean()
        support = lows[lows == recent_prices].mean()
        
        return {
            'resistance': round(resistance, 2) if not np.isnan(resistance) else None,
            'support': round(support, 2) if not np.isnan(support) else None,
            'current_price': round(prices.iloc[-1], 2)
        }

def get_stock_data_with_analysis(ticker: str, period: str = "6mo", news_api_key: str = None) -> Dict:
    """
    Get comprehensive stock data including technical analysis and sentiment
    
    Args:
        ticker: Stock symbol
        period: Time period for historical data
        news_api_key: NewsAPI key for sentiment analysis
        
    Returns:
        Dictionary containing all analysis data
    """
    try:
        # Get stock data
        stock = yf.Ticker(ticker)
        hist = stock.history(period=period)
        info = stock.info
        
        if hist.empty:
            return {'error': 'No data found for ticker'}
        
        # Technical analysis
        tech_analyzer = TechnicalAnalyzer()
        
        # Calculate indicators
        hist['RSI'] = tech_analyzer.calculate_rsi(hist['Close'])
        ma_data = tech_analyzer.calculate_moving_averages(hist['Close'])
        bb_data = tech_analyzer.calculate_bollinger_bands(hist['Close'])
        macd_data = tech_analyzer.calculate_macd(hist['Close'])
        
        # Price prediction
        prediction_model = PricePredictionModel()
        price_prediction = prediction_model.simple_linear_trend(hist['Close'])
        support_resistance = prediction_model.calculate_support_resistance(hist['Close'])
        
        # News sentiment analysis
        news_fetcher = NewsDataFetcher(news_api_key)
        news_articles = news_fetcher.fetch_stock_news(ticker)
        sentiment_summary = news_fetcher.calculate_overall_sentiment(news_articles)
        
        return {
            'ticker': ticker,
            'company_info': info,
            'historical_data': hist,
            'technical_indicators': {
                'moving_averages': ma_data,
                'bollinger_bands': bb_data,
                'macd': macd_data
            },
            'price_prediction': price_prediction,
            'support_resistance': support_resistance,
            'news_articles': news_articles,
            'sentiment_summary': sentiment_summary,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {e}")
        return {'error': str(e)}

def format_currency(value: float, ticker: str) -> str:
    """Format currency based on ticker"""
    is_indian = ticker.endswith('.NS') or ticker.endswith('.BO')
    symbol = "₹" if is_indian else "$"
    return f"{symbol}{value:,.2f}"

def get_risk_assessment(rsi: float, volatility: float, sentiment_score: float) -> Dict:
    """Calculate risk assessment based on multiple factors"""
    risk_factors = []
    overall_risk = 0
    
    # RSI risk
    if rsi > 70:
        risk_factors.append("Overbought conditions")
        overall_risk += 30
    elif rsi < 30:
        risk_factors.append("Oversold conditions")
        overall_risk += 20
    
    # Volatility risk
    if volatility > 0.3:
        risk_factors.append("High volatility")
        overall_risk += 25
    elif volatility > 0.15:
        risk_factors.append("Moderate volatility")
        overall_risk += 15
    
    # Sentiment risk
    if sentiment_score < -0.3:
        risk_factors.append("Negative market sentiment")
        overall_risk += 20
    elif sentiment_score > 0.3:
        overall_risk -= 10  # Positive sentiment reduces risk
    
    # Classify risk level
    if overall_risk < 20:
        risk_level = "Low"
    elif overall_risk < 40:
        risk_level = "Medium"
    else:
        risk_level = "High"
    
    return {
        'risk_level': risk_level,
        'risk_score': min(100, max(0, overall_risk)),
        'risk_factors': risk_factors
    }

def generate_trading_recommendation(price_prediction: Dict, sentiment_summary: Dict, 
                                  rsi: float, support_resistance: Dict) -> Dict:
    """Generate trading recommendation based on multiple factors"""
    score = 0
    reasons = []
    
    # Price trend factor
    if price_prediction['change_percent'] > 5:
        score += 30
        reasons.append("Strong upward price trend")
    elif price_prediction['change_percent'] > 0:
        score += 15
        reasons.append("Positive price trend")
    elif price_prediction['change_percent'] < -5:
        score -= 30
        reasons.append("Strong downward price trend")
    
    # Sentiment factor
    if sentiment_summary['overall_score'] > 0.2:
        score += 25
        reasons.append("Positive market sentiment")
    elif sentiment_summary['overall_score'] < -0.2:
        score -= 25
        reasons.append("Negative market sentiment")
    
    # RSI factor
    if 30 <= rsi <= 70:
        score += 15
        reasons.append("Healthy RSI levels")
    elif rsi < 30:
        score += 10
        reasons.append("Potentially oversold")
    elif rsi > 70:
        score -= 15
        reasons.append("Potentially overbought")
    
    # Generate recommendation
    if score >= 50:
        recommendation = "STRONG BUY"
    elif score >= 25:
        recommendation = "BUY"
    elif score >= -10:
        recommendation = "HOLD"
    elif score >= -35:
        recommendation = "SELL"
    else:
        recommendation = "STRONG SELL"
    
    return {
        'recommendation': recommendation,
        'confidence': min(95, max(50, abs(score) + 40)),
        'reasons': reasons,
        'score': score
    }