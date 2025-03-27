import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any

class GoldDataFetcher:
    """Class to handle gold price data fetching from various sources."""
    
    def __init__(self):
        self.gold_tickers = ["GLD", "IAU", "GC=F"]
        self.cache_dir = os.path.join(os.path.dirname(__file__), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def fetch_live_data(self, days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch live gold price data from Yahoo Finance.
        
        Args:
            days: Number of days of historical data to fetch
            
        Returns:
            DataFrame with gold price data or None if fetch fails
        """
        for ticker in self.gold_tickers:
            try:
                data = yf.download(
                    ticker,
                    start=(datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d'),
                    end=datetime.now().strftime('%Y-%m-%d'),
                    interval="1d"
                )
                if not data.empty:
                    return data
            except Exception as e:
                print(f"Failed to fetch data for {ticker}: {str(e)}")
                continue
        return None
    
    def generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        """
        Generate sample gold price data for testing.
        
        Args:
            days: Number of days of sample data to generate
            
        Returns:
            DataFrame with sample gold price data
        """
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        return pd.DataFrame({
            'Open': np.random.normal(1800, 50, len(dates)),
            'High': np.random.normal(1820, 50, len(dates)),
            'Low': np.random.normal(1780, 50, len(dates)),
            'Close': np.random.normal(1800, 50, len(dates)),
            'Volume': np.random.normal(1000000, 100000, len(dates))
        }, index=dates)
    
    def get_data(self, days: int = 365, use_cache: bool = True) -> pd.DataFrame:
        """
        Get gold price data, trying live data first and falling back to sample data.
        
        Args:
            days: Number of days of data to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with gold price data
        """
        # Try to fetch live data
        data = self.fetch_live_data(days)
        if data is not None:
            return data
            
        # If live data fails, generate sample data
        return self.generate_sample_data(days)
    
    def get_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Calculate technical indicators from price data.
        
        Args:
            data: DataFrame with price data
            
        Returns:
            Dictionary of technical indicators
        """
        indicators = {}
        
        # RSI
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        indicators['MA20'] = data['Close'].rolling(window=20).mean()
        indicators['MA50'] = data['Close'].rolling(window=50).mean()
        
        # Volatility
        indicators['Volatility'] = data['Close'].pct_change().rolling(window=14).std()
        
        return indicators 