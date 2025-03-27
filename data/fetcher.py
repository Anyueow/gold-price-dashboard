import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import Optional, Dict, Any
import time
import streamlit as st

class GoldDataFetcher:
    """Class to handle gold price data fetching from Yahoo Finance."""
    
    def __init__(self):
        self.symbol = "GLD"  # SPDR Gold Trust ETF
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
        try:
            # Calculate start date
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(self.symbol)
            data = ticker.history(start=start_date, end=end_date)
            
            # Rename columns to match expected format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']
            
            # Keep only the columns we need
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            return data
            
        except Exception as e:
            print(f"Failed to fetch data from Yahoo Finance: {str(e)}")
            return None
    
    def generate_sample_data(self, days: int = 365) -> pd.DataFrame:
        """
        Generate realistic sample gold price data for testing.
        
        Args:
            days: Number of days of sample data to generate
            
        Returns:
            DataFrame with sample gold price data
        """
        # Generate dates
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            end=datetime.now(),
            freq='D'
        )
        
        # Generate base price with trend and seasonality
        base_price = 1800
        trend = np.linspace(0, 100, len(dates))
        seasonality = 50 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
        
        # Generate prices with realistic volatility
        prices = base_price + trend + seasonality + np.random.normal(0, 20, len(dates))
        
        # Ensure prices are positive and maintain realistic relationships
        prices = np.maximum(prices, 1000)
        
        # Generate other price components
        open_prices = prices + np.random.normal(0, 5, len(dates))
        high_prices = np.maximum(open_prices, prices) + np.random.uniform(0, 10, len(dates))
        low_prices = np.minimum(open_prices, prices) - np.random.uniform(0, 10, len(dates))
        
        # Generate volume with some correlation to price changes
        price_changes = np.abs(prices[1:] - prices[:-1])
        volume = np.random.normal(1000000, 100000, len(dates))
        volume[1:] += price_changes * 1000  # Higher volume on price changes
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': prices,
            'Volume': volume
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
        print("Using sample data due to API limitations")
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