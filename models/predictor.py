import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

class GoldPricePredictor:
    """Class to handle gold price predictions."""
    
    def __init__(self):
        self.prediction_days = 14
        self.confidence_threshold = 0.75
    
    def predict_price(self, historical_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Predict gold prices for the next 14 days.
        
        Args:
            historical_data: DataFrame with historical price data
            
        Returns:
            DataFrame with predictions or None if prediction fails
        """
        try:
            # Get the last price and volatility
            last_price = historical_data['Close'].iloc[-1]
            volatility = historical_data['Close'].pct_change().std()
            
            # Generate prediction dates
            dates = pd.date_range(
                start=datetime.now(),
                periods=self.prediction_days + 1,
                freq='D'
            )
            
            # Generate predictions using random walk with drift
            predictions = []
            current_price = last_price
            
            for _ in range(self.prediction_days + 1):
                # Add some drift based on recent trend
                drift = historical_data['Close'].pct_change().mean()
                predicted_price = current_price * (1 + drift + np.random.normal(0, volatility))
                predictions.append(predicted_price)
                current_price = predicted_price
            
            return pd.DataFrame({
                'Date': dates,
                'Predicted_Price': predictions
            })
            
        except Exception as e:
            print(f"Error generating predictions: {str(e)}")
            return None
    
    def calculate_prediction_metrics(self, predictions: pd.DataFrame, current_price: float) -> Dict[str, Any]:
        """
        Calculate metrics for the predictions.
        
        Args:
            predictions: DataFrame with price predictions
            current_price: Current gold price
            
        Returns:
            Dictionary with prediction metrics
        """
        if predictions is None:
            return {
                'error': 'Failed to generate predictions'
            }
            
        final_predicted_price = predictions['Predicted_Price'].iloc[-1]
        predicted_change = (final_predicted_price - current_price) / current_price
        
        return {
            'predicted_price': final_predicted_price,
            'predicted_change': predicted_change,
            'confidence': self.confidence_threshold,
            'prediction_days': self.prediction_days
        }
    
    def get_buying_recommendation(self, current_price: float, ma_30: float) -> Dict[str, Any]:
        """
        Generate buying recommendation based on current price and moving average.
        
        Args:
            current_price: Current gold price
            ma_30: 30-day moving average
            
        Returns:
            Dictionary with buying recommendation
        """
        price_diff = (current_price - ma_30) / ma_30
        
        if current_price < ma_30:
            return {
                'recommendation': 'Buy',
                'confidence': 0.8,
                'reason': 'Price is below 30-day average',
                'price_difference': price_diff
            }
        else:
            return {
                'recommendation': 'Wait',
                'confidence': 0.6,
                'reason': 'Price is above 30-day average',
                'price_difference': price_diff
            } 