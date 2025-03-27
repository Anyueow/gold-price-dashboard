import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Any, Optional

class GoldVisualizer:
    """Class to handle gold price visualization."""
    
    @staticmethod
    def create_price_prediction_plot(
        historical_data: pd.DataFrame,
        predictions: Optional[pd.DataFrame] = None
    ) -> go.Figure:
        """
        Create a plot showing historical prices and predictions.
        
        Args:
            historical_data: DataFrame with historical price data
            predictions: Optional DataFrame with price predictions
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Historical prices
        fig.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Historical Price',
            line=dict(color='gold')
        ))
        
        # Predictions if available
        if predictions is not None:
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                name='Price Prediction',
                line=dict(color='red', dash='dash')
            ))
        
        fig.update_layout(
            title='Gold Price History and 14-Day Prediction',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            hovermode='x unified'
        )
        
        return fig
    
    @staticmethod
    def create_technical_indicators_plot(
        historical_data: pd.DataFrame,
        indicators: Dict[str, pd.Series]
    ) -> Dict[str, go.Figure]:
        """
        Create plots for technical indicators.
        
        Args:
            historical_data: DataFrame with historical price data
            indicators: Dictionary of technical indicators
            
        Returns:
            Dictionary of Plotly figure objects
        """
        plots = {}
        
        # RSI plot
        rsi = go.Figure()
        rsi.add_trace(go.Scatter(
            x=historical_data.index,
            y=indicators['RSI'],
            name='RSI',
            line=dict(color='purple')
        ))
        rsi.update_layout(title='Relative Strength Index (RSI)')
        plots['RSI'] = rsi
        
        # Moving Averages plot
        ma = go.Figure()
        ma.add_trace(go.Scatter(
            x=historical_data.index,
            y=historical_data['Close'],
            name='Price',
            line=dict(color='gold')
        ))
        ma.add_trace(go.Scatter(
            x=historical_data.index,
            y=indicators['MA20'],
            name='20-day MA',
            line=dict(color='blue')
        ))
        ma.add_trace(go.Scatter(
            x=historical_data.index,
            y=indicators['MA50'],
            name='50-day MA',
            line=dict(color='red')
        ))
        ma.update_layout(title='Moving Averages')
        plots['MA'] = ma
        
        return plots
    
    @staticmethod
    def create_sentiment_price_plot(
        historical_data: pd.DataFrame,
        sentiment_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Create a plot showing price and sentiment data.
        
        Args:
            historical_data: DataFrame with historical price data
            sentiment_data: Dictionary with sentiment data
            
        Returns:
            Plotly figure object
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Gold Price', 'Market Sentiment')
        )
        
        # Gold price
        fig.add_trace(
            go.Scatter(
                x=historical_data.index,
                y=historical_data['Close'],
                name='Gold Price',
                line=dict(color='gold')
            ),
            row=1, col=1
        )
        
        # Sentiment if available
        if "error" not in sentiment_data:
            fig.add_trace(
                go.Scatter(
                    x=[historical_data.index[-1]],
                    y=[float(sentiment_data["current_sentiment"])],
                    name='Current Sentiment',
                    mode='markers',
                    marker=dict(size=10, color='blue')
                ),
                row=2, col=1
            )
        
        fig.update_layout(height=800, showlegend=True)
        return fig 