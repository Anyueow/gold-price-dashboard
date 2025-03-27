import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any

def plot_predictions(data: pd.DataFrame, predictions: np.ndarray, sequence_length: int) -> go.Figure:
    """
    Create a plot showing actual prices and predictions.
    
    Args:
        data: DataFrame with price data
        predictions: Array of predicted prices
        sequence_length: Length of input sequences
        
    Returns:
        Plotly figure object
    """
    # Create figure
    fig = go.Figure()
    
    # Add actual prices
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Actual',
        line=dict(color='blue')
    ))
    
    # Add predictions
    pred_dates = data.index[sequence_length:]
    fig.add_trace(go.Scatter(
        x=pred_dates,
        y=predictions,
        name='Predicted',
        line=dict(color='red', dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Gold Price Predictions',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode='x unified'
    )
    
    return fig

def plot_technical_indicators(data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> go.Figure:
    """
    Create a plot showing price data and technical indicators.
    
    Args:
        data: DataFrame with price data
        indicators: Dictionary of technical indicators
        
    Returns:
        Plotly figure object
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1, 
                       shared_xaxes=True,
                       vertical_spacing=0.03,
                       subplot_titles=('Price and Moving Averages', 'Technical Indicators'),
                       row_heights=[0.7, 0.3])

    # Add price data
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        name='Price',
        line=dict(color='blue')
    ), row=1, col=1)

    # Add moving averages
    fig.add_trace(go.Scatter(
        x=data.index,
        y=indicators['MA20'],
        name='MA20',
        line=dict(color='orange')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=data.index,
        y=indicators['MA50'],
        name='MA50',
        line=dict(color='green')
    ), row=1, col=1)

    # Add RSI
    fig.add_trace(go.Scatter(
        x=data.index,
        y=indicators['RSI'],
        name='RSI',
        line=dict(color='purple')
    ), row=2, col=1)

    # Add RSI overbought/oversold lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # Update layout
    fig.update_layout(
        title='Technical Analysis',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        yaxis2_title='RSI',
        hovermode='x unified',
        height=800
    )

    # Update y-axes ranges
    fig.update_yaxes(range=[0, 100], row=2, col=1)  # RSI range

    return fig 