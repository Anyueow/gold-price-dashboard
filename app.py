import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from pathlib import Path
import json
import pickle
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from gold_analyzer.data.fetcher import GoldDataFetcher
from gold_analyzer.models.lstm import GoldPricePredictor
from gold_analyzer.utils.preprocessing import prepare_data
from gold_analyzer.utils.visualization import plot_predictions, plot_technical_indicators

# Set page config
st.set_page_config(
    page_title="Gold Price Analyzer",
    page_icon="📈",
    layout="wide"
)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'technical_indicators' not in st.session_state:
    st.session_state.technical_indicators = None

# Title and description
st.title("📈 Gold Price Analyzer")
st.markdown("""
This application analyzes gold prices using machine learning and technical indicators.
""")

# Sidebar controls
st.sidebar.header("Controls")

# Data fetching
st.sidebar.subheader("Data Settings")
days = st.sidebar.slider("Number of days to analyze", 30, 365, 180)
use_sample_data = st.sidebar.checkbox("Use sample data", value=False)

# Model settings
st.sidebar.subheader("Model Settings")
sequence_length = st.sidebar.slider("Sequence length", 5, 30, 10)
hidden_size = st.sidebar.slider("Hidden size", 32, 256, 64)
num_layers = st.sidebar.slider("Number of layers", 1, 4, 2)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, step=0.1)

# Training settings
st.sidebar.subheader("Training Settings")
epochs = st.sidebar.slider("Number of epochs", 10, 100, 50)
batch_size = st.sidebar.slider("Batch size", 16, 128, 32)
learning_rate = st.sidebar.slider("Learning rate", 0.0001, 0.01, 0.001, step=0.0001)

# Fetch data
@st.cache_data
def fetch_data(days: int, use_sample_data: bool) -> pd.DataFrame:
    fetcher = GoldDataFetcher()
    if use_sample_data:
        return fetcher.generate_sample_data(days)
    return fetcher.get_data(days)

# Main content
st.header("Data Overview")

# Fetch and display data
data = fetch_data(days, use_sample_data)
st.session_state.data = data

# Display current price metrics
if data is not None:
    current_price = float(data['Close'].iloc[-1])
    price_change = float(data['Close'].pct_change().iloc[-1])
    volatility = float(data['Close'].pct_change().std() * np.sqrt(14))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Current Gold Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change:.2%}"
        )
    with col2:
        st.metric(
            label="14-Day Volatility",
            value=f"{volatility:.2%}",
            delta=None
        )
    with col3:
        st.metric(
            label="30-Day Change",
            value=f"{((current_price - float(data['Close'].iloc[-30])) / float(data['Close'].iloc[-30])):.2%}",
            delta=None
        )

# Display raw data
st.subheader("Raw Data")
st.dataframe(data.head())

# Technical indicators
st.header("Technical Indicators")
fetcher = GoldDataFetcher()
technical_indicators = fetcher.get_technical_indicators(data)
st.session_state.technical_indicators = technical_indicators

# Plot technical indicators
fig_indicators = plot_technical_indicators(data, technical_indicators)
st.plotly_chart(fig_indicators, use_container_width=True)

# Model training and prediction
st.header("Price Prediction")

# Initialize model if not exists
if st.session_state.model is None:
    st.session_state.model = GoldPricePredictor(
        input_size=5,  # OHLCV features
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    )

# Training section
st.subheader("Model Training")
if st.button("Train Model"):
    with st.spinner("Training model..."):
        # Prepare data
        X, y = prepare_data(data, sequence_length)
        
        # Train model
        st.session_state.model.train(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate
        )
        
        # Make predictions
        predictions = st.session_state.model.predict(X)
        st.session_state.predictions = predictions
        
        st.success("Model trained successfully!")

# Display predictions if available
if st.session_state.predictions is not None:
    st.subheader("Predictions")
    fig_predictions = plot_predictions(data, st.session_state.predictions, sequence_length)
    st.plotly_chart(fig_predictions, use_container_width=True)
    
    # Display prediction metrics
    actual_values = data['Close'].values[sequence_length:]
    mse = float(np.mean((st.session_state.predictions - actual_values)**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(st.session_state.predictions - actual_values)))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("MSE", f"{mse:.2f}")
    with col2:
        st.metric("RMSE", f"{rmse:.2f}")
    with col3:
        st.metric("MAE", f"{mae:.2f}")

# Save model button
if st.session_state.model is not None:
    if st.button("Save Model"):
        model_path = Path("models/gold_price_predictor.pkl")
        model_path.parent.mkdir(exist_ok=True)
        
        # Save model state
        model_state = {
            'model_state_dict': st.session_state.model.state_dict(),
            'input_size': 5,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_state, f)
        
        st.success(f"Model saved to {model_path}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit and PyTorch</p>
    <p>Data source: Alpha Vantage</p>
</div>
""", unsafe_allow_html=True) 