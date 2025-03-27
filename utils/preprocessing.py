import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler

def prepare_data(data: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare data for LSTM model by creating sequences and scaling features.
    
    Args:
        data: DataFrame with price data
        sequence_length: Length of input sequences
        
    Returns:
        Tuple of (X, y) arrays for model training
    """
    # Select features
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = data[features].values
    
    # Scale features
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i:(i + sequence_length)])
        y.append(data_scaled[i + sequence_length, 3])  # Predict Close price
    
    return np.array(X), np.array(y)

def inverse_transform_predictions(predictions: np.ndarray, 
                               scaler: MinMaxScaler,
                               feature_index: int = 3) -> np.ndarray:
    """
    Transform predictions back to original scale.
    
    Args:
        predictions: Array of scaled predictions
        scaler: Fitted MinMaxScaler object
        feature_index: Index of the feature being predicted
        
    Returns:
        Array of predictions in original scale
    """
    # Create dummy array with same shape as original data
    dummy = np.zeros((len(predictions), scaler.n_features_in_))
    dummy[:, feature_index] = predictions
    
    # Inverse transform
    return scaler.inverse_transform(dummy)[:, feature_index] 