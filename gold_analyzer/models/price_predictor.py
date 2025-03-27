import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from typing import Tuple, Dict, Any, List
import tensorflow as tf

class GoldPricePredictor:
    """Class to handle gold price predictions using multiple models."""
    
    def __init__(self, sequence_length: int = 10):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler()
        self.models = {}
        self.feature_columns = [
            'Close', 'Open', 'High', 'Low', 'Volume',
            'RSI', 'MA20', 'MA50', 'Volatility'
        ]
    
    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for modeling.
        
        Args:
            data: DataFrame with price and technical indicators
            
        Returns:
            Tuple of (X, y) arrays for modeling
        """
        # Calculate target variable (1 if price increases next day, 0 otherwise)
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        
        # Drop rows with NaN values
        data = data.dropna()
        
        # Scale features
        scaled_data = self.scaler.fit_transform(data[self.feature_columns])
        
        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(data['Target'].iloc[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple[int, int]) -> Sequential:
        """
        Build LSTM model architecture with simplified structure and batch normalization.
        
        Args:
            input_shape: Shape of input data (sequence_length, n_features)
            
        Returns:
            Compiled LSTM model
        """
        model = Sequential([
            LSTM(32, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy', AUC()]
        )
        
        return model
    
    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple models using time series cross-validation.
        
        Args:
            data: DataFrame with price and technical indicators
            
        Returns:
            Dictionary with trained models and performance metrics
        """
        # Prepare data
        X, y = self.prepare_data(data)
        
        # Initialize time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Initialize models
        self.models = {
            'logistic': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42)
        }
        
        # Initialize LSTM model
        lstm_model = self.build_lstm_model((self.sequence_length, len(self.feature_columns)))
        
        # Store performance metrics
        metrics = {
            'logistic': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'random_forest': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'gradient_boosting': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []},
            'lstm': {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}
        }
        
        # Perform time series cross-validation
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train and evaluate traditional models
            for name, model in self.models.items():
                # Reshape data for traditional models
                X_train_flat = X_train.reshape(X_train.shape[0], -1)
                X_val_flat = X_val.reshape(X_val.shape[0], -1)
                
                # Train model
                model.fit(X_train_flat, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val_flat)
                y_pred_proba = model.predict_proba(X_val_flat)[:, 1]
                
                # Calculate metrics
                metrics[name]['accuracy'].append(accuracy_score(y_val, y_pred))
                metrics[name]['precision'].append(precision_score(y_val, y_pred))
                metrics[name]['recall'].append(recall_score(y_val, y_pred))
                metrics[name]['f1'].append(f1_score(y_val, y_pred))
                metrics[name]['auc'].append(roc_auc_score(y_val, y_pred_proba))
            
            # Train and evaluate LSTM model with early stopping
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_auc',
                patience=10,
                restore_best_weights=True
            )
            
            lstm_model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Make predictions with LSTM
            y_pred_lstm = (lstm_model.predict(X_val) > 0.5).astype(int)
            y_pred_proba_lstm = lstm_model.predict(X_val)
            
            # Calculate metrics for LSTM
            metrics['lstm']['accuracy'].append(accuracy_score(y_val, y_pred_lstm))
            metrics['lstm']['precision'].append(precision_score(y_val, y_pred_lstm))
            metrics['lstm']['recall'].append(recall_score(y_val, y_pred_lstm))
            metrics['lstm']['f1'].append(f1_score(y_val, y_pred_lstm))
            metrics['lstm']['auc'].append(roc_auc_score(y_val, y_pred_proba_lstm))
        
        # Calculate average metrics
        for model_name in metrics:
            for metric in metrics[model_name]:
                metrics[model_name][metric] = np.mean(metrics[model_name][metric])
        
        # Store the LSTM model
        self.models['lstm'] = lstm_model
        
        return metrics
    
    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using all trained models.
        
        Args:
            data: DataFrame with price and technical indicators
            
        Returns:
            Dictionary with predictions from all models
        """
        # Prepare data
        X, _ = self.prepare_data(data)
        
        # Get the last sequence for prediction
        last_sequence = X[-1:]
        
        predictions = {}
        
        # Make predictions with traditional models
        for name, model in self.models.items():
            last_sequence_flat = last_sequence.reshape(1, -1)
            pred_proba = model.predict_proba(last_sequence_flat)[0]
            predictions[name] = {
                'direction': 'up' if pred_proba[1] > 0.5 else 'down',
                'probability': pred_proba[1]
            }
        
        # Make prediction with LSTM
        lstm_pred = self.models['lstm'].predict(last_sequence)[0][0]
        predictions['lstm'] = {
            'direction': 'up' if lstm_pred > 0.5 else 'down',
            'probability': lstm_pred
        }
        
        return predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from Random Forest model.
        
        Returns:
            Dictionary with feature importance scores
        """
        if 'random_forest' not in self.models:
            return {}
        
        # Get feature names for all features in sequences
        feature_names = []
        for i in range(self.sequence_length):
            for col in self.feature_columns:
                feature_names.append(f"{col}_t-{self.sequence_length-i-1}")
        
        # Get feature importance
        importance = self.models['random_forest'].feature_importances_
        
        # Calculate average importance for each feature across time steps
        avg_importance = {}
        for col in self.feature_columns:
            col_indices = [i for i, name in enumerate(feature_names) if name.startswith(col)]
            avg_importance[col] = np.mean(importance[col_indices])
        
        return avg_importance 