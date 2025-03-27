import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from torch.utils.data import Dataset, DataLoader

class GoldDataset(Dataset):
    """Dataset class for gold price data."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

class GoldPricePredictor(nn.Module):
    """LSTM model for gold price prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Get predictions from last time step
        out = self.fc(out[:, -1, :])
        return out.squeeze()
    
    def train(self, X: np.ndarray, y: np.ndarray,
              epochs: int = 50, batch_size: int = 32,
              learning_rate: float = 0.001) -> None:
        """
        Train the model.
        
        Args:
            X: Input features
            y: Target values
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimization
        """
        # Create dataset and dataloader
        dataset = GoldDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
        # Training loop
        self.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input features
            
        Returns:
            Array of predictions
        """
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self(X_tensor)
            return predictions.numpy()
    
    def save(self, path: str) -> None:
        """
        Save model state.
        
        Args:
            path: Path to save model state
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.lstm.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.lstm.dropout
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'GoldPricePredictor':
        """
        Load model state.
        
        Args:
            path: Path to load model state from
            
        Returns:
            Loaded model instance
        """
        checkpoint = torch.load(path)
        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model 