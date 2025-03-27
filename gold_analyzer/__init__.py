"""
Gold Price Analysis Package

This package provides tools for analyzing and predicting gold prices,
including data fetching, technical analysis, and visualization capabilities.
"""

from .data.fetcher import GoldDataFetcher
from .models.predictor import GoldPricePredictor
from .visualization.plots import GoldVisualizer

__version__ = '0.1.0'
__all__ = ['GoldDataFetcher', 'GoldPricePredictor', 'GoldVisualizer'] 