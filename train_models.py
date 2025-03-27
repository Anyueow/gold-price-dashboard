import pandas as pd
import numpy as np
from gold_analyzer.data.fetcher import GoldDataFetcher
from gold_analyzer.models.price_predictor import GoldPricePredictor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
import json
from textblob import TextBlob

def plot_model_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """Plot comparison of model metrics."""
    # Prepare data for plotting
    model_names = list(metrics.keys())
    metric_names = list(metrics[model_names[0]].keys())
    
    # Calculate grid dimensions
    n_metrics = len(metric_names)
    n_cols = min(2, n_metrics)  # Max 2 columns
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Flatten axes array for easier iteration
    axes_flat = axes.ravel()
    
    for i, metric in enumerate(metric_names):
        values = [metrics[model][metric] for model in model_names]
        ax = axes_flat[i]
        
        # Create bar plot
        bars = ax.bar(range(len(model_names)), values)
        
        # Set title and labels
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45)
        ax.set_ylim(0, 1)
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
    
    # Hide empty subplots if any
    for i in range(n_metrics, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('model_metrics.png')
    plt.close()

def plot_feature_importance(importance: Dict[str, float]) -> None:
    """Plot feature importance."""
    plt.figure(figsize=(10, 6))
    features = list(importance.keys())
    values = list(importance.values())
    
    # Create bar plot
    bars = plt.barh(range(len(features)), values)
    
    # Set title and labels
    plt.title('Feature Importance')
    plt.xlabel('Importance Score')
    plt.yticks(range(len(features)), features)
    
    # Add value labels on the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2.,
                f'{width:.3f}',
                ha='left', va='center')
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def main():
    # Load datasets
    gold = pd.read_csv('Google_Stock_Price_Train.csv', parse_dates=['Date'])
    news = pd.read_csv('gold-dataset-sinha-khandait.csv', parse_dates=['Dates'])
    gold_history = pd.read_csv('Gold HistoricalData_1628818323325.csv', parse_dates=['Date'])
    
    # Sort & set index
    gold.sort_values('Date', inplace=True)
    news.sort_values('Dates', inplace=True)
    gold_history.sort_values('Date', inplace=True)
    
    # Remove duplicates/nulls
    news.drop_duplicates(subset='News', inplace=True)
    news.dropna(subset=['News'], inplace=True)
    
    # Rename columns
    news = news.rename(columns={'Dates': 'Date'})
    
    # Convert dates
    news['Date'] = pd.to_datetime(news['Date'], format='mixed', dayfirst=True)
    gold['Date'] = pd.to_datetime(gold['Date'])
    gold_history['Date'] = pd.to_datetime(gold_history['Date'])
    
    # Calculate sentiment
    def get_sentiment(text):
        blob = TextBlob(text)
        return blob.sentiment.polarity, blob.sentiment.subjectivity
    
    news[['polarity', 'subjectivity']] = news['News'].apply(lambda x: pd.Series(get_sentiment(x)))
    
    # Flag if headline is about future movement
    future_keywords = ['will', 'expected', 'projected', 'forecast', 'anticipate']
    news['is_future'] = news['News'].str.lower().apply(lambda x: int(any(word in x for word in future_keywords)))
    
    # Daily sentiment mean
    daily_sentiment = news.groupby('Date').agg({
        'polarity': 'mean',
        'subjectivity': 'mean',
        'is_future': 'mean',
        'News': 'count'
    }).rename(columns={'News': 'news_volume'}).reset_index()
    
    # Merge with gold data
    data = pd.merge(gold, daily_sentiment, on='Date', how='left')
    
    # Calculate technical indicators
    data['price_range'] = data['High'] - data['Low']
    data['price_range_pct'] = data['price_range'] / data['Open']
    data['returns'] = data['Close'].pct_change()
    data['gap'] = (data['Open'] - data['Close'].shift(1)) / data['Close'].shift(1)
    
    # Technical indicators
    data['sma_5'] = data['Close'].rolling(window=5).mean()
    data['sma_20'] = data['Close'].rolling(window=20).mean()
    data['volatility_5'] = data['returns'].rolling(window=5).std()
    data['rsi'] = (
        data['returns']
        .apply(lambda x: 100 - (100 / (1 + (x if x > 0 else 0) / (-x if x < 0 else 1e-9))))
        .rolling(window=14)
        .mean()
    )
    
    # Volume features
    data['Volume'] = pd.to_numeric(data['Volume'].str.replace(',', ''), errors='coerce')
    data['volume_ma5'] = data['Volume'].rolling(window=5).mean()
    data['relative_volume'] = data['Volume'] / data['volume_ma5']
    data['volume_trend'] = data['Volume'].pct_change()
    data['price_volume_ratio'] = data['returns'].abs() / data['Volume']
    
    # Sentiment features
    sentiment_cols = ['polarity', 'subjectivity', 'is_future', 'news_volume']
    data[sentiment_cols] = data[sentiment_cols].fillna(method='ffill')
    
    data['sentiment_ma3'] = data['polarity'].rolling(window=3).mean()
    data['sentiment_ma7'] = data['polarity'].rolling(window=7).mean()
    data['sentiment_trend'] = data['polarity'].diff()
    data['sentiment_volatility'] = data['polarity'].rolling(window=5).std()
    data['cumulative_sentiment'] = data['polarity'].expanding().mean()
    
    # Combined features
    data['sent_vol_impact'] = data['polarity'] * data['relative_volume']
    data['price_sent_agreement'] = (data['returns'] * data['polarity']).apply(lambda x: 1 if x > 0 else -1)
    
    # Target variables
    data['next_day_return'] = data['returns'].shift(-1)
    data['target_direction'] = (data['next_day_return'] > 0).astype(int)
    data['target_significant'] = (data['next_day_return'].abs() > data['returns'].std()).astype(int)
    
    # Time-based features
    data['day_of_week'] = data['Date'].dt.dayofweek
    data['month'] = data['Date'].dt.month
    data['quarter'] = data['Date'].dt.quarter
    
    # Create dummy variables for day of week
    dow_dummies = pd.get_dummies(data['day_of_week'], prefix='dow')
    data = pd.concat([data, dow_dummies], axis=1)
    
    # Initialize and train models
    predictor = GoldPricePredictor(sequence_length=10)
    metrics = predictor.train_models(data)
    
    # Plot model metrics
    plot_model_metrics(metrics)
    
    # Get feature importance
    importance = predictor.get_feature_importance()
    plot_feature_importance(importance)
    
    # Make predictions
    predictions = predictor.predict(data)
    
    # Save results
    results = {
        'metrics': metrics,
        'feature_importance': importance,
        'predictions': predictions
    }
    
    with open('model_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nModel Performance Summary:")
    print("-" * 50)
    for model, model_metrics in metrics.items():
        print(f"\n{model.upper()}:")
        for metric, value in model_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    print("\nFeature Importance:")
    print("-" * 50)
    for feature, importance_score in importance.items():
        print(f"{feature}: {importance_score:.4f}")
    
    print("\nCurrent Predictions:")
    print("-" * 50)
    for model, pred in predictions.items():
        print(f"{model.upper()}: {pred['direction'].upper()} (Probability: {pred['probability']:.4f})")

if __name__ == "__main__":
    main() 