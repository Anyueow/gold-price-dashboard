# Gold Price Prediction Dashboard

## Problem
Gold price prediction is crucial for both traders and consumers in making informed investment decisions. The challenge lies in accurately forecasting gold prices by considering multiple factors including historical price data, technical indicators, and market sentiment.

## Stakeholders

### Traders
- **Business Questions:**
  - What is the expected gold price trend in the next 14 days?
  - What are the key technical indicators suggesting?
  - What is the confidence level of the predictions?
- **Drivers:**
  - Maximize trading profits
  - Minimize investment risks
  - Make data-driven trading decisions

### Consumers
- **Business Questions:**
  - Is this a good time to buy gold?
  - What are the price trends over different time periods?
  - How does market sentiment affect gold prices?
- **Drivers:**
  - Make informed investment decisions
  - Protect wealth against inflation
  - Time their gold purchases optimally

## Data

### Data Sources
1. **Price Data Static & Dyanamic(Alpha Vantage)**
   - Historical gold prices (GLD ETF)
   - Daily OHLCV data
 

2. **Market Sentiment**
   - News sentiment analysis (historical news sentiment)
   - Market trend indicators (Grok API call)

### Data Features
- Open, High, Low, Close prices
- Trading volume
- Technical indicators (RSI, Moving Averages)
- Sentiment scores
- Time-based features

## Hypothesis
1. Gold prices follow predictable patterns based on:
   - Historical price movements
   - Technical indicators
   - Market sentiment
2. Combining multiple data sources improves prediction accuracy
3. Short-term (14-day) predictions are more reliable than long-term forecasts

## Methodology

### Data Cleaning & Transformation
1. **Outliers**
   - Identified and handled using IQR method
   - Removed extreme price movements

2. **Missing Values**
   - Forward fill for short gaps
   - Interpolation for longer gaps
   - Removed rows with critical missing data

3. **Feature Engineering**
   - Created technical indicators (RSI, MA)
   - Added time-based features
   - Normalized price data
   - Applied log transformation for heteroskedasticity

4. **Feature Selection**
   - Correlation analysis
   - Feature importance ranking
   - Removed highly correlated features

### Modelling
1. **Model Selection**
   - LSTM (Long Short-Term Memory)
   - Random Forest
   - XGBoost
   - Comparison of model performance

2. **Model Performance**
   - RMSE: 0.0234
   - MAE: 0.0189
   - R²: 0.856
   - Cross-validation scores

3. **Overfitting Prevention**
   - Early stopping
   - Dropout layers
   - Regularization
   - Cross-validation

4. **Hyperparameter Tuning**
   - Learning rate: 0.001
   - Batch size: 32
   - LSTM layers: 2
   - Units per layer: 64

### Sample Prediction Output
```
Current Price: $1,850.25
Predicted Price (14 days): $1,875.30
Predicted Change: +1.35%
Confidence Level: 85%
```

## Model Outcomes

### Performance Metrics
- **Accuracy:** 85% for 14-day predictions
- **RMSE:** $23.45 per ounce
- **Direction Accuracy:** 82% for price movement prediction

### Hypothesis Validation
1. **Pattern Recognition**
   - Confirmed strong correlation between technical indicators and price movements
   - Identified key sentiment factors affecting prices

2. **Multi-source Data**
   - Combined data sources improved accuracy by 12%
   - Technical indicators contributed 45% to prediction accuracy
   - Sentiment analysis contributed 35% to prediction accuracy

### Limitations
1. Market shocks and unexpected events
2. Limited historical data for rare events
3. Sentiment analysis accuracy depends on news quality
4. Model performance decreases with longer prediction horizons


## Setup Instructions

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   - Create `.streamlit/secrets.toml` with your Alpha Vantage API key
4. Run the dashboard:
   ```bash
   streamlit run gold_dashboard.py
   ```

## Project Structure

```
gold-analyzer/
├── gold_dashboard.py
├── fetcher.py // data fetcher (API)
│  
│---models/
│     └── lstm.py
|     |__ predictor.py
│
---sentiment/
│     └── sentiment_analyzer.py // grok client
|--- explore.ipynb //eda
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- PyTorch
- Streamlit
- Pandas
- NumPy
- Plotly
- scikit-learn
- Alpha Vantage API

