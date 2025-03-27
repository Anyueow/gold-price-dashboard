# Gold Price Prediction Dashboard

A comprehensive dashboard for analyzing and predicting gold prices, featuring both technical analysis and sentiment analysis capabilities.

## Features

- Real-time gold price data fetching from multiple sources
- Technical analysis indicators (RSI, Moving Averages, Volatility)
- Price predictions using statistical models
- Sentiment analysis using Grok API
- Interactive visualizations using Plotly
- Separate views for traders and consumers
- Price alerts and buying recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gold-price-dashboard.git
cd gold-price-dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory with your Grok API key:
```
GROK_API_KEY=your_api_key_here
```

## Usage

Run the dashboard:
```bash
streamlit run gold_dashboard.py
```

## Project Structure

```
gold-price-dashboard/
├── gold_analyzer/
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetcher.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── predictor.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   └── plots.py
│   └── __init__.py
├── sentiment/
│   ├── __init__.py
│   └── sentiment_analysis.py
├── gold_dashboard.py
├── requirements.txt
├── .env
└── README.md
```

## Dependencies

- streamlit==1.32.0
- pandas==2.2.0
- numpy==1.26.3
- plotly==5.18.0
- yfinance==0.2.36
- requests==2.31.0
- python-dotenv==1.0.0

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 