# Gold Price Analyzer

A Streamlit application for analyzing gold prices using machine learning and technical indicators.

## Features

- Real-time gold price data fetching from Alpha Vantage
- Technical indicators (RSI, Moving Averages)
- LSTM-based price prediction
- Interactive visualizations
- Model training and saving capabilities

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd gold-analyzer
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up Streamlit secrets:
Create a `.streamlit/secrets.toml` file with your Alpha Vantage API key:
```toml
ALPHA_VANTAGE_API_KEY = "your-api-key-here"
```

## Running the Application

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

## Usage

1. **Data Settings**
   - Choose the number of days to analyze
   - Toggle between live and sample data

2. **Model Settings**
   - Adjust sequence length, hidden size, and number of layers
   - Configure dropout rate

3. **Training Settings**
   - Set number of epochs, batch size, and learning rate
   - Click "Train Model" to start training

4. **Visualization**
   - View price data and technical indicators
   - Compare actual vs predicted prices
   - Analyze model performance metrics

## Project Structure

```
gold-analyzer/
├── app.py
├── gold_analyzer/
│   ├── data/
│   │   └── fetcher.py
│   ├── models/
│   │   └── lstm.py
│   └── utils/
│       ├── preprocessing.py
│       └── visualization.py
├── models/
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

## License

MIT License 