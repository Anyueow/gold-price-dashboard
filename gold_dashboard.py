import streamlit as st
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from gold_analyzer.data.fetcher import GoldDataFetcher
from gold_analyzer.models.predictor import GoldPricePredictor
from gold_analyzer.visualization.plots import GoldVisualizer
from sentiment.sentiment_analysis import SentimentAnalyzer, format_sentiment_for_dashboard

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Gold Price Prediction Dashboard",
    page_icon="💰",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Gold Price Prediction Dashboard")
st.markdown("---")

# Initialize components
if 'gold_data' not in st.session_state:
    st.session_state.gold_data = None
if 'sentiment_analyzer' not in st.session_state:
    st.session_state.sentiment_analyzer = SentimentAnalyzer(grok_api_key=os.getenv('GROK_API_KEY'))
if 'data_fetcher' not in st.session_state:
    st.session_state.data_fetcher = GoldDataFetcher()
if 'predictor' not in st.session_state:
    st.session_state.predictor = GoldPricePredictor()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = GoldVisualizer()

# User type selection
user_type = st.radio(
    "Select User Type",
    ["Trader", "Consumer"],
    horizontal=True
)

# Fetch and store gold data
if st.session_state.gold_data is None:
    st.session_state.gold_data = st.session_state.data_fetcher.get_data()

# Main dashboard content
if user_type == "Trader":
    st.header("Trader View")
    
    if st.session_state.gold_data is not None:
        # Create two columns for metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Current Gold Price",
                value=f"${st.session_state.gold_data['Close'].iloc[-1]:.2f}",
                delta=f"{st.session_state.gold_data['Close'].pct_change().iloc[-1]:.2%}"
            )
        
        with col2:
            st.metric(
                label="14-Day Volatility",
                value=f"{st.session_state.gold_data['Close'].pct_change().std() * 14 ** 0.5:.2%}",
                delta=None
            )
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Price Analysis", "Technical Indicators"])
        
        with tab1:
            # Get predictions and create price plot
            predictions = st.session_state.predictor.predict_price(st.session_state.gold_data)
            price_plot = st.session_state.visualizer.create_price_prediction_plot(
                st.session_state.gold_data,
                predictions
            )
            st.plotly_chart(price_plot, use_container_width=True)
            
            # Display prediction metrics
            if predictions is not None:
                metrics = st.session_state.predictor.calculate_prediction_metrics(
                    predictions,
                    st.session_state.gold_data['Close'].iloc[-1]
                )
                
                st.markdown("### Price Prediction Metrics")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Predicted Price (14 days)",
                        value=f"${metrics['predicted_price']:.2f}"
                    )
                
                with col2:
                    st.metric(
                        label="Predicted Change",
                        value=f"{metrics['predicted_change']:.2%}"
                    )
                
                with col3:
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{metrics['confidence']:.0%}"
                    )
        
        with tab2:
            # Get technical indicators and create plots
            indicators = st.session_state.data_fetcher.get_technical_indicators(st.session_state.gold_data)
            indicator_plots = st.session_state.visualizer.create_technical_indicators_plot(
                st.session_state.gold_data,
                indicators
            )
            
            st.markdown("### Technical Indicators")
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(indicator_plots['RSI'], use_container_width=True)
            
            with col2:
                st.plotly_chart(indicator_plots['MA'], use_container_width=True)

else:  # Consumer View
    st.header("Consumer View")
    
    if st.session_state.gold_data is not None:
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Price Trends", "Buying Guide"])
        
        with tab1:
            # Get sentiment analysis and create combined plot
            sentiment_report = st.session_state.sentiment_analyzer.generate_sentiment_report("gold price market trends")
            sentiment_data = format_sentiment_for_dashboard(sentiment_report)
            
            sentiment_plot = st.session_state.visualizer.create_sentiment_price_plot(
                st.session_state.gold_data,
                sentiment_data
            )
            st.plotly_chart(sentiment_plot, use_container_width=True)
            
            # Price trend indicators
            st.markdown("### Price Trend Indicators")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="30-Day Trend",
                    value=f"{((st.session_state.gold_data['Close'].iloc[-1] - st.session_state.gold_data['Close'].iloc[-30]) / st.session_state.gold_data['Close'].iloc[-30]):.2%}"
                )
            
            with col2:
                st.metric(
                    label="90-Day Trend",
                    value=f"{((st.session_state.gold_data['Close'].iloc[-1] - st.session_state.gold_data['Close'].iloc[-90]) / st.session_state.gold_data['Close'].iloc[-90]):.2%}"
                )
            
            with col3:
                if "error" not in sentiment_data:
                    st.metric(
                        label="Market Sentiment",
                        value=sentiment_data["sentiment_direction"].capitalize()
                    )
        
        with tab2:
            # Buying guide
            st.markdown("### Gold Buying Guide")
            
            # Price comparison
            st.markdown("#### Price Comparison")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Current Price per Gram",
                    value=f"${st.session_state.gold_data['Close'].iloc[-1] / 31.1035:.2f}"
                )
            
            with col2:
                st.metric(
                    label="Price per Ounce",
                    value=f"${st.session_state.gold_data['Close'].iloc[-1]:.2f}"
                )
            
            # Get buying recommendation
            current_price = st.session_state.gold_data['Close'].iloc[-1]
            ma_30 = st.session_state.gold_data['Close'].rolling(window=30).mean().iloc[-1]
            recommendation = st.session_state.predictor.get_buying_recommendation(current_price, ma_30)
            
            st.markdown("#### Buying Recommendations")
            if recommendation['recommendation'] == 'Buy':
                st.success(f"{recommendation['reason']} - Good buying opportunity!")
            else:
                st.warning(f"{recommendation['reason']} - Consider waiting for better prices")
            
            # Price alerts
            st.markdown("#### Set Price Alerts")
            alert_price = st.number_input(
                "Enter your target price per ounce",
                min_value=0.0,
                value=float(current_price),
                step=10.0
            )
            
            if st.button("Set Price Alert"):
                st.success(f"Alert set for ${alert_price:.2f} per ounce")

# Footer
st.markdown("---")
st.markdown("Data source: Yahoo Finance | Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")) 