import requests
from datetime import datetime
from typing import Dict
import numpy as np
from openai import OpenAI

class SentimentAnalyzer:
    def __init__(self, xai_api_key: str):
        self.client = OpenAI(
            api_key=xai_api_key,
            base_url="https://api.x.ai/v1",
        )

    def get_sentiment(self, query: str) -> Dict:
        """Get sentiment analysis using Grok model"""
        try:
            # Construct the prompt
            prompt = f"""
            Analyze the sentiment of the following gold market related text:
            "{query}"
            
            Provide a JSON response with:
            1. sentiment: overall sentiment (positive/negative/neutral)
            2. confidence_score: confidence score (0-1)
            3. key_indicators: list of key sentiment indicators
            4. market_impact: assessment of market impact (high/medium/low)
            5. price_trend: price trend prediction (up/down/stable)
            6. factors: key factors influencing sentiment
            """
            
            # Make API call to Grok
            response = self.client.chat.completions.create(
                model="grok-2-latest",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial sentiment analysis expert specializing in gold market analysis."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            
            # Extract the response content
            return response.choices[0].message.content
                
        except Exception as e:
            return {"error": f"Error in sentiment analysis: {str(e)}"}

    def calculate_sentiment_score(self, sentiment_data: Dict) -> Dict:
        """Calculate sentiment score from analysis"""
        try:
            # Extract sentiment components
            sentiment = sentiment_data.get("sentiment", "neutral")
            confidence = float(sentiment_data.get("confidence_score", 0))
            price_trend = sentiment_data.get("price_trend", "stable")
            
            # Convert sentiment to numerical score
            sentiment_map = {
                "positive": 1,
                "negative": -1,
                "neutral": 0
            }
            sentiment_score = sentiment_map.get(sentiment, 0)
            
            # Convert price trend to numerical score
            trend_map = {
                "up": 1,
                "down": -1,
                "stable": 0
            }
            trend_score = trend_map.get(price_trend, 0)
            
            # Calculate final weighted score
            final_score = (sentiment_score * 0.6 + trend_score * 0.4) * confidence
            
            # Calculate confidence interval
            confidence_interval = 1.96 * (1 - confidence)  # 95% confidence interval
            
            return {
                "aggregate_score": final_score,
                "confidence_interval": confidence_interval,
                "sentiment_score": sentiment_score,
                "trend_score": trend_score,
                "sentiment_strength": abs(final_score),
                "sentiment_direction": "positive" if final_score > 0 else "negative",
                "confidence_level": confidence,
                "market_impact": sentiment_data.get("market_impact", "Low")
            }
            
        except Exception as e:
            return {"error": f"Error in sentiment score calculation: {str(e)}"}

    def generate_sentiment_report(self, query: str) -> Dict:
        """Generate comprehensive sentiment report"""
        try:
            # Get sentiment analysis
            sentiment_data = self.get_sentiment(query)
            
            # Calculate sentiment score
            sentiment_score = self.calculate_sentiment_score(sentiment_data)
            
            # Generate report
            report = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "aggregate_sentiment": sentiment_score,
                "analysis": sentiment_data,
                "summary": {
                    "overall_sentiment": sentiment_score["sentiment_direction"],
                    "confidence": f"{sentiment_score['confidence_level']:.2%}",
                    "strength": f"{sentiment_score['sentiment_strength']:.2f}",
                    "market_impact": sentiment_score["market_impact"],
                    "price_trend": sentiment_data.get("price_trend", "stable").capitalize()
                }
            }
            
            return report
            
        except Exception as e:
            return {"error": f"Error generating sentiment report: {str(e)}"}

def format_sentiment_for_dashboard(sentiment_report: Dict) -> Dict:
    """Format sentiment report for dashboard display"""
    try:
        if "error" in sentiment_report:
            return {"error": sentiment_report["error"]}
            
        return {
            "current_sentiment": sentiment_report["aggregate_sentiment"]["aggregate_score"],
            "sentiment_direction": sentiment_report["summary"]["overall_sentiment"],
            "confidence": sentiment_report["summary"]["confidence"],
            "market_impact": sentiment_report["summary"]["market_impact"],
            "price_trend": sentiment_report["summary"]["price_trend"],
            "timestamp": sentiment_report["timestamp"]
        }
        
    except Exception as e:
        return {"error": f"Error formatting sentiment data: {str(e)}"} 