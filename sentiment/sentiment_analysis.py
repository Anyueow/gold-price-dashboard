import requests
from datetime import datetime
from typing import Dict
import numpy as np

class SentimentAnalyzer:
    def __init__(self, grok_api_key: str):
        self.grok_api_key = grok_api_key
        self.base_url = "https://api.grok.ai/v1"

    def get_grok_sentiment(self, query: str) -> Dict:
        """Get sentiment analysis from Grok API"""
        try:
            # Construct the prompt for Grok
            prompt = f"""
            Analyze the sentiment of the following gold market related text:
            "{query}"
            
            Provide a detailed analysis including:
            1. Overall sentiment (positive/negative/neutral)
            2. Confidence score (0-1)
            3. Key sentiment indicators
            4. Market impact assessment
            5. Price trend prediction (up/down/stable)
            6. Key factors influencing sentiment
            """
            
            # Make API call to Grok
            headers = {
                "Authorization": f"Bearer {self.grok_api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                f"{self.base_url}/analyze",
                headers=headers,
                json={"prompt": prompt}
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"Grok API error: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Error in Grok sentiment analysis: {str(e)}"}

    def calculate_sentiment_score(self, grok_data: Dict) -> Dict:
        """Calculate sentiment score from Grok analysis"""
        try:
            # Extract sentiment components
            sentiment = grok_data.get("sentiment", "neutral")
            confidence = float(grok_data.get("confidence_score", 0))
            price_trend = grok_data.get("price_trend", "stable")
            
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
                "market_impact": "High" if abs(final_score) > 0.7 else 
                               "Medium" if abs(final_score) > 0.4 else "Low"
            }
            
        except Exception as e:
            return {"error": f"Error in sentiment score calculation: {str(e)}"}

    def generate_sentiment_report(self, query: str) -> Dict:
        """Generate comprehensive sentiment report"""
        try:
            # Get sentiment from Grok
            grok_sentiment = self.get_grok_sentiment(query)
            
            # Calculate sentiment score
            sentiment_score = self.calculate_sentiment_score(grok_sentiment)
            
            # Generate report
            report = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "aggregate_sentiment": sentiment_score,
                "grok_analysis": grok_sentiment,
                "summary": {
                    "overall_sentiment": sentiment_score["sentiment_direction"],
                    "confidence": f"{sentiment_score['confidence_level']:.2%}",
                    "strength": f"{sentiment_score['sentiment_strength']:.2f}",
                    "market_impact": sentiment_score["market_impact"],
                    "price_trend": grok_sentiment.get("price_trend", "stable").capitalize()
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