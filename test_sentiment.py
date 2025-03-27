from sentiment.sentiment_analysis import SentimentAnalyzer, format_sentiment_for_dashboard
import os
from dotenv import load_dotenv

def test_sentiment_analysis():
    # Load environment variables
    load_dotenv()
    
    # Get API key
    xai_api_key = os.getenv('XAI_API_KEY')
    if not xai_api_key:
        print("Error: XAI_API_KEY not found in environment variables")
        return
    else:
        print(f"API Key found: {xai_api_key[:5]}...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer(xai_api_key)
    
    # Test cases
    test_queries = [
        "Gold prices surged to new highs as investors seek safe haven assets amid market uncertainty",
        "Gold prices declined sharply as the Federal Reserve announced higher interest rates",
        "Gold prices remained stable as market participants await key economic data"
    ]
    
    print("\nTesting Sentiment Analysis System\n")
    
    for query in test_queries:
        print(f"\nAnalyzing query: {query}")
        print("-" * 50)
        
        try:
            # Generate sentiment report
            report = analyzer.generate_sentiment_report(query)
            
            if "error" in report:
                print(f"Error in report: {report['error']}")
                continue
                
            # Print raw analysis response for debugging
            print("\nRaw Analysis Response:")
            if "analysis" in report:
                print(report["analysis"])
            else:
                print("No analysis found in report")
                
            # Print formatted results
            formatted = format_sentiment_for_dashboard(report)
            print("\nFormatted Results:")
            print(f"Sentiment Score: {formatted['current_sentiment']:.2f}")
            print(f"Direction: {formatted['sentiment_direction']}")
            print(f"Confidence: {formatted['confidence']}")
            print(f"Market Impact: {formatted['market_impact']}")
            print(f"Price Trend: {formatted['price_trend']}")
            print(f"Timestamp: {formatted['timestamp']}")
            
        except Exception as e:
            print(f"Error processing query: {str(e)}")
            
        print("-" * 50)

if __name__ == "__main__":
    test_sentiment_analysis() 