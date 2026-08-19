# Class Late Binding Annotaion
from __future__ import annotations 

# Import Python Native Lib
import os, sys

from typing_extensions import Literal

# Transformer Lib
import transformers
from transformers import pipeline


# Sentiment Analysis Class
class SentimentAnalysis:

    def __init__(self):

        self.sentiment_pipeline = pipeline(
            model = "distilbert-base-uncased-finetuned-sst-2-english",
            task = "sentiment-analysis",
            framework = "pt",
            torch_dtype = "auto",
            model_kwargs={
                "local_files_only": True
            }
        )

        self.sentiment_threshold = 0.90

    # Check for Sentiment in Sentence
    def check_sentiment(self, input_str: str) -> Literal["POSITIVE", "NEGATIVE"]:
        try:
            result = self.sentiment_pipeline(input_str)

            sentiment = result[0]["label"]
            sentiment_score = round(result[0]["score"], 2)

            sentiment_value = None
            if ((sentiment == "POSITIVE") and (sentiment_score >= self.sentiment_threshold)):
                sentiment_value = sentiment
            else:
                sentiment_value = sentiment

            return(sentiment_value)
        
        except Exception as exp:
            error = f"Error in Checking Sentiment of AI Response - {exp}"
            print(error, flush=True)
            return("NEGATIVE")

# Start the main program when running standalone
if __name__ == "__main__":
    try:
        sentiment_agent = SentimentAnalysis()
        print("Sentiment: ", sentiment_agent.check_sentiment("AI: User Age Validation Failed."))

    except KeyboardInterrupt:
        print("\nShutdown signal received. Cleaning up resources...", file=sys.stderr)
        print("Server successfully stopped. Goodbye!", file=sys.stderr)


