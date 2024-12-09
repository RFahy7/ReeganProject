import streamlit as st
import pandas as pd
import numpy as np
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from sklearn.linear_model import Ridge

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

# Title
st.title("Predicting Amazon Stock Prices Using Headlines")

# Collect user input for today's and yesterday's headlines
today_headline_input = st.text_area("Enter today's headlines (separated by new lines):")
yesterday_headline_input = st.text_area("Enter yesterday's headlines (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")

# Load the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Process headlines
if st.button("Process Headlines"):
    if today_headline_input and yesterday_headline_input:
        # Function to compute average sentiment score
        def compute_average_sentiment(headlines):
            scores = [sia.polarity_scores(h)['compound'] for h in headlines]
            return sum(scores) / len(scores) if scores else 0.0


        # Split input into separate headlines and calculate average sentiment score for both days
        today_headlines = today_headline_input.splitlines()
        yesterday_headlines = yesterday_headline_input.splitlines()

        today_average_sentiment = compute_average_sentiment(today_headlines)
        yesterday_average_sentiment = compute_average_sentiment(yesterday_headlines)

        # Display sentiment results
        st.write(f"Today's sentiment score: {today_average_sentiment:.2f}")
        st.write(f"Yesterday's sentiment score: {yesterday_average_sentiment:.2f}")

        # Prepare features for Ridge regression
        features = np.array([[today_average_sentiment, yesterday_average_sentiment]])

        # Load the pre-trained Ridge model
        if 'ridge.joblib' in st.secrets:  # path can also be given directly if not using st.secrets
            ridge_model = joblib.load('ridge.joblib')
            # Prediction can be made here
            prediction = ridge_model.predict(features)
            st.write(f"Predicted Stock Price: {prediction}")
            st.success("Ridge model loaded successfully.")
        else:
            st.warning("Ridge model file not found. Please check the path.")

    else:
        st.warning("Please enter headlines for both today and yesterday.")
