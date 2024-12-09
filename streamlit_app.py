import streamlit as st
import numpy as np
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk


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

        # Calculate average sentiment score
        today_average_sentiment = compute_average_sentiment(today_headline_input.splitlines())
        yesterday_average_sentiment = compute_average_sentiment(yesterday_headline_input.splitlines())

        # Display sentiment results
        st.write(f"Today's sentiment score: {today_average_sentiment:.2f}")
        st.write(f"Yesterday's sentiment score: {yesterday_average_sentiment:.2f}")

        # Prepare features for Ridge model
        features = np.array([[today_average_sentiment, yesterday_average_sentiment]])

        # Load and predict with the Ridge model
        try:
            ridge_model = joblib.load('ridge.joblib')
            st.success("Ridge model loaded successfully.")

            # Predict the change in price
            predicted_change = ridge_model.predict(features)[0]
            st.write(f"Predicted Change in Stock Price: {predicted_change:.2f}")

            # Calculate the predicted stock price for tomorrow
            predicted_price_tomorrow = current_price + predicted_change
            st.write(f"Predicted Stock Price for Tomorrow: {predicted_price_tomorrow:.2f}")

        except FileNotFoundError:
            st.error("Ridge model file not found. Please verify the location of 'ridge.joblib'.")
        except ValueError as e:
            st.error(f"Prediction error: {e}")

    else:
        st.warning("Please ensure all fields are filled out: headlines for today, yesterday, and today's stock price.")
