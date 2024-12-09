import streamlit as st
import pandas as pd
import numpy as np
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

# Title
st.title("Predicting Amazon Stock Prices Using Headlines")

# Collect user input for today's headlines
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

        # Load the vectorizer
        vectorizer = joblib.load('vectorizer.joblib')

        # Prepare features for Ridge regression
        feature_array = np.array([[today_average_sentiment, yesterday_average_sentiment]])

        # Vectorize the average sentiment scores
        feature_tfidf = vectorizer.transform(feature_array.astype('U'))

        # Display TF-IDF representation
        st.write("TF-IDF feature representation for sentiment scores:")
        st.write(pd.DataFrame(feature_tfidf.toarray(), columns=vectorizer.get_feature_names_out()).head(10))

        # Placeholder for Ridge model processing
        # ridge = Ridge()  # Uncomment and load your trained Ridge model as needed
        # ridge_result = ridge.predict(...)  # Use the features to predict or implement your logic

        # Save the Ridge model (or results as relevant)
        # joblib.dump(ridge, 'ridge.joblib')  # Uncomment if you need to save the Ridge model
    else:
        st.warning("Please enter headlines for both today and yesterday.")
