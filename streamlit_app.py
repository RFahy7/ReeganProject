import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

# Title
st.title("Predicting Amazon Stock Prices Using Headlines")

# User input
headline_input1 = st.text_area("Enter one or more headlines from today (separated by new lines):")
headline_input2 = st.text_area("Enter one of more headlines from yesterday (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")

# Load the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Process the input
if st.button("Process Headlines"):
    if headline_input1:
        # Split input into separate headlines
        headlines1 = headline_input1.splitlines()
        
        # List to store sentiment scores
        sentiment_scores1 = []

        # Calculate the VADER sentiment score for each headline
        for headline in headlines1:
            score1 = sia.polarity_scores(headline)  # VADER returns a dictionary
            sentiment_scores1.append(score1['compound'])  # Use the compound score

        # Display the sentiment scores
        st.write("TODAY: VADER Sentiment Scores per Headline:")
        st.write(sentiment_scores1)
        
        # Calculate average sentiment score
        average_score1 = sum(sentiment_scores1) / len(sentiment_scores1)

        # Load the vectorizer
        vectorizer = joblib.load('vectorizer.joblib')

        # Transform the average sentiment score into vectorizer
        score_array1 = np.array([[average_score1]])  
        average_score_tfidf1 = vectorizer.transform(score_array1)

        #Load the Ridge model
        ridge = joblib.load('ridge.joblib')

        #

    else:
        st.warning("Please enter at least one headline.")
