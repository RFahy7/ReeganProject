import streamlit as st
import pandas as pd
import joblib
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Ensure the VADER lexicon is downloaded
nltk.download('vader_lexicon')

# Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

# Title
st.title("Predicting Amazon Stock Prices Using Headlines")

# User input
headline_input = st.text_area("Enter one or more headlines (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")

# Load the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Vectorize input
if st.button("Process Headlines"):
    if headline_input:
        # Split input into separate headlines
        headlines = headline_input.splitlines()

        # List to store sentiment scores
        sentiment_scores = []

        # Calculate the VADER sentiment score for each headline
        for headline in headlines:
            score = sia.polarity_scores(headline)  # VADER returns a dictionary
            sentiment_scores.append(score['compound'])  # Use the compound score

        # Display the sentiment scores
        st.write("VADER Sentiment Scores per Headline:")
        st.write(sentiment_scores)

        # Calculate average sentiment score
        average_score = sum(sentiment_scores) / len(sentiment_scores)

        # Categorize average score
        if average_score >= 0.05:
            sentiment_category = "Positive"
        elif average_score <= -0.05:
            sentiment_category = "Negative"
        else:
            sentiment_category = "Neutral"

        # Calculate percentage
        sentiment_percentage = abs(average_score) * 100

        # Display average sentiment and percentage
        st.write(f"The averaged sentiment of the headlines is: **{sentiment_category}**")
        st.write(f"Sentiment Confidence: **{sentiment_percentage:.2f}%**")

        # Load the vectorizer
        vectorizer = joblib.load('vectorizer.joblib')

        # Transform the sentiment scores into vectorizer
        scores_df = pd.DataFrame(sentiment_scores, columns=['sentiment'])
        headline_tfidf = vectorizer.transform(scores_df['sentiment'].values.astype('U'))

        # You can now use the transformed data in your model or display some transformation details
        st.write("Sample of TF-IDF feature representation based on sentiment scores:")
        st.write(pd.DataFrame(headline_tfidf.toarray(), columns=vectorizer.get_feature_names_out()).head(10))

    else:
        st.warning("Please enter at least one headline.")
