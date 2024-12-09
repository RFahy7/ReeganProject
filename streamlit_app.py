import streamlit as st
import pandas as pd
import joblib

#Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

#Title
st.title("Predicting Amazon Stock Prices Using Headlines")

#User input
headline_input = st.text_area("Enter one or more headlines (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")

#Vectorize input
if st.button("Process Headlines"):
    if headline_input:
        # Split input into separate headlines
        headlines = headline_input.splitlines()

        # Load the vectorizer
        vectorizer = joblib.load('vectorizer.joblib')

        # Transform the input headlines
        headline_tfidf = vectorizer.transform(headlines)

        # Display the TF-IDF shape and some example data from the transformation
        st.write(f"Number of features: {headline_tfidf.shape[1]}")
        st.write("Sample of TF-IDF feature representation:")
        st.write(pd.DataFrame(headline_tfidf.toarray(), columns=vectorizer.get_feature_names_out()).head(10))
    else:
        st.warning("Please enter at least one headline.")



