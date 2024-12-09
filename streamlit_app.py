import streamlit as st
import pandas as pd

#Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

#Title
st.title("Predicting Amazon Stock Prices Using Headlines")

#User input
headline_input = st.text_area("Enter one or more headlines (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")

#Vectorize input
# Check if there is an input
if headline_input:
    headlines = headline_input.splitlines()
    vectorizer = joblib.load('vectorizer.joblib')
    headline_tfidf = vectorizer.transform(headlines)

    st.write(f"Number of features: {headline_tfidf.shape[1]}")
    st.write("Sample of TF-IDF feature representation:")
    st.write(pd.DataFrame(headline_tfidf.toarray(), columns=vectorizer.get_feature_names_out()).head(10))



