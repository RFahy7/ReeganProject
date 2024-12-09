import streamlit as st
import pandas as pd

#Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

#Title
st.title(""Predicting Amazon Stock Prices Using Headlines")

# File uploader
uploaded_file = st.file_uploader("aggregated_df.cvs", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
     





