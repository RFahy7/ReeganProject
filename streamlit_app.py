import streamlit as st
import pandas as pd

#Page title
st.set_page_config(page_title="Predicting Amazon Stock Prices Using Headlines", layout="wide")

#Title
st.title("Predicting Amazon Stock Prices Using Headlines")

#User input
headline_input = st.text_area("Enter one or more headlines (separated by new lines):")
current_price = st.number_input("Enter today's price of Amazon stock:", min_value=0.0, format="%.2f")





