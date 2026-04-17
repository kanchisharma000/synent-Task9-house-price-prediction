import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv("house_data.csv")

# Prepare data
X = df[['area', 'Year', 'Rooms', 'Floor']]
y = df['Price']

X = X.apply(pd.to_numeric, errors='coerce')  # convert to numbers
X = X.dropna()                               # remove missing values
y = y[X.index]                               # align target with cleaned data

# Train model
model = LinearRegression()
model.fit(X, y)

# Title
st.title("🏠 House Price Prediction App")

# Description
st.write("Enter details below to predict house price")

# Inputs
area = st.number_input("Area (sq ft)")
year = st.number_input("Year")
rooms = st.number_input("Number of Rooms")
floor = st.number_input("Floor")

# Prediction button
if st.button("Predict Price"):
    prediction = model.predict([[area, year, rooms, floor]])
    st.success(f"Estimated Price: {prediction[0]:,.2f}")