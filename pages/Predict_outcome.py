import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load dataset
data_path = "./sorce.csv"
df = pd.read_csv(data_path)

# Ensure column names are correct
df.columns = df.columns.str.strip()
if "Company" not in df.columns:
    st.error("Error: 'Company' column is missing from the dataset.")
else:
    # Preprocess Data
    df = df.drop(columns=["Date", "Year", "Adj Close"])  # Remove unnecessary columns
    companies = df["Company"].unique()

    def train_regression_model(df):
        X = df.drop(columns=["Close", "Company"], errors='ignore')
        y = df["Close"]
        model = LinearRegression()
        model.fit(X, y)
        return model

    # Train Model
    model = train_regression_model(df)

    # Streamlit UI
    st.title("Stock Price Prediction with Linear Regression")
    st.header("User Input")

    # User Inputs
    col1,col2,col3=st.columns(3)
    with col1:
     company = st.selectbox("Select Company", companies)
     open_price = st.number_input("Open Price", min_value=0.0, value=100.0)
    with col2:
        high_price = st.number_input("High Price", min_value=0.0, value=110.0)
        low_price = st.number_input("Low Price", min_value=0.0, value=90.0)
    with col3:
        volume = st.number_input("Volume", min_value=0, value=1000000)

    # Prepare input data
    input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], 
                               columns=["Open", "High", "Low", "Volume"])

    # Prediction
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.write(f"### Predicted Close Price: {prediction:.2f}")
