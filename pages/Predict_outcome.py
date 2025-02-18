import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load dataset
data_path = "./sorce.csv"
df = pd.read_csv(data_path)
col_forecast, col_prices = st.columns(2)
@st.cache_data
def load_data():
    file_path = "sorce.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        st.error("❌ 'Date' column not found in dataset.")
        return None
    
    # Ensure required columns are present (needed for various analyses)
    required_columns = {"Company", "Close", "Volume", "Open", "High", "Low"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    
    # Handle duplicate entries by aggregating numeric values (averaging prices, summing volume)
    df = df.groupby(["Date", "Company"], as_index=False).agg({
        "Open": "mean",
        "High": "mean",
        "Low": "mean",
        "Close": "mean",
        "Volume": "sum"
    })
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    df["Year"] = df["Date"].dt.year
    return df

df = load_data()
if df is not None:
    st.sidebar.header("Dashboard Filters")
    selected_company = st.sidebar.selectbox("🔍 Select a Company", df["Company"].unique())

    # -------------------------
    # Forecasting & Latest Price
    # -------------------------
    col_forecast, col_prices = st.columns(2)
    
    with col_forecast:
        company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        if len(company_data) > 0:
            model = Prophet()
            model.fit(company_data)
            
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
            
            
            fig = go.Figure()
            
           
            fig.add_trace(go.Scatter(
                x=company_data["ds"], 
                y=company_data["y"], 
                mode='lines', 
                name='Actual Data',
                line=dict(color='blue')
            ))
            
            # Predicted data
            fig.add_trace(go.Scatter(
                x=forecast["ds"], 
                y=forecast["yhat"], 
                mode='lines', 
                name='Predicted Data',
                line=dict(color='red')
            ))
            
            
            fig.update_layout(
                title=f"Stock Price Forecast - {selected_company}",
                xaxis_title='Date',
                yaxis_title='Stock Price ($)',
                template='plotly_white'
            )
            
           
            st.plotly_chart(fig)

    
    with col_prices:
        latest_data_all = df.groupby("Company").last()
        trade_price_table = latest_data_all[["Close"]].rename(columns={"Close": "Last Trade Price"})
        st.subheader("Latest Trade Prices")
        st.dataframe(trade_price_table, use_container_width=True)
# Ensure column names are correct
data_path = "./sorce.csv"
df = pd.read_csv(data_path)
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

    
    model = train_regression_model(df)

    
    st.title("Stock Price Prediction with Linear Regression")
    st.header("User Input")

    
    col1,col2,col3=st.columns(3)
    with col1:
     company = st.selectbox("Select Company", companies)
     open_price = st.number_input("Open Price", min_value=0.0, value=100.0)
    with col2:
        high_price = st.number_input("High Price", min_value=0.0, value=110.0)
        low_price = st.number_input("Low Price", min_value=0.0, value=90.0)
    with col3:
        volume = st.number_input("Volume", min_value=0, value=1000000)

    
    input_data = pd.DataFrame([[open_price, high_price, low_price, volume]], 
                               columns=["Open", "High", "Low", "Volume"])

    
    if st.button("Predict"):
        prediction = model.predict(input_data)[0]
        st.write(f"### Predicted Close Price: {prediction:.2f}")
