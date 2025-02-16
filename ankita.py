import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# Load Data
file_path = "sorce.xlsx"

@st.cache_data
def load_data():
    file_path = "sorce.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip() 

    # Display column names for debugging
    # st.write("Columns in dataset:", df.columns.tolist())

    # Ensure 'Date' column is in datetime format
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
    else:
        st.error("❌ 'Date' column not found in dataset.")

    # Ensure necessary columns exist
    required_columns = {"Company", "Close", "Volume"}
    if not required_columns.issubset(set(df.columns)):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    df = df[df["Year"] >= 2015]
    return df

df = load_data()

if df is not None:
    # Aggregate Data (Check if columns exist)
    if "Year" in df.columns and "Company" in df.columns:
        yearly_trade_value = df.groupby(["Year", "Company"])["Total_Trade_Value"].sum().unstack()
    else:
        st.error("❌ The dataset does not contain 'Year' or 'Company'. Please check the column names.")
        yearly_trade_value = None

    # Sidebar Filters
    st.sidebar.header("📌 Dashboard Filters")

    if yearly_trade_value is not None:
        selected_company = st.sidebar.selectbox("🔍 Select a Company", yearly_trade_value.columns)

        # Layout: Yearly Performance & Best/Worst Stocks
        st.markdown("## 📊 Stock Market Dashboard")
        col_vol1,col_vol2=st.columns(2)
# Layout: Yearly Performance & Best/Worst Stocks
    
    with col_vol1:   
        
        st.subheader("📊 Volatility Analysis")

        # Calculate daily log returns
        df_selected = df[df["Company"] == selected_company].sort_values("Date")
        df_selected["Log Returns"] = np.log(df_selected["Close"] / df_selected["Close"].shift(1))
        
        # Compute rolling standard deviation (volatility)
        df_selected["Volatility"] = df_selected["Log Returns"].rolling(window=30).std()
        
        # Plot volatility
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_selected["Date"], df_selected["Volatility"], label="30-Day Rolling Volatility", color="purple", linewidth=2)
        ax.axhline(df_selected["Volatility"].mean(), color="red", linestyle="dashed", label="Avg Volatility")
        ax.set_title(f"Stock Volatility - {selected_company}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (Rolling 30-Day Std Dev)")
        ax.legend()
        
        st.pyplot(fig)

        #st.markdown("---")

        #  Bollinger Bands (Price + Volatility in One Chart)
        st.subheader("📊 Bollinger Bands (Volatility Indicator)")

        df_selected["SMA_20"] = df_selected["Close"].rolling(window=20).mean()
        df_selected["Upper_Band"] = df_selected["SMA_20"] + (df_selected["Close"].rolling(window=20).std() * 2)
        df_selected["Lower_Band"] = df_selected["SMA_20"] - (df_selected["Close"].rolling(window=20).std() * 2)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_selected["Date"], df_selected["Close"], label="Close Price", color="blue", alpha=0.6)
        ax.plot(df_selected["Date"], df_selected["SMA_20"], label="20-Day SMA", color="orange", linestyle="dashed")
        ax.fill_between(df_selected["Date"], df_selected["Upper_Band"], df_selected["Lower_Band"], color="gray", alpha=0.2, label="Bollinger Bands")
        
        ax.set_title(f"Bollinger Bands - {selected_company}")
        ax.legend()
        st.pyplot(fig)

        # st.markdown("---")

        st.subheader("⚡ Extreme Volatility Events")

        threshold = df_selected["Log Returns"].std() * 2  # 2x standard deviation
        df_selected["Extreme"] = df_selected["Log Returns"].abs() > threshold

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_selected["Date"], df_selected["Log Returns"], label="Log Returns", color="gray", alpha=0.6)
        ax.scatter(df_selected[df_selected["Extreme"]]["Date"], 
                df_selected[df_selected["Extreme"]]["Log Returns"], 
                color="red", label="Extreme Events", zorder=3)

        ax.set_title(f"Extreme Volatility Events - {selected_company}")
        ax.legend()
        st.pyplot(fig)

        st.markdown("💡 *Use filters to analyze different stocks and trends.*")
    with col_vol2:    
        st.subheader("📊 Historical Volatility Heatmap")

        df_selected["Month"] = df_selected["Date"].dt.month
        df_selected["Year"] = df_selected["Date"].dt.year
        volatility_pivot = df_selected.pivot_table(index="Year", columns="Month", values="Volatility", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(volatility_pivot, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5, ax=ax)

        ax.set_title(f"Volatility Heatmap - {selected_company}")
        st.pyplot(fig)

        #st.markdown("---")
        st.subheader("📊 Volatility Clustering")

        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(df_selected["Date"], df_selected["Log Returns"], 
                            c=df_selected["Volatility"], cmap="coolwarm", edgecolors="black", alpha=0.7)

        ax.set_title(f"Volatility Clustering - {selected_company}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Returns")
        plt.colorbar(scatter, label="Volatility Level")
        
        st.pyplot(fig)
        