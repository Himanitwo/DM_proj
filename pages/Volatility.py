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
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
    else:
        st.error("❌ 'Date' column not found in dataset.")
    
    required_columns = {"Company", "Close", "Volume"}
    if not required_columns.issubset(set(df.columns)):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    df = df[df["Year"] >= 2015]
    return df

df = load_data()

if df is not None:
    if "Year" in df.columns and "Company" in df.columns:
        yearly_trade_value = df.groupby(["Year", "Company"])["Total_Trade_Value"].sum().unstack()
    else:
        st.error("❌ The dataset does not contain 'Year' or 'Company'. Please check the column names.")
        yearly_trade_value = None

    # Sidebar Filters
    st.sidebar.header("📌 Dashboard Filters")
    
    if yearly_trade_value is not None:
        selected_company = st.sidebar.selectbox("🔍 Select a Company", yearly_trade_value.columns)
        st.markdown("## Volatility Analysis")
        
        
        df_selected = df[df["Company"] == selected_company].sort_values("Date")
        df_selected["Log Returns"] = np.log(df_selected["Close"] / df_selected["Close"].shift(1))
        df_selected["Volatility"] = df_selected["Log Returns"].rolling(window=30).std()
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_selected["Date"], df_selected["Volatility"], label="30-Day Rolling Volatility", color="purple", linewidth=2)
        ax.axhline(df_selected["Volatility"].mean(), color="red", linestyle="dashed", label="Avg Volatility")
        ax.set_title(f"Stock Volatility - {selected_company}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Volatility (Rolling 30-Day Std Dev)")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("💡 **Insight:** High spikes indicate increased market uncertainty, while lower periods suggest stability.")
        st.markdown("---")
        st.subheader("Bollinger Bands (Volatility Indicator)")
        df_selected["SMA_20"] = df_selected["Close"].rolling(window=20).mean()
        df_selected["Upper_Band"] = df_selected["SMA_20"] + (df_selected["Close"].rolling(window=20).std() * 2)
        df_selected["Lower_Band"] = df_selected["SMA_20"] - (df_selected["Close"].rolling(window=20).std() * 2)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df_selected["Date"], df_selected["Close"], label="Close Price", color="blue", alpha=0.6)
        ax.plot(df_selected["Date"], df_selected["SMA_20"], label="20-Day SMA", color="orange", linestyle="dashed")
        ax.fill_between(df_selected["Date"], df_selected["Upper_Band"], df_selected["Lower_Band"], color="gray", alpha=0.2, label="Bollinger Bands")
        ax.set_title(f"Bollinger Bands - {selected_company}")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("💡 **Insight:** Prices near the upper band may indicate overbought conditions, while the lower band suggests oversold conditions.")
        st.markdown("---")

        st.subheader("Extreme Volatility Events")
        threshold = df_selected["Log Returns"].std() * 2
        df_selected["Extreme"] = df_selected["Log Returns"].abs() > threshold


        ax.plot(df_selected["Date"], df_selected["Log Returns"], label="Log Returns", color="gray", alpha=0.6)
        ax.scatter(df_selected[df_selected["Extreme"]]["Date"], 
                   df_selected[df_selected["Extreme"]]["Log Returns"], 
                   color="red", label="Extreme Events", zorder=3)
        ax.set_title(f"Extreme Volatility Events - {selected_company}")
        ax.legend()
        st.pyplot(fig)
        
        st.markdown("💡 **Insight:** Red points highlight extreme price movements, often linked to major market events.")
        st.markdown("---")

        st.subheader("Historical Volatility Heatmap")
        df_selected["Month"] = df_selected["Date"].dt.month
        df_selected["Year"] = df_selected["Date"].dt.year
        volatility_pivot = df_selected.pivot_table(index="Year", columns="Month", values="Volatility", aggfunc="mean")

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(volatility_pivot, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.5, ax=ax)

        # fig, ax = plt.subplots(figsize=(7, 4))  # Reduce heatmap size
        # sns.heatmap(volatility_pivot, cmap="coolwarm", annot=True, fmt=".4f", linewidths=0.3, ax=ax)

        ax.set_title(f"Volatility Heatmap - {selected_company}")
        st.pyplot(fig)
        
        st.markdown("💡 **Insight:** Red months indicate periods of high volatility, which may align with earnings seasons or economic events.")
        st.markdown("---")

        st.subheader("Volatility Clustering")
        fig, ax = plt.subplots(figsize=(12, 6))
        scatter = ax.scatter(df_selected["Date"], df_selected["Log Returns"], 
                            c=df_selected["Volatility"], cmap="coolwarm", edgecolors="black", alpha=0.7)
        ax.set_title(f"Volatility Clustering - {selected_company}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Log Returns")
        plt.colorbar(scatter, label="Volatility Level")
        st.pyplot(fig)
        
        st.markdown("💡 **Insight:** High-volatility days tend to cluster together, indicating prolonged periods of market uncertainty.")
        st.markdown("---")
        