import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet


import numpy as np

# Streamlit UI Configuration
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# Load Data
file_path = "sorce.xlsx"  # Update with actual file path

@st.cache_data
def load_data():
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df = df[df["Year"] >= 2015]
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    return df

df = load_data()

# Aggregate Data
yearly_trade_value = df.groupby(["Year", "Source_File"])["Total_Trade_Value"].sum().unstack()

# Sidebar Filters
st.sidebar.header("📌 Dashboard Filters")
selected_company = st.sidebar.selectbox("🔍 Select a Company", yearly_trade_value.columns)

# Layout Columns
col1, col2 = st.columns((2, 1))

# Yearly Performance Chart
with col1:
    st.subheader("📊 Yearly Performance")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=yearly_trade_value.index, y=yearly_trade_value[selected_company], color="#4CAF50", ax=ax)
    ax.set_title(f"Total Trade Value per Year - {selected_company}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Year")
    ax.set_ylabel("Total Trade Value ($)")
    st.pyplot(fig)

# Best & Worst Performing Stocks
with col2:
    st.subheader("🏆 Best & Worst Performing Stocks")
    avg_trade = yearly_trade_value.mean().sort_values(ascending=False)
    st.write("**🔥 Best Performer:**", avg_trade.idxmax(), f"(${avg_trade.max():,.2f})")
    st.write("**❄️ Worst Performer:**", avg_trade.idxmin(), f"(${avg_trade.min():,.2f})")

st.markdown("---")

# Correlation Analysis
st.subheader("📉 Correlation Analysis")
st.markdown("Compare trade values between two companies.")
col1, col2 = st.columns(2)

with col1:
    company1 = st.selectbox("Select First Company", yearly_trade_value.columns, key="company1")
with col2:
    company2 = st.selectbox("Select Second Company", yearly_trade_value.columns, key="company2")

fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=yearly_trade_value[company1], y=yearly_trade_value[company2], color="#FF5733", s=100, alpha=0.75)
ax.set_title(f"🔗 Correlation: {company1} vs {company2}")
ax.set_xlabel(f"Total Trade Value - {company1}")
ax.set_ylabel(f"Total Trade Value - {company2}")
st.pyplot(fig)

st.markdown("---")

# Moving Averages & Anomaly Detection
st.subheader("📈 Moving Averages & Anomalies")
st.markdown("Analyze trends and detect anomalies in stock prices.")

df_selected = df[df["Source_File"] == selected_company]
df_selected = df_selected.sort_values("Date")
df_selected["SMA_30"] = df_selected["Close"].rolling(window=30).mean()
df_selected["Anomaly"] = np.abs(df_selected["Close"] - df_selected["SMA_30"]) > 2 * df_selected["Close"].std()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df_selected["Date"], df_selected["Close"], label="Close Price", color="blue")
ax.plot(df_selected["Date"], df_selected["SMA_30"], label="30-Day SMA", color="red")
ax.scatter(df_selected["Date"], df_selected["Close"], c=df_selected["Anomaly"].map({True: "red", False: "blue"}), label="Anomaly")
ax.set_title(f"Stock Price Trend - {selected_company}")
ax.legend()
st.pyplot(fig)

st.markdown("---")

# Heatmap for Correlation
st.subheader("🌍 Heatmap of Correlations")
st.markdown("See how stocks are correlated with each other.")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(yearly_trade_value.corr(), annot=True, cmap="coolwarm", linewidths=0.5, ax=ax)
st.pyplot(fig)

st.markdown("---")

# Stock Prediction using Prophet
st.subheader("🔮 Stock Price Prediction")

df_forecast = df_selected[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
model = Prophet()
model.fit(df_forecast)
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Price", color="green")
ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="green", alpha=0.3)
ax.set_title(f"Predicted Stock Prices - {selected_company}")
ax.legend()
st.pyplot(fig)

st.markdown("---")
st.markdown("💡 *Use filters to analyze different stocks and trends.*")
