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
    df = pd.read_excel(file_path, sheet_name="Sheet1")

    # Standardize Column Names
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
    
    # Display column names for debugging
    st.write("Columns in dataset:", df.columns.tolist())

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
        col1, col2 = st.columns((2, 1))

        with col1:
            st.subheader("📊 Yearly Performance")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=yearly_trade_value.index, y=yearly_trade_value[selected_company], color="#4CAF50", ax=ax)
            ax.set_title(f"Total Trade Value per Year - {selected_company}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Trade Value ($)")
            st.pyplot(fig)

        with col2:
            st.subheader("🏆 Best & Worst Performing Stocks")
            avg_trade = yearly_trade_value.mean().sort_values(ascending=False)
            st.write("**🔥 Best Performer:**", avg_trade.idxmax(), f"(${avg_trade.max():,.2f})")
            st.write("**❄️ Worst Performer:**", avg_trade.idxmin(), f"(${avg_trade.min():,.2f})")

        st.markdown("---")

        # Correlation Analysis Section
        st.subheader("📉 Correlation Analysis")
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
        df_selected = df[df["Company"] == selected_company].sort_values("Date")
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

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Price", color="green")
        ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="green", alpha=0.3)
        ax.set_title(f"Predicted Stock Prices - {selected_company}")
        ax.legend()
        st.pyplot(fig)

        st.markdown("---")
        st.markdown("💡 *Use filters to analyze different stocks and trends.*")
