import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import numpy as np
st.set_page_config(page_title="stock market dashboard")
file_path = "sorce.xlsx"
st.cache_data
def load_data():
    df=pd.read_excel(file_path,sheet_name="Sheet1")
    df.columns=df.columns.str.strip()
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
df= load_data()
if df is not None:
    st.sidebar.header("Dashboard Filters")
    yearly_trade_value = df.groupby(["Year", "Company"])["Total_Trade_Value"].sum().unstack()
    selected_company = st.sidebar.selectbox("🔍 Select a Company", yearly_trade_value.columns)
    col1,col2,col3 =st.columns(3,border=True)
    with col1:
            
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.barplot(x=yearly_trade_value.index, y=yearly_trade_value[selected_company], color="#4CAF50", ax=ax)
            ax.set_title(f"Total Trade Value per Year - {selected_company}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Year")
            ax.set_ylabel("Total Trade Value ($)")
            st.pyplot(fig)
    with col2:
        
        company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        
        if len(company_data) > 0:
            model = Prophet()
            model.fit(company_data)
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(company_data["ds"], company_data["y"], label="Actual Data", color="blue")
            ax.plot(forecast["ds"], forecast["yhat"], label="Predicted Data", color="red")
            ax.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="pink", alpha=0.3)
            ax.set_title(f"Stock Price Forecast - {selected_company}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Date")
            ax.set_ylabel("Stock Price ($)")
            ax.legend()
            st.pyplot(fig)
    with col3:
       
        latest_data = df[df["Company"] == selected_company].groupby("Company").last()
        summary_table = latest_data[["Close", "Total_Trade_Value"]].rename(columns={"Close": "Last Trade Price", "Total_Trade_Value": "Total Trade Value"})
        st.table(summary_table)
    