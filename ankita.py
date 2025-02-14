import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet


st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
import streamlit as st


#pg.run()
@st.cache_data
def load_data():
    file_path = "sorce.xlsx"
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    df.columns = df.columns.str.strip()
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        st.error("❌ 'Date' column not found in dataset.")
        return None
    
    required_columns = {"Company", "Close", "Volume"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    return df

df = load_data()

if df is not None:
    st.sidebar.header("Dashboard Filters")
    yearly_trade_value = df.groupby([df["Date"].dt.year, "Company"])["Total_Trade_Value"].sum().unstack()
    selected_company = st.sidebar.selectbox("🔍 Select a Company", yearly_trade_value.columns)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(f"Total Trade Value per Year - {selected_company}")
        chart_data = yearly_trade_value[selected_company].reset_index()
        st.bar_chart(chart_data, x="Date", y=selected_company, use_container_width=True)
    
    with col2:
        company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        if len(company_data) > 0:
            model = Prophet()
            model.fit(company_data)
            
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)
            
            forecast_data = forecast[["ds", "yhat"]].set_index("ds")
            actual_data = company_data.set_index("ds")
            combined_data = actual_data.join(forecast_data, how="outer")
            
            st.subheader(f"Stock Price Forecast - {selected_company}")
            st.line_chart(combined_data, use_container_width=True)
    
    with col3:
        latest_data_all = df.groupby("Company").last()
        trade_price_table = latest_data_all[["Close"]].rename(columns={"Close": "Last Trade Price"})
        st.subheader("Latest Trade Prices")
        st.dataframe(trade_price_table, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Details")
        st.write("The prediction model used is Facebook's Prophet, which considers seasonality and trends to forecast future stock prices.")
    
    with col2:
        st.subheader("Prediction Data")
        st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10), use_container_width=True)



#moving average
