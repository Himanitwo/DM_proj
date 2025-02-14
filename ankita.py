import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# Page Configuration


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
            company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

            if not company_data.empty:
                # Moving Average
                company_data["7-day MA"] = company_data["y"].rolling(window=7).mean()
                company_data["30-day MA"] = company_data["y"].rolling(window=30).mean()

                # Prophet Model
                model = Prophet()
                model.fit(company_data[["ds", "y"]])

                future = model.make_future_dataframe(periods=365)
                forecast = model.predict(future)

                # Merge actual & forecasted data
                forecast_data = forecast[["ds", "yhat"]].set_index("ds")
                actual_data = company_data.set_index("ds")
                combined_data = actual_data.join(forecast_data, how="outer")

                st.subheader(f"Stock Price Forecast - {selected_company}")
                st.line_chart(combined_data[["y", "7-day MA", "30-day MA", "yhat"]], use_container_width=True)

        # **Latest Stock Prices**
with col2:
            latest_data_all = df.groupby("Company").last()
            trade_price_table = latest_data_all[["Close"]].rename(columns={"Close": "Last Trade Price"})
            st.subheader("Latest Trade Prices")
            st.dataframe(trade_price_table, use_container_width=True)

    # **PAGE 2: TREND ANALYSIS & CLUSTERING**

        # **Select Company & Year**
            selected_company = st.sidebar.selectbox("🔍 Select a Company", df["Company"].unique())
            selected_year = st.sidebar.selectbox("📅 Select Year", sorted(df["Year"].unique(), reverse=True))

            # Filter data for company & year
            df_filtered = df[(df["Company"] == selected_company) & (df["Year"] == selected_year)].copy()

            if df_filtered.empty:
                st.warning(f"No data available for {selected_company} in {selected_year}")
            else:
                # **Moving Average & Trend Line**
                df_filtered['7-day MA'] = df_filtered['Close'].rolling(window=7, min_periods=1).mean()
                df_filtered['30-day MA'] = df_filtered['Close'].rolling(window=30, min_periods=1).mean()

                # Convert Date to Numeric for Regression
                df_filtered['Date_Num'] = (df_filtered['Date'] - df_filtered['Date'].min()).dt.days

                # Train Linear Regression Model
                X = df_filtered[['Date_Num']]
                y = df_filtered['Close']
                model = LinearRegression()
                model.fit(X, y)
                df_filtered['Trend'] = model.predict(X)

                # **Plot Moving Averages & Trend**
                st.subheader(f"Stock Price Trends - {selected_company} ({selected_year})")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_filtered['Date'], df_filtered['Close'], label="Close Price", color='blue')
                ax.plot(df_filtered['Date'], df_filtered['7-day MA'], label="7-day MA", linestyle='dashed', color='orange')
                ax.plot(df_filtered['Date'], df_filtered['30-day MA'], label="30-day MA", linestyle='dotted', color='green')
                ax.plot(df_filtered['Date'], df_filtered['Trend'], label="Trend (Linear Regression)", color='red')
                ax.set_xlabel("Date")
                ax.set_ylabel("Stock Price")
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

            # **K-Means Clustering on Stock Returns**
                st.subheader("📊 Stock Movement Clustering")
                
                # Pivot Data for Clustering
                df_pivot = df.pivot(index='Date', columns='Company', values='Close').pct_change().dropna()
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(df_pivot.T)

                # Apply K-Means Clustering (3 clusters)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(X_scaled)

                # Assign Cluster Labels
                cluster_df = pd.DataFrame({'Company': df_pivot.columns, 'Cluster': cluster_labels})

                # Display Cluster Table
                st.dataframe(cluster_df.sort_values(by="Cluster"))

                # **Cluster Scatter Plot**
                fig, ax = plt.subplots(figsize=(10, 5))
                scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
                for i, company in enumerate(df_pivot.columns):
                    ax.annotate(company, (X_scaled[i, 0], X_scaled[i, 1]))
                ax.set_xlabel("Feature 1")
                ax.set_ylabel("Feature 2")
                ax.set_title("Stock Movement Clusters")
                st.pyplot(fig)




