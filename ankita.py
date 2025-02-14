import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet

# 📌 Load Data with Duplicate Handling
@st.cache_data
def load_data():
    file_path = "sorce.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove whitespace from column names
    
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        st.error("❌ 'Date' column not found in dataset.")
        return None
    
    required_columns = {"Company", "Close", "Volume"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    
    # Handle duplicate entries by averaging 'Close' prices
    df = df.groupby(["Date", "Company"], as_index=False).agg({"Close": "mean", "Volume": "sum"})
    
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    df["Year"] = df["Date"].dt.year
    return df

df = load_data()

if df is not None:
    st.sidebar.header("Dashboard Filters")
    selected_company = st.sidebar.selectbox("🔍 Select a Company", df["Company"].unique())

    col1, col2 = st.columns(2)
    
    with col1:
        company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

        if not company_data.empty:
            company_data["7-day MA"] = company_data["y"].rolling(window=7).mean()
            company_data["30-day MA"] = company_data["y"].rolling(window=30).mean()

            model = Prophet()
            model.fit(company_data[["ds", "y"]])
            
            future = model.make_future_dataframe(periods=365)
            forecast = model.predict(future)

            forecast_data = forecast[["ds", "yhat"]].set_index("ds")
            actual_data = company_data.set_index("ds")
            combined_data = actual_data.join(forecast_data, how="outer")

            # Ensure only numeric columns are plotted
            combined_data = combined_data.apply(pd.to_numeric, errors='coerce')
            combined_data = combined_data.select_dtypes(include=[np.number])

            st.subheader(f"📈 Stock Price Forecast - {selected_company}")
            st.line_chart(combined_data, use_container_width=True)
    
    with col2:
        latest_data_all = df.groupby("Company").last()
        trade_price_table = latest_data_all[["Close"]].rename(columns={"Close": "Last Trade Price"})
        st.subheader("💰 Latest Trade Prices")
        st.dataframe(trade_price_table, use_container_width=True)
    
    # 📌 Trend Analysis & Clustering
    selected_year = st.selectbox("📅 Select Year", sorted(df["Year"].dropna().unique(), reverse=True))
    df_filtered = df[(df["Company"] == selected_company) & (df["Year"] == selected_year)].copy()
    
    if df_filtered.empty:
        st.warning(f"No data available for {selected_company} in {selected_year}")
    else:
        df_filtered['7-day MA'] = df_filtered['Close'].rolling(window=7, min_periods=1).mean()
        df_filtered['30-day MA'] = df_filtered['Close'].rolling(window=30, min_periods=1).mean()
        df_filtered['Date_Num'] = (df_filtered['Date'] - df_filtered['Date'].min()).dt.days

        X = df_filtered[['Date_Num']]
        y = df_filtered['Close']
        model = LinearRegression()
        model.fit(X, y)
        df_filtered['Trend'] = model.predict(X)

        st.subheader(f"📊 Stock Price Trends - {selected_company} ({selected_year})")
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

    # 📌 K-Means Clustering with Proper Data Handling
    st.subheader("📊 Stock Movement Clustering")

    # Handle duplicates and pivot properly
    df_cluster = df.groupby(["Date", "Company"], as_index=False).agg({"Close": "mean"})
    df_pivot = df_cluster.pivot(index='Date', columns='Company', values='Close')

    # Handle missing values by forward-filling
    df_pivot = df_pivot.pct_change().dropna().fillna(method="ffill")

    # Standardize data for K-Means
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pivot.T)

    # Apply K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    cluster_df = pd.DataFrame({'Company': df_pivot.columns, 'Cluster': cluster_labels})

    st.write("📌 **Company Clusters:**")
    st.dataframe(cluster_df.sort_values(by="Cluster"))

    # 📌 Interactive K-Means Scatter Plot in Streamlit
    cluster_df["X"] = X_scaled[:, 0]
    cluster_df["Y"] = X_scaled[:, 1]

    st.subheader("📍 Stock Clusters Visualization")
    st.scatter_chart(cluster_df, x="X", y="Y", color="Cluster", size=None, use_container_width=True)

    # Annotate Companies
    # fig, ax = plt.subplots(figsize=(10, 5))
    # scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
    # for i, company in enumerate(df_pivot.columns):
    #     ax.annotate(company, (X_scaled[i, 0], X_scaled[i, 1]))
    # ax.set_xlabel("Feature 1")
    # ax.set_ylabel("Feature 2")
    # ax.set_title("Stock Movement Clusters")
    # st.pyplot(fig)
