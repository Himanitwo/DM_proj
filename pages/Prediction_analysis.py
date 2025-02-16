import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Set up the Streamlit page
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

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

def calculate_rsi(series, window=14):
    """Calculate Relative Strength Index (RSI) for a given series."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def compute_technical_indicators(df):
    """Compute technical indicators (RSI and Volatility) for each company."""
    companies = df["Company"].unique()
    tech_list = []
    
    for comp in companies:
        df_comp = df[df["Company"] == comp].copy().sort_values("Date")
        # Calculate RSI on closing prices
        df_comp["RSI"] = calculate_rsi(df_comp["Close"])
        # Use the most recent RSI value (or an average over a recent period)
        latest_rsi = df_comp["RSI"].iloc[-1]
        
        # Compute daily percentage change and volatility (std dev of % changes)
        df_comp["Pct_Change"] = df_comp["Close"].pct_change()
        volatility = df_comp["Pct_Change"].std()
        
        tech_list.append({"Company": comp, "RSI": latest_rsi, "Volatility": volatility})
    
    tech_df = pd.DataFrame(tech_list)
    return tech_df

if df is not None:
    st.sidebar.header("Dashboard Filters")
    selected_company = st.sidebar.selectbox("🔍 Select a Company", df["Company"].unique())

    # -------------------------
    # Forecasting & Latest Price
    # -------------------------
    col_forecast, col_prices = st.columns(2)
    
    with col_forecast:
        company_data = df[df["Company"] == selected_company][["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
        if not company_data.empty:
            company_data["7-day MA"] = company_data["y"].rolling(window=7).mean()
            company_data["30-day MA"] = company_data["y"].rolling(window=30).mean()
            model_prophet = Prophet()
            model_prophet.fit(company_data[["ds", "y"]])
            future = model_prophet.make_future_dataframe(periods=365)
            forecast = model_prophet.predict(future)
            forecast_data = forecast[["ds", "yhat"]].set_index("ds")
            actual_data = company_data.set_index("ds")
            combined_data = actual_data.join(forecast_data, how="outer")
            combined_data = combined_data.apply(pd.to_numeric, errors='coerce')
            combined_data = combined_data.select_dtypes(include=[np.number])
            st.subheader(f"📈 Stock Price Forecast - {selected_company}")
            st.line_chart(combined_data)
    
    with col_prices:
        latest_data_all = df.groupby("Company").last()
        trade_price_table = latest_data_all[["Close"]].rename(columns={"Close": "Last Trade Price"})
        st.subheader("💰 Latest Trade Prices")
        st.dataframe(trade_price_table, use_container_width=True)
    
    # -------------------------
    # Trend Analysis with Linear Regression & Suggestion Box
    # -------------------------
    selected_year = st.selectbox("📅 Select Year for Trend Analysis", sorted(df["Year"].dropna().unique(), reverse=True))
    df_filtered = df[(df["Company"] == selected_company) & (df["Year"] == selected_year)].copy()
    col_gr,col_tab=st.columns(2)
    with col_gr:
        if df_filtered.empty:
            st.warning(f"No data available for {selected_company} in {selected_year}")
        else:
            df_filtered['7-day MA'] = df_filtered['Close'].rolling(window=7, min_periods=1).mean()
            df_filtered['30-day MA'] = df_filtered['Close'].rolling(window=30, min_periods=1).mean()
            df_filtered['Date_Num'] = (df_filtered['Date'] - df_filtered['Date'].min()).dt.days

            X = df_filtered[['Date_Num']]
            y = df_filtered['Close']
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            df_filtered['Trend'] = lr_model.predict(X)
            # Use st.line_chart for an interactive trend graph.
            st.subheader(f"📊 Stock Price Trends - {selected_company} ({selected_year})")
            trend_chart_data = df_filtered.set_index("Date")[["Close", "7-day MA", "30-day MA", "Trend"]]
            st.line_chart(trend_chart_data)
    with col_tab:
        '''Determine trend based on slope'''
        slope = lr_model.coef_[0]
        if slope > 0:
            trend_label = "📈 Uptrend"
            suggestion_trend = "Consider buying or holding for further growth."
        elif slope < 0:
            trend_label = "📉 Downtrend"
            suggestion_trend = "Consider selling or using caution."
        else:
            trend_label = "➖ Sideways Movement"
            suggestion_trend = "Market is stable; consider holding."

        st.subheader("Trend Analysis & Suggestion")
        st.info(f"**Trend:** {trend_label}\n\n**Suggestion:** {suggestion_trend}")
        
        
    
    # -------------------------
    # K-Means Clustering for Risk Analysis (Filtered by Year)
    # -------------------------
    selected_year_cluster = st.selectbox(
        "Select Year for Clustering", 
        sorted(df["Year"].dropna().unique(), reverse=True),
        key="clustering_year"
    )
    df_year = df[df["Year"] == selected_year_cluster].copy()

    # Create pivot table with average closing prices for the selected year
    df_cluster = df_year.groupby(["Date", "Company"], as_index=False).agg({"Close": "mean"})
    df_pivot = df_cluster.pivot(index='Date', columns='Company', values='Close')

    # Compute daily percentage change, drop initial NaNs, and forward-fill
    df_pivot = df_pivot.pct_change().dropna().fillna(method="ffill")

    # Compute volatility for each company (std dev of daily % changes)
    volatility = df_pivot.std()

    # Standardize data for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_pivot.T)

    # Run K-Means clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    cluster_df = pd.DataFrame({'Company': df_pivot.columns, 'Cluster': cluster_labels})
    cluster_df["X"] = X_scaled[:, 0]
    cluster_df["Y"] = X_scaled[:, 1]

    # Merge volatility into the clustering DataFrame
    cluster_df["Volatility"] = cluster_df["Company"].map(volatility)

    # For risk analysis, interpret higher volatility as higher risk.
    cluster_risk = cluster_df.groupby("Cluster")["Volatility"].mean().reset_index()
    cluster_risk = cluster_risk.sort_values("Volatility")
    cluster_risk["Risk_Rank"] = range(1, len(cluster_risk) + 1)
    risk_mapping = {1: "Low Risk", 2: "Moderate Risk", 3: "High Risk"}
    cluster_risk["Risk_Level"] = cluster_risk["Risk_Rank"].map(risk_mapping)

    cluster_df = cluster_df.merge(cluster_risk[["Cluster", "Risk_Level"]], on="Cluster", how="left")

    col_cluster, col_cluster_details = st.columns((2, 1))
    with col_cluster:
        st.subheader("📍 Stock Clusters Visualization (Filtered by Year)")
        st.scatter_chart(cluster_df, x="X", y="Y", color="Cluster", use_container_width=True)
    with col_cluster_details:
        st.subheader("Cluster & Risk Details")
        st.dataframe(cluster_df.sort_values(by="Cluster"))
        company_cluster = cluster_df[cluster_df["Company"] == selected_company]
        if not company_cluster.empty:
            risk_level = company_cluster["Risk_Level"].values[0]
            if risk_level == "Low Risk":
                suggestion_risk = "This stock shows low volatility. It may be a stable investment."
            elif risk_level == "Moderate Risk":
                suggestion_risk = "This stock shows moderate volatility. Consider balancing with other assets."
            else:
                suggestion_risk = "This stock shows high volatility. Exercise caution and consider your risk tolerance."
            st.info(f"**Risk Level for {selected_company}:** {risk_level}\n\n**Suggestion:** {suggestion_risk}")
    
    # -------------------------
    # Cumulative Returns Chart (Using st.line_chart)
    # -------------------------
    # st.subheader("📈 Cumulative Returns")
    df_selected = df[df["Company"] == selected_company].copy().sort_values("Date")
    df_selected.set_index("Date", inplace=True)
    df_selected["Daily_Return"] = df_selected["Close"].pct_change()
    # Calculate cumulative returns; starting from a base value of 100
    df_selected["Cumulative_Return"] = (1 + df_selected["Daily_Return"]).cumprod() * 100

    # st.line_chart(df_selected["Cumulative_Return"])

# -------------------------
# Volume-Price Relationship Analysis (Dual-Axis Graph)
# -------------------------

    import plotly.express as px

    st.subheader("📊 Volume-Price Relationship (Bar Chart)")

    # Dropdown for selecting the number of years
    years = st.selectbox("Select number of years:", [3, 4, 5], index=2)  # Default is 5 years

    # Filter data based on selected years (assumes df_selected has a DatetimeIndex)
    df_filtered = df_selected[df_selected.index >= pd.Timestamp.today() - pd.DateOffset(years=years)].copy()

    # Calculate daily price change and 20-day moving average for volume
    df_filtered['Price_Change'] = df_filtered['Close'].diff()
    df_filtered['Volume_MA20'] = df_filtered['Volume'].rolling(window=20).mean()

    # Define a function to classify each day based on price change and volume
    def classify_day(row):
        if pd.isna(row['Volume_MA20']):
            return np.nan  # Not enough data to classify
        if row['Price_Change'] > 0 and row['Volume'] >= row['Volume_MA20']:
            return 'High Volume Up'
        elif row['Price_Change'] > 0 and row['Volume'] < row['Volume_MA20']:
            return 'Low Volume Up'
        elif row['Price_Change'] < 0 and row['Volume'] >= row['Volume_MA20']:
            return 'High Volume Down'
        elif row['Price_Change'] < 0 and row['Volume'] < row['Volume_MA20']:
            return 'Low Volume Down'
        else:
            return 'No Change'

    # Apply classification
    df_filtered['Trend'] = df_filtered.apply(classify_day, axis=1)
    df_filtered = df_filtered.dropna(subset=['Trend'])

    # Define color mapping for the classifications
    color_map = {
        'High Volume Up': 'green',
        'Low Volume Up': 'lightgreen',
        'High Volume Down': 'red',
        'Low Volume Down': 'orange'
    }

    # Create a bar chart using Plotly Express
    fig = px.bar(
        df_filtered,
        x=df_filtered.index,
        y="Volume",
        color="Trend",
        color_discrete_map=color_map,
        title=f'Volume-Price Relationship for {selected_company} (Last {years} Years)',
        labels={'Volume': 'Trading Volume', 'Trend': 'Market Trend'}
    )

    # Create two columns: one for the graph and one for insights
    col1, col2 = st.columns([3, 1])  # Wider column for the graph

    with col1:
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Insights")
        st.markdown(
            """
            **High Volume Up:**  
            - Price increased with high volume  
            - Indicates strong bullish momentum  
            
            **Low Volume Up:**  
            - Price increased with low volume  
            - Suggests weak upward movement or potential reversal  
            
            **High Volume Down:**  
            - Price decreased with high volume  
            - Indicates strong bearish sentiment  
            
            **Low Volume Down:**  
            - Price decreased with low volume  
            - May signal a weak downtrend or a possible reversal  
            """
        )

    


    
    #here
    

    # ---------------------------------------------------------------------
    # Create a single area chart with the closing price, SMA, and manually calculated EMA
    # ---------------------------------------------------------------------
    fig = go.Figure()

    # Closing Price as an area chart
    fig.add_trace(go.Scatter(
        x=df_year.index,
        y=df_year['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue'),
        fill='tozeroy',
        opacity=0.5
    ))

    # 20-Day SMA Line
    fig.add_trace(go.Scatter(
        x=df_year.index,
        y=df_year['SMA_20'],
        mode='lines',
        name='50-Day SMA',
        line=dict(color='green', width=2)
    ))

    # 20-Day EMA Line (Manual Calculation)
    fig.add_trace(go.Scatter(
        x=df_year.index,
        y=df_year['EMA_20'],
        mode='lines',
        name='20-Day EMA',
        line=dict(color='red', width=2)
    ))

    fig.update_layout(
        title=f"20-Day SMA & EMA for {selected_company} in {selected_year}",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    


    col_vol1,col_vol2=st.columns(2)
# Layout: Yearly Performance & Best/Worst Stocks
    st.markdown("## 📊 Stock Market Dashboard")
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

        #st.markdown("---")

        

