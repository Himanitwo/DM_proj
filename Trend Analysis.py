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
import plotly.express as px


import matplotlib.pyplot as plt

from sklearn.metrics import silhouette_score

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
    
    
    required_columns = {"Company", "Close", "Volume", "Open", "High", "Low"}
    if not required_columns.issubset(df.columns):
        st.error(f"❌ Missing columns: {required_columns - set(df.columns)}")
        return None
    

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
    
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=window, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(window=window, min_periods=1).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

@st.cache_data
def compute_technical_indicators(df):
    
    companies = df["Company"].unique()
    tech_list = []
    
    for comp in companies:
        df_comp = df[df["Company"] == comp].copy().sort_values("Date")
        
        df_comp["RSI"] = calculate_rsi(df_comp["Close"])
        
        latest_rsi = df_comp["RSI"].iloc[-1]
        
        
        df_comp["Pct_Change"] = df_comp["Close"].pct_change()
        volatility = df_comp["Pct_Change"].std()
        
        tech_list.append({"Company": comp, "RSI": latest_rsi, "Volatility": volatility})
    
    tech_df = pd.DataFrame(tech_list)
    return tech_df

if df is not None:
    st.sidebar.header("Dashboard Filters")
    selected_company = st.sidebar.selectbox("🔍 Select a Company", df["Company"].unique())

    
    # -------------------------
    # Trend Analysis with Linear Regression & Suggestion Box
    # -------------------------
    selected_year = st.selectbox("Select Year for Trend Analysis", sorted(df["Year"].dropna().unique(), reverse=True))
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
            st.subheader(f"Stock Price Trends - {selected_company} ({selected_year})")
            trend_chart_data = df_filtered.set_index("Date")[["Close", "7-day MA", "30-day MA", "Trend"]]
            st.line_chart(trend_chart_data)
    with col_tab:
      
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
 
    
    st.subheader("K-Means Clustering for Trend Analysis")

    
    selected_year_trend = st.selectbox(
        "Select Year for Trend Analysis", 
        sorted(df["Year"].dropna().unique(), reverse=True), 
        key="trend_year"
    )

   
    df_trend_year = df[df["Year"] == selected_year_trend].copy()

    # Ensure the Date column is in datetime format and sort the data by Date.
    if not np.issubdtype(df_trend_year['Date'].dtype, np.datetime64):
        df_trend_year['Date'] = pd.to_datetime(df_trend_year['Date'])
    df_trend_year = df_trend_year.sort_values("Date")

    # calclulate For each company trend features:
    #    - Cumulative Return: Overall return from the first to last closing price.
    #    - Avg Daily Return: Mean of daily percentage returns.
    #    - Volatility: Standard deviation of daily percentage returns.'''

    companies = df_trend_year["Company"].unique()
    trend_features = []

    for comp in companies:
        comp_df = df_trend_year[df_trend_year["Company"] == comp].copy()
        comp_df = comp_df.sort_values("Date")
        if comp_df.empty:
            continue
        
        first_close = comp_df["Close"].iloc[0]
        last_close = comp_df["Close"].iloc[-1]
        cumulative_return = (last_close - first_close) / first_close
        
        daily_returns = comp_df["Close"].pct_change()
        avg_daily_return = daily_returns.mean()
        volatility = daily_returns.std()
        
        trend_features.append({
            "Company": comp,
            "Cumulative_Return": cumulative_return,
            "Avg_Daily_Return": avg_daily_return,
            "Volatility": volatility
        })

    trend_df = pd.DataFrame(trend_features)

 
    features = trend_df[["Cumulative_Return", "Avg_Daily_Return", "Volatility"]].fillna(0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    kmeans_trend = KMeans(n_clusters=3, random_state=42, n_init=10)
    trend_clusters = kmeans_trend.fit_predict(features_scaled)
    trend_df["Cluster"] = trend_clusters

  
    fig = px.scatter(
        trend_df, 
        x="Cumulative_Return", 
        y="Volatility", 
        color="Cluster",
        hover_data=["Company", "Avg_Daily_Return"],
        title=f"Trend Analysis Clusters for {selected_year_trend}"
    )
    st.plotly_chart(fig, use_container_width=True)

    
    st.subheader("Trend Features and Cluster Assignments")
    st.dataframe(trend_df.sort_values("Cluster"))
    # -------------------------
    # Cumulative Returns Chart
    # -------------------------
    # st.subheader("Cumulative Returns")
    
    

    # st.line_chart(df_selected["Cumulative_Return"])

# -------------------------
# Volume-Price Relationship Analysis (Dual-Axis Graph)
# -------------------------

    

    st.subheader("Volume-Price Relationship (Bar Chart)")

    # Dropdown for the number of years
    years = st.selectbox("Select number of years:", [3, 4, 5], index=2)  # Default is 5 years
    df_selected = df[df["Company"] == selected_company].copy().sort_values("Date")
    df_selected.set_index("Date", inplace=True)
    # Filter data based on selected years (note:assume df_selected has a DatetimeIndex)
    df_filtered = df_selected[df_selected.index >= pd.Timestamp.today() - pd.DateOffset(years=years)].copy()

    # Calculate daily price change and 20-day ma for volume
    df_filtered['Price_Change'] = df_filtered['Close'].diff()
    df_filtered['Volume_MA20'] = df_filtered['Volume'].rolling(window=20).mean()

    # function to classify each day based on price change and volume
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

    
    df_filtered['Trend'] = df_filtered.apply(classify_day, axis=1)
    df_filtered = df_filtered.dropna(subset=['Trend'])

   
    color_map = {
        'High Volume Up': 'green',
        'Low Volume Up': 'lightgreen',
        'High Volume Down': 'red',
        'Low Volume Down': 'orange'
    }

   
    fig = px.bar(
        df_filtered,
        x=df_filtered.index,
        y="Volume",
        color="Trend",
        color_discrete_map=color_map,
        title=f'Volume-Price Relationship for {selected_company} (Last {years} Years)',
        labels={'Volume': 'Trading Volume', 'Trend': 'Market Trend'}
    )

   
    col1, col2 = st.columns([3, 1])  

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
    st.subheader("Trend indicators")
    df["SMA_20"] = df["Close"].rolling(window=20).mean()


    df["Price Change (%)"] = (df["Close"] - df["Close"].shift(30)) / df["Close"].shift(30) * 100

    

    latest_sma = df["SMA_20"].iloc[-1]  
    latest_price = df["Close"].iloc[-1]  

    if latest_price > latest_sma * 1.02: 
        trend = "📈 Upward"
    elif latest_price < latest_sma * 0.98:  
        trend = "📉 Downward"
    else:
        trend = "Sideways"

  

    if latest_price > latest_sma * 1.05:  # If price is 5%(test) above SMA
        breakout_alert = "Strong Uptrend (Breakout Above SMA)"
    elif latest_price < latest_sma * 0.95:  # If price is 5%(test) below SMA
        breakout_alert = "Potential Downtrend (Breakout Below SMA)"
    else:
        breakout_alert = "No Significant Breakout"


    trend_data = {
    "Metric": ["20-Day SMA", "Price Change (%)", "Overall Trend", "Breakout Alert"],
    "Value": [f"{latest_sma:.2f}", f"{df['Price Change (%)'].iloc[-1]:.2f}%", trend, breakout_alert]
    }

    trend_df = pd.DataFrame(trend_data)
    st.table(trend_df)
