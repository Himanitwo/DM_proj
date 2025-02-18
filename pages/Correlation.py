import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data_path = "./sorce.csv"
df = pd.read_csv(data_path)

# Ensure column names are correct
df.columns = df.columns.str.strip()
if "Company" not in df.columns or "Close" not in df.columns or "Date" not in df.columns:
    st.error("Error: Required columns are missing from the dataset.")
else:
    st.title("Competitor Stock Correlation Analysis")
    st.sidebar.header("Select Companies")
    
    # Convert Date column to datetime
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    
    # Get unique companies
    companies = df["Company"].unique()
    company1 = st.sidebar.selectbox("Select First Company", companies)
    company2 = st.sidebar.selectbox("Select Second Company", companies)
    
    if company1 != company2:
        
        selected_year = st.sidebar.selectbox("Select Year", sorted(df["Year"].unique(), reverse=True))
        df = df[df["Year"] == selected_year]
        
        
        df1 = df[df["Company"] == company1][["Date", "Close"]].rename(columns={"Close": company1})
        df2 = df[df["Company"] == company2][["Date", "Close"]].rename(columns={"Close": company2})
        
        
        merged_df = pd.merge(df1, df2, on="Date", how="inner")
        
       
        correlation = merged_df[[company1, company2]].corr().iloc[0, 1]
        st.metric(label="Pearson Correlation", value=f"{correlation:.2f}")
        
       
        st.dataframe(merged_df[[company1, company2]].corr())
        
        
        graph_type = st.sidebar.selectbox("Select Graph Type", ["Scatter Plot", "Line Chart"])
        
        
        if graph_type == "Scatter Plot":
            fig, ax = plt.subplots()
            sns.scatterplot(x=merged_df[company1], y=merged_df[company2], ax=ax)
            ax.set_xlabel(company1)
            ax.set_ylabel(company2)
            ax.set_title(f"Scatter Plot: {company1} vs {company2}")
            st.pyplot(fig)
        
        elif graph_type == "Line Chart":
            st.line_chart(merged_df.set_index("Date"))
        
    else:
        st.warning("Please select two different companies for correlation analysis.")