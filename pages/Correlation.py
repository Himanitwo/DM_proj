import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data_path = "./sorce.csv"
df = pd.read_csv(data_path)

# Ensure column names are correct
df.columns = df.columns.str.strip()
if "Company" not in df.columns or "Close" not in df.columns:
    st.error("Error: Required columns are missing from the dataset.")
else:
    st.title("Competitor Stock Correlation Analysis")
    st.sidebar.header("Select Companies")
    
    # Get unique companies
    companies = df["Company"].unique()
    company1 = st.sidebar.selectbox("Select First Company", companies)
    company2 = st.sidebar.selectbox("Select Second Company", companies)
    
    if company1 != company2:
        # Filter data for selected companies
        df1 = df[df["Company"] == company1][["Date", "Close"]].rename(columns={"Close": company1})
        df2 = df[df["Company"] == company2][["Date", "Close"]].rename(columns={"Close": company2})
        
        # Merge data on Date
        merged_df = pd.merge(df1, df2, on="Date", how="inner")
        
        # Compute Pearson Correlation
        correlation = merged_df[[company1, company2]].corr().iloc[0, 1]
        st.write(f"### Pearson Correlation: {correlation:.2f}")
        
        # Display Heatmap
        fig, ax = plt.subplots()
        sns.heatmap(merged_df[[company1, company2]].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Please select two different companies for correlation analysis.")
