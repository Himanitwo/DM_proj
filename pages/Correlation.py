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
        st.metric(label="Pearson Correlation", value=f"{correlation:.2f}")
        
        # Display Correlation Dataframe
        st.dataframe(merged_df[[company1, company2]].corr())
        
        # Dropdown for graph selection
        graph_type = st.sidebar.selectbox("Select Graph Type", ["Scatter Plot", "Line Chart", "Heatmap"])
        
        # Generate selected graph
        if graph_type == "Scatter Plot":
            fig, ax = plt.subplots()
            sns.scatterplot(x=merged_df[company1], y=merged_df[company2], ax=ax)
            ax.set_xlabel(company1)
            ax.set_ylabel(company2)
            ax.set_title(f"Scatter Plot: {company1} vs {company2}")
            st.pyplot(fig)
        
        elif graph_type == "Line Chart":
            st.line_chart(merged_df.set_index("Date"))
        
        elif graph_type == "Heatmap":
            fig, ax = plt.subplots()
            sns.heatmap(merged_df[[company1, company2]].corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
    else:
        st.warning("Please select two different companies for correlation analysis.")
