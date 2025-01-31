import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit UI Configuration (Must be first Streamlit command)
st.set_page_config(page_title="Stock Market Analysis", layout="wide")

# Load the Excel file
file_path = "sorce.xlsx"  # Update with your file path

@st.cache_data
def load_data():
    df = pd.read_excel(file_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df = df[df["Year"] >= 2015]
    df["Total_Trade_Value"] = df["Close"] * df["Volume"]
    return df

df = load_data()

# Aggregate by Year and Company
yearly_trade_value = df.groupby(["Year", "Source_File"])["Total_Trade_Value"].sum().unstack()

st.title("📈 Yearly Performance of Companies")
st.markdown("---")

# Select Company for Individual Performance
selected_company = st.selectbox("🔍 Select a Company", yearly_trade_value.columns)

# Plot the selected company's yearly trade values
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x=yearly_trade_value.index, y=yearly_trade_value[selected_company], color="#4CAF50", ax=ax)
ax.set_title(f"📊 Total Trade Value per Year (2015-Current) - {selected_company}", fontsize=14, fontweight='bold')
ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Total Trade Value ($)", fontsize=12)
ax.grid(axis="y", linestyle="--", alpha=0.7)
st.pyplot(fig)

st.markdown("---")

# Select Companies for Correlation Analysis
st.subheader("📉 Correlation Analysis")
col1, col2 = st.columns(2)

with col1:
    company1 = st.selectbox("Select First Company", yearly_trade_value.columns, key="company1")
with col2:
    company2 = st.selectbox("Select Second Company", yearly_trade_value.columns, key="company2")

# Plot correlation between selected companies
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=yearly_trade_value[company1], y=yearly_trade_value[company2], color="#FF5733", s=100, alpha=0.75)
ax.set_title(f"🔗 Correlation between {company1} and {company2}", fontsize=14, fontweight='bold')
ax.set_xlabel(f"Total Trade Value - {company1} ($)", fontsize=12)
ax.set_ylabel(f"Total Trade Value - {company2} ($)", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.7)
st.pyplot(fig)

st.markdown("---")
st.markdown("💡 *Use the dropdowns above to explore individual company trends and correlations.*")
