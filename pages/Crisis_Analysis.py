import streamlit as st
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
st.title("Power BI Report in Streamlit")


powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=d3264b96-ebb6-46d2-988a-e5a199a8c028&autoAuth=true&ctid=accfabe1-9185-4015-b07c-68da5f8f25ad"


st.components.v1.iframe(powerbi_url, width=800,height=600)