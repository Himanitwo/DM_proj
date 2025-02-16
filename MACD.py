import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("sorce.csv")
    return df

def calculate_macg(df, short_window=12, long_window=26, signal_window=9):
    df['Short_EMA'] = df['Close'].ewm(span=short_window, adjust=False).mean()
    df['Long_EMA'] = df['Close'].ewm(span=long_window, adjust=False).mean()
    df['MACG'] = df['Short_EMA'] - df['Long_EMA']
    df['Signal_Line'] = df['MACG'].ewm(span=signal_window, adjust=False).mean()
    return df

st.title("Stock Data Dashboard")

df = load_data()
df = calculate_macg(df)

st.subheader("Data Preview")
st.write(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

st.subheader("Filter Data")
column = st.selectbox("Select Column", df.columns)
value = st.text_input("Enter Value")
if value:
    filtered_df = df[df[column].astype(str).str.contains(value, case=False, na=False)]
    st.write(filtered_df)

st.subheader("MACG Visualization")
fig, ax = plt.subplots()
ax.plot(df.index, df['MACG'], label='MACG', color='blue')
ax.plot(df.index, df['Signal_Line'], label='Signal Line', color='red')
ax.set_title("MACG and Signal Line")
ax.set_xlabel("Index")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)
