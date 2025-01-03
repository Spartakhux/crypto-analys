
import pandas as pd
import numpy as np
import talib
import requests
import streamlit as st
import matplotlib.pyplot as plt

# Function to fetch cryptocurrency data from an API (CoinGecko example)
def fetch_crypto_data(crypto_symbol, currency="usd", days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data['prices'], columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Volume'] = [v[1] for v in data['total_volumes']]
    return df

# Function to calculate indicators
def calculate_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['Signal_Line'], _ = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    upper, middle, lower = talib.BBANDS(df['Close'], timeperiod=20)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = upper, middle, lower
    return df

# Streamlit application
st.title("Cryptocurrency Analysis Tool")

# User selects cryptocurrency and duration
crypto_symbol = st.selectbox(
    "Select Cryptocurrency:",
    ["bitcoin", "ethereum", "binancecoin", "ripple", "cardano"]
)
days = st.slider("Select duration (days):", min_value=7, max_value=90, value=30)

if st.button("Analyze"):
    with st.spinner("Fetching data and performing analysis..."):
        # Fetch data
        data = fetch_crypto_data(crypto_symbol, days=days)
        data = calculate_indicators(data)

        # Display data
        st.subheader("Recent Data")
        st.write(data.tail(10))

        # Plot price and indicators
        st.subheader("Price and Indicators")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Price', color='blue')
        plt.plot(data['Date'], data['SMA_10'], label='SMA 10', color='orange')
        plt.plot(data['Date'], data['SMA_20'], label='SMA 20', color='red')
        plt.fill_between(data['Date'], data['BB_upper'], data['BB_lower'], color='gray', alpha=0.2, label='Bollinger Bands')
        plt.legend()
        plt.title(f"Price and Indicators for {crypto_symbol.capitalize()}")
        plt.xlabel("Date")
        plt.ylabel("Price")
        st.pyplot(plt)

        # Display RSI and MACD
        st.subheader("RSI and MACD")
        fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax[0].plot(data['Date'], data['RSI'], label='RSI', color='green')
        ax[0].axhline(70, color='red', linestyle='--', label='Overbought')
        ax[0].axhline(30, color='blue', linestyle='--', label='Oversold')
        ax[0].set_title("RSI")
        ax[0].legend()

        ax[1].plot(data['Date'], data['MACD'], label='MACD', color='purple')
        ax[1].plot(data['Date'], data['Signal_Line'], label='Signal Line', color='orange')
        ax[1].set_title("MACD")
        ax[1].legend()

        st.pyplot(fig)

st.sidebar.title("About")
st.sidebar.info("This tool analyzes cryptocurrency data using multiple indicators to detect potential market trends.")
