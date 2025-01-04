import pandas as pd
import numpy as np
import requests
import streamlit as st

# Function to fetch cryptocurrency data from Binance API
def fetch_binance_data(symbol, interval="1d", limit=90):
    url = f"https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
    response = requests.get(url, params=params)
    try:
        data = response.json()
        if response.status_code == 200 and len(data) > 0:
            df = pd.DataFrame(data, columns=["Open Time", "Open", "High", "Low", "Close", "Volume", "Close Time", "Quote Volume", "Trades", "TBBAV", "TBQAV", "Ignore"])
            df["Date"] = pd.to_datetime(df["Open Time"], unit="ms")
            df["Close"] = pd.to_numeric(df["Close"])
            return df[["Date", "Close"]]
        else:
            return None
    except Exception as e:
        return None

# Function to calculate indicators
def calculate_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

# Function to calculate RSI
def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to provide recommendations
def provide_recommendations(dataframes):
    recommendations = {}
    for crypto, df in dataframes.items():
        if df['RSI'].iloc[-1] < 30:
            recommendations[crypto] = "Strong Buy"
        elif df['RSI'].iloc[-1] > 70:
            recommendations[crypto] = "Strong Sell"
        elif df['EMA_10'].iloc[-1] > df['EMA_50'].iloc[-1]:
            recommendations[crypto] = "Buy"
        elif df['EMA_10'].iloc[-1] < df['EMA_50'].iloc[-1]:
            recommendations[crypto] = "Sell"
        else:
            recommendations[crypto] = "Hold"
    return recommendations

# Streamlit application
st.title("Recommandations Avancées des Cryptomonnaies (Binance API)")

crypto_list = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT", "DOTUSDT", "ATOMUSDT", "AAVEUSDT", "NEARUSDT"
]
dataframes = {}

st.write("### Téléchargement des données depuis Binance...")
for crypto in crypto_list:
    data = fetch_binance_data(crypto, interval="1d", limit=90)
    if data is not None:
        data.set_index("Date", inplace=True)
        data = calculate_indicators(data)
        dataframes[crypto] = data

if len(dataframes) < 1:
    st.warning("Aucune donnée disponible pour les cryptomonnaies sélectionnées.")
else:
    recommendations = provide_recommendations(dataframes)

    st.write("### Recommandations d'Investissement")
    for crypto, recommendation in recommendations.items():
        st.write(f"{crypto[:-4]}: {recommendation}")

    st.write("### Données des Indicateurs Techniques")
    for crypto, df in dataframes.items():
        st.write(f"#### {crypto[:-4]}")
        st.line_chart(df[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50']])
