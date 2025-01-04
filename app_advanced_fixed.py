import pandas as pd
import numpy as np
import requests
import streamlit as st

# Function to fetch cryptocurrency data from CoinGecko
def fetch_crypto_data(crypto_symbol, currency="usd", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    try:
        data = response.json()
        if 'prices' not in data or not data['prices']:
            return None

        # Convert to DataFrame
        df = pd.DataFrame(data['prices'], columns=["Date", "Close"])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        return df
    except Exception as e:
        return None

# Function to calculate indicators
def calculate_indicators(df):
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
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
        else:
            recommendations[crypto] = "Hold"
    return recommendations

# Streamlit application
st.title("Recommandations Avancées des Cryptomonnaies")

crypto_list = [
    "bitcoin", "ethereum", "solana", "cardano", "dot", "cosmos", "aave", "near"
]
dataframes = {}

st.write("### Téléchargement des données...")
for crypto in crypto_list:
    data = fetch_crypto_data(crypto, days=90)
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
        st.write(f"{crypto.capitalize()}: {recommendation}")

    st.write("### Données des Indicateurs Techniques")
    for crypto, df in dataframes.items():
        st.write(f"#### {crypto.capitalize()}")
        st.line_chart(df[['Close', 'SMA_10', 'SMA_50']])
