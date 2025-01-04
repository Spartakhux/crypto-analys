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

# Function to fetch market data (Market Cap and Circulating Supply)
def fetch_market_data(crypto_symbol):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}"
    response = requests.get(url)
    try:
        data = response.json()
        market_cap = data['market_data']['market_cap']['usd']
        circulating_supply = data['market_data']['circulating_supply']
        max_supply = data['market_data']['max_supply']
        return market_cap, circulating_supply, max_supply
    except Exception as e:
        return None, None, None

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

# Function to evaluate supply and demand
def evaluate_supply(circulating_supply, max_supply):
    if max_supply is None:
        return "Unknown"
    percentage_circulated = (circulating_supply / max_supply) * 100
    if percentage_circulated > 90:
        return "Rare (High Demand)"
    elif percentage_circulated < 50:
        return "Plenty Available (Low Demand)"
    else:
        return "Balanced"

# Function to provide advanced recommendations
def provide_advanced_recommendations(dataframes, market_data):
    recommendations = {}
    for crypto, df in dataframes.items():
        market_cap, circulating_supply, max_supply = market_data[crypto]
        rsi = df['RSI'].iloc[-1]

        # Analyze market cap and supply
        supply_status = evaluate_supply(circulating_supply, max_supply)

        if rsi < 30 and market_cap < 1e9:
            recommendations[crypto] = f"Strong Buy (Undervalued, {supply_status})"
        elif rsi > 70 and market_cap > 1e10:
            recommendations[crypto] = f"Strong Sell (Overvalued, {supply_status})"
        elif df['EMA_10'].iloc[-1] > df['EMA_50'].iloc[-1]:
            recommendations[crypto] = f"Buy ({supply_status})"
        elif df['EMA_10'].iloc[-1] < df['EMA_50'].iloc[-1]:
            recommendations[crypto] = f"Sell ({supply_status})"
        else:
            recommendations[crypto] = f"Hold ({supply_status})"
    return recommendations

# Streamlit application
st.title("Recommandations Avancées des Cryptomonnaies")

crypto_list = [
    "bitcoin", "ethereum", "solana", "cardano", "polkadot", "xrp ledger", "aave", "chainlink"
]
dataframes = {}
market_data = {}

st.write("### Téléchargement des données...")
for crypto in crypto_list:
    data = fetch_crypto_data(crypto, days=365)
    market_cap, circulating_supply, max_supply = fetch_market_data(crypto)

    if data is not None and market_cap is not None:
        data.set_index("Date", inplace=True)
        data = calculate_indicators(data)
        dataframes[crypto] = data
        market_data[crypto] = (market_cap, circulating_supply, max_supply)

if len(dataframes) < 1:
    st.warning("Aucune donnée disponible pour les cryptomonnaies sélectionnées.")
else:
    recommendations = provide_advanced_recommendations(dataframes, market_data)

    st.write("### Recommandations d'Investissement")
    for crypto, recommendation in recommendations.items():
        st.write(f"{crypto.capitalize()}: {recommendation}")

    st.write("### Données des Indicateurs Techniques")
    for crypto, df in dataframes.items():
        st.write(f"#### {crypto.capitalize()}")
        st.line_chart(df[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50']])
