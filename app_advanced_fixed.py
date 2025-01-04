import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

# Function to fetch cryptocurrency data from Yahoo Finance
def fetch_yfinance_data(symbol, period="90d", interval="1d"):
    try:
        data = yf.download(tickers=symbol, period=period, interval=interval)
        if not data.empty:
            data.reset_index(inplace=True)
            data = data.rename(columns={"Adj Close": "Close"})
            return data[["Date", "Close"]]
        else:
            st.warning(f"Aucune donnée pour {symbol} sur Yahoo Finance.")
            return None
    except Exception as e:
        st.error(f"Erreur pour {symbol} : {str(e)}")
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
st.title("Recommandations Avancées des Cryptomonnaies (Yahoo Finance)")

crypto_list = [
    "BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD", "ATOM-USD", "AAVE-USD", "NEAR-USD"
]
dataframes = {}

st.write("### Téléchargement des données depuis Yahoo Finance...")
missing_data = []
for crypto in crypto_list:
    st.write(f"Téléchargement des données pour {crypto}...")
    data = fetch_yfinance_data(crypto, period="90d")
    if data is not None:
        data.set_index("Date", inplace=True)
        data = calculate_indicators(data)
        dataframes[crypto] = data
    else:
        missing_data.append(crypto)

if missing_data:
    st.warning(f"Données indisponibles pour : {', '.join(missing_data)}.")

if len(dataframes) < 1:
    st.warning("Aucune donnée disponible pour les cryptomonnaies sélectionnées.")
else:
    recommendations = provide_recommendations(dataframes)

    st.write("### Recommandations d'Investissement")
    for crypto, recommendation in recommendations.items():
        st.write(f"{crypto.split('-')[0]}: {recommendation}")

    st.write("### Données des Indicateurs Techniques")
    for crypto, df in dataframes.items():
        st.write(f"#### {crypto.split('-')[0]}")
        st.line_chart(df[['Close', 'SMA_10', 'SMA_50', 'EMA_10', 'EMA_50']])
