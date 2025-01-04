import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt

# Function to fetch cryptocurrency data from CoinGecko
def fetch_crypto_data(crypto_symbol, currency="usd", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()

    # Convert to DataFrame
    df = pd.DataFrame(data['prices'], columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    df['Volume'] = [v[1] for v in data['total_volumes']]
    df['High'] = [h[1] for h in data['prices']]  # Approximation for high
    df['Low'] = [l[1] for l in data['prices']]   # Approximation for low
    return df

# Function to calculate advanced indicators
def calculate_indicators(df):
    # SMA and EMA
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = rolling_mean + (rolling_std * 2)
    df['BB_lower'] = rolling_mean - (rolling_std * 2)

    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = high_low.combine(high_close, max).combine(low_close, max)
    df['ATR'] = tr.rolling(window=14).mean()

    # Stochastic Oscillator
    df['%K'] = ((df['Close'] - df['Low'].rolling(window=14).min()) /
                (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
    df['%D'] = df['%K'].rolling(window=3).mean()

    return df

# Function to evaluate investment opportunity
def evaluate_investment(data):
    latest = data.iloc[-1]
    signals = []

    # RSI evaluation
    if latest['RSI'] < 30:
        signals.append("RSI indique des conditions de survente (potentiel signal d'achat).")
    elif latest['RSI'] > 70:
        signals.append("RSI indique des conditions de surachat (potentiel signal de vente).")

    # MACD evaluation
    if latest['MACD'] > latest['Signal_Line']:
        signals.append("Le MACD indique un élan haussier.")
    else:
        signals.append("Le MACD indique un élan baissier.")

    # Bollinger Bands evaluation
    if latest['Close'] < latest['BB_lower']:
        signals.append("Le prix est en dessous de la bande inférieure de Bollinger (potentiel signal d'achat).")
    elif latest['Close'] > latest['BB_upper']:
        signals.append("Le prix est au-dessus de la bande supérieure de Bollinger (potentiel signal de vente).")

    # Stochastic Oscillator evaluation
    if latest['%K'] < 20:
        signals.append("L'oscillateur stochastique indique des conditions de survente (potentiel signal d'achat).")
    elif latest['%K'] > 80:
        signals.append("L'oscillateur stochastique indique des conditions de surachat (potentiel signal de vente).")

    return signals

# Streamlit application
st.title("Outil Avancé d'Analyse des Cryptomonnaies")

# List of top 30 cryptocurrencies by market cap
crypto_list = [
    "bitcoin", "ethereum", "binancecoin", "ripple", "cardano", "solana", "dogecoin", "polygon", "polkadot", "tron",
    "litecoin", "shiba-inu", "avalanche", "chainlink", "uniswap", "cosmos", "monero", "stellar", "ethereum-classic", "bitcoin-cash",
    "filecoin", "hedera", "aptos", "vechain", "algorand", "quant", "the-graph", "flow", "axie-infinity", "tezos"
]

# User selects cryptocurrency and duration
crypto_symbol = st.selectbox(
    "Sélectionnez une cryptomonnaie :",
    crypto_list
)
days = st.slider("Sélectionnez la durée (jours) :", min_value=7, max_value=90, value=30)

if st.button("Analyser"):
    with st.spinner("Récupération des données et analyse en cours..."):
        # Fetch data
        data = fetch_crypto_data(crypto_symbol, days=days)
        data = calculate_indicators(data)

        # Display data
        st.subheader("Données Récentes")
        st.write(data.tail(10))

        # Plot price and indicators
        st.subheader("Prix et Indicateurs")
        st.write("Ce graphique montre les variations de prix, les moyennes mobiles et les bandes de Bollinger.")
        plt.figure(figsize=(10, 6))
        plt.plot(data['Date'], data['Close'], label='Prix', color='blue')
        plt.plot(data['Date'], data['SMA_10'], label='Moyenne Mobile (SMA 10)', color='orange')
        plt.plot(data['Date'], data['SMA_20'], label='Moyenne Mobile (SMA 20)', color='red')
        plt.fill_between(data['Date'], data['BB_upper'], data['BB_lower'], color='gray', alpha=0.2, label='Bandes de Bollinger')
        plt.legend()
        plt.title(f"Prix et Indicateurs pour {crypto_symbol.capitalize()}")
        plt.xlabel("Date")
        plt.ylabel("Prix")
        st.pyplot(plt)

        # Display RSI, MACD, and Stochastic Oscillator
        st.subheader("RSI, MACD et Oscillateur Stochastique")
        st.write("Ces graphiques illustrent les tendances de l'indice RSI, le MACD et l'oscillateur stochastique.")
        fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # RSI
        ax[0].plot(data['Date'], data['RSI'], label='RSI', color='green')
        ax[0].axhline(70, color='red', linestyle='--', label='Surachat')
        ax[0].axhline(30, color='blue', linestyle='--', label='Survente')
        ax[0].set_title("RSI")
        ax[0].legend()

        # MACD
        ax[1].plot(data['Date'], data['MACD'], label='MACD', color='purple')
        ax[1].plot(data['Date'], data['Signal_Line'], label='Ligne de Signal', color='orange')
        ax[1].set_title("MACD")
        ax[1].legend()

        # Stochastic Oscillator
        ax[2].plot(data['Date'], data['%K'], label='%K', color='cyan')
        ax[2].plot(data['Date'], data['%D'], label='%D', color='magenta')
        ax[2].set_title("Oscillateur Stochastique")
        ax[2].legend()

        st.pyplot(fig)

        # Evaluate investment opportunity
        st.subheader("Analyse des Opportunités d'Investissement")
        st.write("Cette section évalue les opportunités d'investissement en fonction des indicateurs actuels.")
        signals = evaluate_investment(data)
        for signal in signals:
            st.write(f"- {signal}")

st.sidebar.title("À propos")
st.sidebar.info("Cet outil analyse les données des cryptomonnaies en utilisant des indicateurs avancés pour détecter les tendances du marché et évaluer les opportunités d'investissement.")
