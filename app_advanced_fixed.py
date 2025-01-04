
import pandas as pd
import numpy as np
import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Function to fetch cryptocurrency data from CoinGecko
def fetch_crypto_data(crypto_symbol, currency="usd", days=90):
    url = f"https://api.coingecko.com/api/v3/coins/{crypto_symbol}/market_chart"
    params = {"vs_currency": currency, "days": days}
    response = requests.get(url, params=params)
    data = response.json()

    # Vérification de la présence des données
    if 'prices' not in data or not data['prices']:
        st.warning(f"Aucune donnée disponible pour {crypto_symbol}. Elle sera ignorée.")
        return None

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

# Function for sentiment analysis
def fetch_sentiment_data(crypto_symbol):
    # Simulated sentiment score (replace with actual API call if available)
    sentiment_score = np.random.uniform(-1, 1)  # Simulated sentiment score between -1 and 1
    return sentiment_score

# Function for market correlation
def fetch_market_data():
    # Simulated market data (replace with actual API call if available)
    market_correlation = {
        "S&P 500": np.random.uniform(-1, 1),
        "Nasdaq": np.random.uniform(-1, 1)
    }
    return market_correlation

# Machine learning model for trend prediction
def train_ml_model(data):
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    features = data[['RSI', 'MACD', 'SMA_10', 'SMA_20', 'ATR']].dropna()
    target = data['Target']
    target = target.loc[features.index]  # Synchronize indices
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy

# Function to evaluate investment opportunity
def evaluate_investment(data, sentiment, market_corr):
    latest = data.iloc[-1]
    score = 0

    # RSI evaluation
    if latest['RSI'] < 30:
        score += 1
    elif latest['RSI'] > 70:
        score -= 1

    # MACD evaluation
    if latest['MACD'] > latest['Signal_Line']:
        score += 1
    else:
        score -= 1

    # Bollinger Bands evaluation
    if latest['Close'] < latest['BB_lower']:
        score += 1
    elif latest['Close'] > latest['BB_upper']:
        score -= 1

    # Sentiment analysis
    if sentiment > 0.2:
        score += 1
    elif sentiment < -0.2:
        score -= 1

    # Market correlation
    if market_corr['S&P 500'] > 0.5 or market_corr['Nasdaq'] > 0.5:
        score += 1
    elif market_corr['S&P 500'] < -0.5 or market_corr['Nasdaq'] < -0.5:
        score -= 1

    return score

# Recommendations
def get_recommendations():
    scores = {}
    for crypto in crypto_list:
        try:
            data = fetch_crypto_data(crypto, days=30)
            if data is None:
                continue  # Ignore cryptos without data
            data = calculate_indicators(data)
            sentiment = fetch_sentiment_data(crypto)
            market_corr = fetch_market_data()
            score = evaluate_investment(data, sentiment, market_corr)
            scores[crypto] = score
        except Exception as e:
            scores[crypto] = None

    sorted_scores = sorted(scores.items(), key=lambda x: x[1] if x[1] is not None else -999, reverse=True)
    best = sorted_scores[:5]
    worst = sorted_scores[-5:]
    return best, worst

# Streamlit application
st.title("Outil Avancé d'Analyse des Cryptomonnaies")

# List of top 30 cryptocurrencies by market cap
crypto_list = [
    "bitcoin", "ethereum", "binancecoin", "ripple", "cardano", "solana", "dogecoin", "polygon", "polkadot", "tron",
    "litecoin", "shiba-inu", "avalanche", "chainlink", "uniswap", "cosmos", "monero", "stellar", "ethereum-classic", "bitcoin-cash",
    "filecoin", "hedera", "aptos", "vechain", "algorand", "quant", "the-graph", "flow", "axie-infinity", "tezos"
]

if st.button("Afficher les recommandations"):
    with st.spinner("Analyse de toutes les cryptomonnaies..."):
        best, worst = get_recommendations()

        st.subheader("Meilleures opportunités")
        for crypto, score in best:
            st.write(f"- {crypto.capitalize()} avec un score de {score}")

        st.subheader("Cryptomonnaies à éviter")
        for crypto, score in worst:
            st.write(f"- {crypto.capitalize()} avec un score de {score}")

crypto_symbol = st.selectbox(
    "Sélectionnez une cryptomonnaie :",
    crypto_list
)
days = st.slider("Sélectionnez la durée (jours) :", min_value=7, max_value=90, value=30)

if st.button("Analyser"):
    with st.spinner("Analyse en cours..."):
        data = fetch_crypto_data(crypto_symbol, days=days)
        if data is not None:
            data = calculate_indicators(data)
            sentiment = fetch_sentiment_data(crypto_symbol)
            market_corr = fetch_market_data()

            model, accuracy = train_ml_model(data)
            st.write(f"Précision du modèle de machine learning : {accuracy:.2f}")

            score = evaluate_investment(data, sentiment, market_corr)
            st.write(f"Score global pour {crypto_symbol} : {score}")
            st.write("### Analyse de Sentiment")
            st.write(f"Score de sentiment : {sentiment:.2f}")
            st.write("### Corrélation avec le Marché")
            st.write(market_corr)

            # Plot indicators
            st.subheader("Prix et Indicateurs")
            plt.figure(figsize=(10, 6))
            plt.plot(data['Date'], data['Close'], label='Prix', color='blue')
            plt.plot(data['Date'], data['SMA_10'], label='SMA 10', color='orange')
            plt.plot(data['Date'], data['SMA_20'], label='SMA 20', color='red')
            plt.fill_between(data['Date'], data['BB_upper'], data['BB_lower'], color='gray', alpha=0.2, label='Bandes de Bollinger')
            plt.legend()
            st.pyplot(plt)
