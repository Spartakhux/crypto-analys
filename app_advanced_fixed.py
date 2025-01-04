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

    # Vérification de la présence des données
    if 'prices' not in data or not data['prices']:
        st.warning(f"Aucune donnée disponible pour {crypto_symbol}. Elle sera ignorée.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(data['prices'], columns=["Date", "Close"])
    df['Date'] = pd.to_datetime(df['Date'], unit='ms')
    return df

# Function to normalize data
def normalize_data(df):
    return df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

# Function to plot normalized data
def plot_normalized_data(dataframes, duration_name):
    combined_data = pd.concat(dataframes.values(), axis=1)
    combined_data.columns = dataframes.keys()

    # Interpolate missing data and sort by date
    combined_data = combined_data.sort_index().interpolate(method='linear', axis=0)

    # Normalize the data
    normalized_data = normalize_data(combined_data)

    # Plot the data
    st.write(f"### Courbes de Prix Normalisées - {duration_name}")
    fig, ax = plt.subplots(figsize=(12, 8))
    for crypto in normalized_data.columns:
        ax.plot(normalized_data.index, normalized_data[crypto], label=crypto)

    ax.set_title(f"Courbes de Prix Normalisées ({duration_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix Normalisé")
    ax.legend()
    st.pyplot(fig)

# Streamlit application
st.title("Analyse de Corrélation des Cryptomonnaies")

crypto_list = [
    "bitcoin", "ethereum", "solana", "link", "cardano", "dot", "cosmos", "aave", "near"
]

# Durations for analysis
durations = {
    "5 ans": 1825,
    "1 an": 365,
    "1 mois": 30
}

for duration_name, days in durations.items():
    st.write(f"### Analyse sur {duration_name}")
    dataframes = {}

    for crypto in crypto_list:
        data = fetch_crypto_data(crypto, days=days)
        if data is not None:
            data.set_index("Date", inplace=True)
            dataframes[crypto] = data["Close"]

    if len(dataframes) < 2:
        st.warning(f"Pas assez de données disponibles pour afficher les courbes pour {duration_name}.")
    else:
        plot_normalized_data(dataframes, duration_name)
