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

# Streamlit application
st.title("Analyse de Corrélation des Cryptomonnaies")

crypto_list = [
    "bitcoin", "ethereum", "solana", "link", "cardano", "dot", "cosmos", "aave", "near"
]

dataframes = {}

st.write("### Téléchargement des données en cours...")
for crypto in crypto_list:
    data = fetch_crypto_data(crypto, days=90)
    if data is not None:
        data.set_index("Date", inplace=True)
        dataframes[crypto] = data["Close"]

if len(dataframes) < 2:
    st.warning("Pas assez de données disponibles pour afficher les courbes superposées.")
else:
    # Combine all data into one DataFrame
    combined_data = pd.concat(dataframes, axis=1)

    # Ajuster dynamiquement les noms des colonnes
    combined_data.columns = [crypto for crypto in crypto_list if crypto in dataframes]

    # Normaliser les données pour chaque cryptomonnaie (échelle relative)
    normalized_data = combined_data.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)

    # Plot the normalized data
    st.write("### Courbes de Prix Normalisées (Superposées)")
    fig, ax = plt.subplots(figsize=(12, 8))
    for crypto in normalized_data.columns:
        ax.plot(normalized_data.index, normalized_data[crypto], label=crypto)

    ax.set_title("Courbes de Prix Normalisées (Superposées)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix Normalisé")
    ax.legend()
    st.pyplot(fig)
