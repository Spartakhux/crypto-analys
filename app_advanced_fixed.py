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
    st.warning("Pas assez de données disponibles pour effectuer une analyse de corrélation.")
else:
    # Combine all data into one DataFrame
    combined_data = pd.concat(dataframes, axis=1)
    combined_data.columns = crypto_list

    # Interpolation des données manquantes
    combined_data = combined_data.interpolate(method='linear', axis=0)

    # Compute the correlation matrix
    correlation_matrix = combined_data.corr()

    st.write("### Matrice de Corrélation des Cryptomonnaies")
    st.dataframe(correlation_matrix)

    # Plot the correlation matrix
    st.write("### Visualisation de la Corrélation")
    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.matshow(correlation_matrix, cmap="coolwarm")
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.colorbar(cax)
    st.pyplot(fig)
