
import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Prévision des ventes", layout="wide")
st.title("📈 Dashboard interactif - Prévision des ventes")

fichier = st.file_uploader("📁 Importer le fichier Excel des ventes", type="xlsx")

if fichier:
    df_ventes = pd.read_excel(fichier, sheet_name="F_Ventes")
    df_produit = pd.read_excel(fichier, sheet_name="D_Produit")
    df_client = pd.read_excel(fichier, sheet_name="D_Client")
    df_temps = pd.read_excel(fichier, sheet_name="D_Temps")
    df_magasin = pd.read_excel(fichier, sheet_name="D_Magasin")

    df_temps.columns = [col.lower().replace('é', 'e') for col in df_temps.columns]
    df_temps = df_temps.rename(columns={'année': 'annee'})

    df = df_ventes.merge(df_produit, on="id_produit")                  .merge(df_client, on="id_client")                  .merge(df_temps, on="id_temps")                  .merge(df_magasin, on="id_magasin")

    df["ds"] = pd.to_datetime(dict(year=df["annee"], month=df["mois"], day=df["jour"]))

    produit = st.selectbox("🛍️ Choisir un produit", sorted(df["nom_produit"].unique()))
    region = st.selectbox("📍 Choisir une région", sorted(df["region"].unique()))

    df_filtre = df[(df["nom_produit"] == produit) & (df["region"] == region)]

    if df_filtre.empty:
        st.warning("Aucune donnée pour cette combinaison produit + région.")
    else:
        df_prophet = df_filtre.groupby("ds")[["montant_total"]].sum().reset_index()
        df_prophet.columns = ["ds", "y"]

        if df_prophet.shape[0] < 2:
            st.error("Pas assez de données pour faire une prédiction.")
        else:
            st.success(f"Prévision des ventes pour {produit} en région {region}")
            st.line_chart(df_prophet.rename(columns={"ds": "index"}).set_index("index"))

            model = Prophet()
            model.fit(df_prophet)

            future = model.make_future_dataframe(periods=15)
            forecast = model.predict(future)

            fig1 = model.plot(forecast)
            st.pyplot(fig1)

            st.subheader("📊 Composantes de la prévision")
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

            st.subheader("📁 Télécharger les résultats")
            output = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            towrite = BytesIO()
            output.to_excel(towrite, index=False)
            st.download_button("📥 Télécharger la prévision", towrite.getvalue(), file_name="prevision.xlsx")
