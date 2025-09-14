import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import datetime as dt

# Importamos funciones desde EDA.py
from EDA import clean_dt, metric

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(page_title="EDA - Telco Customer Churn", layout="wide")
st.title("📊 Exploratory Data Analysis - Telco Churn")

# ==============================
# CARGA DE DATOS
# ==============================
uploaded_file = st.file_uploader("📂 Cargar dataset CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_dt(df)

    st.subheader("Vista previa de los datos")
    st.dataframe(df.head())

    # ==============================
    # MÉTRICAS DE COLUMNA
    # ==============================
    st.sidebar.header("⚙️ Opciones")
    column_selected = st.sidebar.selectbox("Selecciona una columna para ver métricas", df.columns)

    st.markdown(f"### 📌 Métricas para **{column_selected}**")
    st.write(metric(df, column_selected))

    st.markdown("---")

    # ==============================
    # VARIABLES
    # ==============================
    num_vars = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_vars = [c for c in df.columns if c not in num_vars]

    chart_type = st.sidebar.selectbox(
        "Tipo de gráfico",
        ["Scatterplot", "Boxplot", "Countplot", "Histogram", "Heatmap"]
    )

    # ==============================
    # VISUALIZACIONES
    # ==============================
    if chart_type == "Scatterplot":
        x_var = st.sidebar.selectbox("Variable X (numérica)", num_vars)
        y_var = st.sidebar.selectbox("Variable Y (numérica)", num_vars)
        hue_var = st.sidebar.selectbox("Hue (categórica)", ["Ninguno"] + cat_vars)

        fig, ax = plt.subplots(figsize=(7, 5))
        if hue_var != "Ninguno":
            sns.scatterplot(x=x_var, y=y_var, hue=hue_var, data=df, ax=ax)
        else:
            sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax)
        st.pyplot(fig)

    elif chart_type == "Boxplot":
        x_var = st.sidebar.selectbox("Variable X (categórica)", cat_vars)
        y_var = st.sidebar.selectbox("Variable Y (numérica)", num_vars)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=x_var, y=y_var, data=df, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        st.pyplot(fig)

    elif chart_type == "Countplot":
        x_var = st.sidebar.selectbox("Variable categórica", cat_vars)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(x=x_var, data=df, ax=ax, palette="Set2")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

    elif chart_type == "Histogram":
        num_var = st.sidebar.selectbox("Variable numérica", num_vars)
        bins = st.sidebar.slider("Número de bins", 5, 100, 20)
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.histplot(df[num_var], bins=bins, kde=True, ax=ax, color="skyblue")
        st.pyplot(fig)

    elif chart_type == "Heatmap":
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df[num_vars].corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)

else:
    st.info("👆 Sube un archivo CSV para comenzar el análisis.")
