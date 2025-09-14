import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn  as sns
import datetime as dt

# Configuración de la página
st.set_page_config(
    page_title="EDA - Telco Customer Churn", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# TÍTULO
# ==========================
st.markdown("<h1 style='text-align:center;color:#1f4e79;'>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)

# ==========================
# CARGA DE DATOS
# ==========================


df = pd.read_csv("/home/carlos/Documents/8vo/MID_MACHINE/Mid_Machine/WA_Fn-UseC_-Telco-Customer-Churn.csv")

if df is not None and not df.empty:

    # ==========================
    # SIDEBAR: FILTROS
    # ==========================
    st.sidebar.header(" Filtros de Análisis")
    chart_type = st.sidebar.selectbox("Tipo de gráfico", ["Scatterplot", "Boxplot", "Countplot", "Histogram", "Heatmap", "Correlation Matrix"])
    # Variables numéricas y categóricas
    num_vars = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_vars = ["Churn", "Contract", "InternetService", "PaymentMethod", "PaperlessBilling",
                "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
                "OnlineSecurity", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


if chart_type == "Scatterplot":
    x_var = st.sidebar.selectbox("Variable X (numérica)", num_vars)
    y_var = st.sidebar.selectbox("Variable Y (numérica)", num_vars)
    hue_var = st.sidebar.selectbox("Hue (categórica)", ["Ninguno"] + cat_vars)

    fig, ax = plt.subplots(figsize=(6,4))
    if hue_var != "Ninguno":
        sns.scatterplot(x=x_var, y=y_var, hue=hue_var, data=df, ax=ax, alpha=0.7)
    else:
        sns.scatterplot(x=x_var, y=y_var, data=df, ax=ax, alpha=0.7)
    st.pyplot(fig)

elif chart_type == "Boxplot":
    x_var = st.sidebar.selectbox("Variable X (categórica)", cat_vars)
    y_var = st.sidebar.selectbox("Variable Y (numérica)", num_vars)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.boxplot(x=x_var, y=y_var, data=df, ax=ax)
    st.pyplot(fig)

elif chart_type == "Countplot":
    cat_var = st.sidebar.selectbox("Variable Categórica", cat_vars)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x=cat_var, data=df, ax=ax)
    st.pyplot(fig)

elif chart_type == "Histogram":
    num_var = st.sidebar.selectbox("Variable Numérica", num_vars)
    bins = st.sidebar.slider("Número de Bins", 5, 100, 20)

    fig, ax = plt.subplots(figsize=(6,4))
    sns.histplot(df[num_var], bins=bins, kde=True, ax=ax)
    st.pyplot(fig)

elif chart_type == "Heatmap":
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[num_vars].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
elif chart_type == "Correlation Matrix":
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(df[num_vars].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)



    # ==========================
    # MÉTRICAS PRINCIPALES
    # ==========================
    st.subheader(" Métricas Principales")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Tasa de Churn", "--")
    with col2: st.metric("Cargos Promedio", "--")
    with col3: st.metric("Tenure Promedio", "--")
    with col4: st.metric("Total Clientes", "--")





    # ==========================
    # TABS PRINCIPALES
    # ==========================
    tab1, tab2, tab3, tab4 = st.tabs([" Distribuciones", " Comparativo", " Segmentación", " Insights"])

    # TAB 1 - DISTRIBUCIONES
    with tab1:
        st.subheader("Distribuciones de Variables")
        st.write(" Aquí irá un histograma, boxplot y matriz de correlación")

    # TAB 2 - COMPARATIVO
    with tab2:
        st.subheader("Análisis Comparativo")
        st.write(" Aquí irá churn por servicios y demografía")

    # TAB 3 - SEGMENTACIÓN
    with tab3:
        st.subheader("Segmentación de Clientes")
        st.write(" Aquí irá el heatmap y la tabla de resumen por segmentos")

    # TAB 4 - INSIGHTS
    with tab4:
        st.subheader("Insights Clave")
        st.write(" Aquí se mostrarán hallazgos principales y factores de riesgo")

    # ==========================
    # FOOTER
    # ==========================
    st.markdown("---")


