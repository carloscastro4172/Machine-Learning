import streamlit as st
import pandas as pd
import plotly.express as px

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
    st.sidebar.write("Aquí irán los filtros (contrato, género, rangos numéricos, etc.)")

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


