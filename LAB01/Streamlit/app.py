import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from EDA import clean_dt, metric, kpis_por_segmento

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(page_title="EDA - Telco Customer Churn", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1f4e79;'> Telco Customer Churn - EDA</h1>", unsafe_allow_html=True)

# ==============================
# CARGA DE DATOS
# ==============================
uploaded_file = st.file_uploader(" Cargar dataset CSV", type=["csv"])

if uploaded_file is None:
    st.info(" Sube un archivo CSV para comenzar el análisis.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df = clean_dt(df_raw.copy())

st.subheader("Vista previa de los datos")
st.dataframe(df.head())

# ==============================
# SIDEBAR: FILTROS (afectan todo)
# ==============================
st.sidebar.header(" Filtros de análisis")

# Variables base
num_vars = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] if c in df.columns]
cat_vars = [c for c in df.columns if c not in num_vars]

# Categóricos
with st.sidebar.expander("Categóricos", expanded=True):
    # Prepara opciones seguras (orden alfabético, sin NaN)
    def opts(col):
        return sorted([v for v in df[col].dropna().unique()])

    sel_contract = st.multiselect("Contract", opts("Contract") if "Contract" in df.columns else [], default=opts("Contract") if "Contract" in df.columns else [])
    sel_internet = st.multiselect("InternetService", opts("InternetService") if "InternetService" in df.columns else [], default=opts("InternetService") if "InternetService" in df.columns else [])
    sel_payment  = st.multiselect("PaymentMethod", opts("PaymentMethod") if "PaymentMethod" in df.columns else [], default=opts("PaymentMethod") if "PaymentMethod" in df.columns else [])
    sel_gender   = st.multiselect("gender", opts("gender") if "gender" in df.columns else [], default=opts("gender") if "gender" in df.columns else [])
    sel_senior   = st.multiselect("SeniorCitizen", sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else [], default=sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else [])
    solo_churn   = st.checkbox("Mostrar solo clientes con Churn = 1", value=False)

# Numéricos
with st.sidebar.expander("Numéricos", expanded=True):
    def range_slider_for(col, step=1.0):
        vmin = float(df[col].min()) if col in df.columns else 0.0
        vmax = float(df[col].max()) if col in df.columns else 0.0
        if col == "tenure": step = 1
        return st.slider(f"{col}", min_value=float(vmin), max_value=float(vmax), value=(float(vmin), float(vmax)), step=float(step))
    ten_r = range_slider_for("tenure", step=1)
    mon_r = range_slider_for("MonthlyCharges", step=0.5)
    tot_r = range_slider_for("TotalCharges", step=1.0)

# Aplicar filtros
df_f = df.copy()

if sel_contract and "Contract" in df_f.columns:
    df_f = df_f[df_f["Contract"].isin(sel_contract)]
if sel_internet and "InternetService" in df_f.columns:
    df_f = df_f[df_f["InternetService"].isin(sel_internet)]
if sel_payment and "PaymentMethod" in df_f.columns:
    df_f = df_f[df_f["PaymentMethod"].isin(sel_payment)]
if sel_gender and "gender" in df_f.columns:
    df_f = df_f[df_f["gender"].isin(sel_gender)]
if sel_senior and "SeniorCitizen" in df_f.columns:
    df_f = df_f[df_f["SeniorCitizen"].isin(sel_senior)]

if "tenure" in df_f.columns:
    df_f = df_f[df_f["tenure"].between(ten_r[0], ten_r[1])]
if "MonthlyCharges" in df_f.columns:
    df_f = df_f[df_f["MonthlyCharges"].between(mon_r[0], mon_r[1])]
if "TotalCharges" in df_f.columns:
    df_f = df_f[df_f["TotalCharges"].between(tot_r[0], tot_r[1])]

if solo_churn and "Churn" in df_f.columns:
    df_f = df_f[df_f["Churn"] == 1]

if df_f.empty:
    st.warning(" No hay datos para los filtros seleccionados. Ajusta los filtros en el sidebar.")
    st.stop()

# ==============================
# KPIs PRINCIPALES (dinámicos)
# ==============================
st.markdown("##  Métricas Principales (dataset filtrado)")
col1, col2, col3, col4 = st.columns(4)
churn_rate = (df_f["Churn"].mean() * 100) if "Churn" in df_f.columns else 0.0
avg_charge = df_f["MonthlyCharges"].mean() if "MonthlyCharges" in df_f.columns else 0.0
avg_tenure = df_f["tenure"].mean() if "tenure" in df_f.columns else 0.0
total_clients = len(df_f)

col1.metric("Tasa de Churn", f"{churn_rate:.2f}%")
col2.metric("Cargos Promedio", f"${avg_charge:.2f}")
col3.metric("Tenure Promedio", f"{avg_tenure:.1f} meses")
col4.metric("Total Clientes", f"{total_clients:,}")

st.markdown("---")

# ==============================
# TABS PRINCIPALES
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([" Distribuciones", " Comparativo", "Segmentación", " Insights"])

# ---- Tab 1: Distribuciones (hist + box + scatter interactivo)
with tab1:
    st.subheader("Distribución de variables numéricas")
    for col in num_vars:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(df_f[col], kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histograma: {col}")
        st.pyplot(fig)

    st.subheader("Boxplots por categoría")
    x_cat = st.selectbox("Categoría para boxplots", [c for c in cat_vars if c in df_f.columns], index=([c for c in cat_vars if c in df_f.columns].index("Churn") if "Churn" in df_f.columns else 0))
    for y_num in num_vars:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(x=x_cat, y=y_num, data=df_f, ax=ax)
        ax.set_title(f"{y_num} por {x_cat}")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
        st.pyplot(fig)

    st.subheader("Scatterplot interactivo")
    if len(num_vars) >= 2:
        x_var = st.selectbox("X (numérica)", num_vars, index=0, key="sc_x")
        y_var = st.selectbox("Y (numérica)", num_vars, index=1, key="sc_y")
        hue_choices = ["Ninguno"] + [c for c in cat_vars if c in df_f.columns]
        hue_var = st.selectbox("Hue (categórica)", hue_choices, index=hue_choices.index("Churn") if "Churn" in hue_choices else 0)
        fig, ax = plt.subplots(figsize=(7, 5))
        if hue_var != "Ninguno":
            sns.scatterplot(data=df_f, x=x_var, y=y_var, hue=hue_var, ax=ax, alpha=0.7)
        else:
            sns.scatterplot(data=df_f, x=x_var, y=y_var, ax=ax, alpha=0.7)
        ax.set_title(f"{x_var} vs {y_var}")
        st.pyplot(fig)

# ---- Tab 2: Comparativo (countplots clave)
with tab2:
    st.subheader("Churn por Contrato")
    if "Contract" in df_f.columns and "Churn" in df_f.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_f, x="Contract", hue="Churn", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

    st.subheader("Churn por InternetService")
    if "InternetService" in df_f.columns and "Churn" in df_f.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df_f, x="InternetService", hue="Churn", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        st.pyplot(fig)

# ---- Tab 3: Segmentación (heatmap + KPIs por segmento)
with tab3:
    st.subheader("Correlación (solo numéricas)")
    if len(num_vars) >= 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(df_f[num_vars].corr(), annot=True, cmap="coolwarm", ax=ax, fmt=".2f")
        st.pyplot(fig)

    st.subheader("KPIs por segmento")
    seg_opts = [c for c in cat_vars if c in df_f.columns]
    if seg_opts:
        seg_col = st.selectbox("Segmentar por", seg_opts, index=(seg_opts.index("Contract") if "Contract" in seg_opts else 0))
        seg_table = kpis_por_segmento(df_f, seg_col)
        st.dataframe(seg_table, use_container_width=True)
    else:
        st.info("No hay variables categóricas disponibles para segmentar.")

# ---- Tab 4: Insights (conclusiones automáticas simples)
with tab4:
    st.subheader("Insights clave (basados en los filtros actuales)")
    bullets = []

    # Mayor churn por tipo de contrato
    if "Contract" in df_f.columns and "Churn" in df_f.columns and df_f["Contract"].nunique() > 0:
        gr = df_f.groupby("Contract")["Churn"].mean().sort_values(ascending=False) * 100
        top_c = gr.index[0]
        bullets.append(f"• El mayor churn se observa en **{top_c}** ({gr.iloc[0]:.2f}%).")

    # Mayor churn por InternetService
    if "InternetService" in df_f.columns and df_f["InternetService"].nunique() > 0:
        gr = df_f.groupby("InternetService")["Churn"].mean().sort_values(ascending=False) * 100
        bullets.append(f"• Por servicio de Internet, destaca **{gr.index[0]}** con churn de {gr.iloc[0]:.2f}%.")

    # Método de pago
    if "PaymentMethod" in df_f.columns and df_f["PaymentMethod"].nunique() > 0:
        gr = df_f.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False) * 100
        bullets.append(f"• En métodos de pago, **{gr.index[0]}** presenta mayor churn ({gr.iloc[0]:.2f}%).")

    # Relación tenure-churn
    if "tenure" in df_f.columns and "Churn" in df_f.columns:
        low_ten = df_f[df_f["tenure"] <= df_f["tenure"].median()]["Churn"].mean() * 100
        high_ten = df_f[df_f["tenure"] > df_f["tenure"].median()]["Churn"].mean() * 100
        if pd.notna(low_ten) and pd.notna(high_ten):
            bullets.append(f"• Clientes con **tenure bajo** tienen churn {low_ten:.2f}% vs. {high_ten:.2f}% en tenure alto.")

    if bullets:
        st.markdown("\n".join(bullets))
    else:
        st.info("Ajusta los filtros para generar insights.")
