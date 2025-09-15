import streamlit as st
import pandas as pd
import altair as alt

from EDA import clean_dt, kpis_por_segmento

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
n_rows = st.slider("Número de filas a mostrar", min_value=5, max_value=len(df), value=5, step=5)
st.dataframe(df.head(n_rows))

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
    for i in range(0, len(num_vars), 2):
        cols = st.columns(2)
        for j, col in enumerate(num_vars[i:i+2]):
            with cols[j]:
                chart = alt.Chart(df_f).mark_bar(opacity=0.8, color="#1f77b4").encode(
                    alt.X(col, bin=alt.Bin(maxbins=30), title=col),
                    alt.Y('count()', title='Frecuencia'),
                    tooltip=[col]
                ).properties(height=300).configure_axis(
                    labelColor="lightgray",
                    titleColor="white"
                ).interactive()
                st.altair_chart(chart, use_container_width=True)

    st.subheader("Boxplots por categoría")
    x_cat = st.selectbox("Categoría para boxplots", [c for c in cat_vars if c in df_f.columns], index=([c for c in cat_vars if c in df_f.columns].index("Churn") if "Churn" in df_f.columns else 0))
    for i in range(0, len(num_vars), 2):
        cols = st.columns(2)
        for j, y_num in enumerate(num_vars[i:i+2]):
            with cols[j]:
                chart = alt.Chart(df_f).mark_boxplot(color="#ff7f0e").encode(
                    x=alt.X(x_cat, title=x_cat),
                    y=alt.Y(y_num, title=y_num),
                    color=x_cat
                ).properties(height=300).configure_axis(
                    labelColor="lightgray",
                    titleColor="white"
                )
                st.altair_chart(chart, use_container_width=True)

    st.subheader("Scatterplot interactivo")
    if len(num_vars) >= 2:
        x_var = st.selectbox("X (numérica)", num_vars, index=0, key="sc_x")
        y_var = st.selectbox("Y (numérica)", num_vars, index=1, key="sc_y")
        hue_choices = ["Ninguno"] + [c for c in cat_vars if c in df_f.columns]
        hue_var = st.selectbox("Hue (categórica)", hue_choices, index=hue_choices.index("Churn") if "Churn" in hue_choices else 0)
        color_enc = alt.Color(hue_var) if hue_var != "Ninguno" else alt.value("#2ca02c")
        chart = alt.Chart(df_f).mark_circle(size=70, opacity=0.6).encode(
            x=alt.X(x_var, title=x_var),
            y=alt.Y(y_var, title=y_var),
            color=color_enc,
            tooltip=num_vars + cat_vars
        ).properties(height=350).configure_axis(
            labelColor="lightgray",
            titleColor="white"
        ).interactive()
        st.altair_chart(chart, use_container_width=True)

# ---- Tab 2: Comparativo (countplots clave)
with tab2:
    st.subheader("Churn por categoría")
    comp_vars = ["Contract", "InternetService", "PaymentMethod"]
    for i in range(0, len(comp_vars), 2):
        cols = st.columns(2)
        for j, var in enumerate(comp_vars[i:i+2]):
            if var in df_f.columns and "Churn" in df_f.columns:
                with cols[j]:
                    chart = alt.Chart(df_f).mark_bar(opacity=0.85).encode(
                        x=alt.X(var, title=var),
                        y='count()',
                        color=alt.Color('Churn:N', scale=alt.Scale(scheme="dark2")),
                        tooltip=[var, 'Churn']
                    ).properties(height=300).configure_axis(
                        labelColor="lightgray",
                        titleColor="white"
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)

# ---- Tab 3: Segmentación (heatmap + KPIs por segmento)
with tab3:
    st.subheader("Correlación (solo numéricas)")
    if len(num_vars) >= 2:
        corr = df_f[num_vars].corr().reset_index().melt("index")
        corr_chart = alt.Chart(corr).mark_rect().encode(
            x=alt.X('index:N', title=''),
            y=alt.Y('variable:N', title=''),
            color=alt.Color('value:Q', scale=alt.Scale(scheme="redblue", domain=(-1, 1))),
            tooltip=['index', 'variable', alt.Tooltip('value:Q', format=".2f")]
        ).properties(height=400).configure_axis(
            labelColor="white",
            titleColor="white"
        )
        st.altair_chart(corr_chart, use_container_width=True)

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
        bullets.append(f"• El mayor churn se observa en **{gr.index[0]}** ({gr.iloc[0]:.2f}%).")
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
