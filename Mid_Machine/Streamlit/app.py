import streamlit as st
import pandas as pd
import altair as alt
from Transformers import DataFramePreparer

from EDA import clean_dt, kpis_por_segmento

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(page_title="Telco Customer Churn", layout="wide")
st.title("Telco Customer Churn")

# ==============================
# CARGA DE DATOS
# ==============================
uploaded_file = st.file_uploader("Cargar dataset CSV", type=["csv"])
if uploaded_file is None:
    st.info("Sube un archivo CSV para comenzar el análisis.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)
df = clean_dt(df_raw.copy())

# ==============================
# TABS PRINCIPALES
# ==============================
tab1, tab2, tab3, tab4 = st.tabs([
    "EDA",
    "Inference",
    "Dashboard",
    "Business Impact"
])

# -------------------------------------------------------------------
# ---------------------------- TAB 1: EDA ---------------------------
# -------------------------------------------------------------------
with tab1:
    st.subheader("Vista previa")
    n_rows = st.slider("Filas a mostrar", min_value=5, max_value=len(df), value=5, step=5)
    st.dataframe(df.head(n_rows), use_container_width=True)

    # Sidebar de filtros
    st.sidebar.header("Filtros de análisis")

    num_vars = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] if c in df.columns]
    cat_vars = [c for c in df.columns if c not in num_vars]

    def opts(col):
        return sorted([v for v in df[col].dropna().unique()]) if col in df.columns else []

    # Filtros categóricos
    with st.sidebar.expander("Variables categóricas", expanded=True):
        sel_contract = st.multiselect("Contract", opts("Contract"), default=opts("Contract"))
        sel_internet = st.multiselect("InternetService", opts("InternetService"), default=opts("InternetService"))
        sel_payment  = st.multiselect("PaymentMethod", opts("PaymentMethod"), default=opts("PaymentMethod"))
        sel_gender   = st.multiselect("gender", opts("gender"), default=opts("gender"))
        sel_senior   = st.multiselect("SeniorCitizen",
                                      sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else [],
                                      default=sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else [])
        solo_churn   = st.checkbox("Mostrar solo clientes con Churn = 1", value=False)

    # Filtros numéricos
    with st.sidebar.expander("Variables numéricas", expanded=True):
        def range_slider_for(col, step=1.0):
            vmin = float(df[col].min()) if col in df.columns else 0.0
            vmax = float(df[col].max()) if col in df.columns else 0.0
            if col == "tenure": step = 1
            return st.slider(f"{col}", min_value=float(vmin), max_value=float(vmax),
                             value=(float(vmin), float(vmax)), step=float(step))
        ten_r = range_slider_for("tenure", step=1)
        mon_r = range_slider_for("MonthlyCharges", step=0.5)
        tot_r = range_slider_for("TotalCharges", step=1.0)

    # Aplicar filtros
    df_f = df.copy()
    if sel_contract:
        df_f = df_f[df_f["Contract"].isin(sel_contract)]
    if sel_internet:
        df_f = df_f[df_f["InternetService"].isin(sel_internet)]
    if sel_payment:
        df_f = df_f[df_f["PaymentMethod"].isin(sel_payment)]
    if sel_gender:
        df_f = df_f[df_f["gender"].isin(sel_gender)]
    if sel_senior:
        df_f = df_f[df_f["SeniorCitizen"].isin(sel_senior)]
    df_f = df_f[df_f["tenure"].between(ten_r[0], ten_r[1])]
    df_f = df_f[df_f["MonthlyCharges"].between(mon_r[0], mon_r[1])]
    df_f = df_f[df_f["TotalCharges"].between(tot_r[0], tot_r[1])]
    if solo_churn:
        df_f = df_f[df_f["Churn"] == 1]

    if df_f.empty:
        st.warning("No hay datos para los filtros seleccionados. Ajusta los filtros.")
        st.stop()

    # KPIs principales
    st.subheader("Métricas principales")
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df_f["Churn"].mean() * 100) if "Churn" in df_f.columns else 0.0
    avg_charge = df_f["MonthlyCharges"].mean()
    avg_tenure = df_f["tenure"].mean()
    total_clients = len(df_f)

    col1.metric("Tasa de Churn", f"{churn_rate:.2f}%")
    col2.metric("Cargos promedio", f"${avg_charge:.2f}")
    col3.metric("Tenure promedio", f"{avg_tenure:.1f} meses")
    col4.metric("Total clientes", f"{total_clients:,}")

    st.markdown("---")

    # Sub-secciones de EDA
    s1, s2, s3, s4 = st.tabs(["Distribuciones", "Comparativo", "Segmentación", "Insights"])

    # Distribuciones
    with s1:
        st.subheader("Histogramas")
        for i in range(0, len(num_vars), 2):
            cols = st.columns(2)
            for j, col in enumerate(num_vars[i:i+2]):
                with cols[j]:
                    chart = alt.Chart(df_f).mark_bar(opacity=0.8).encode(
                        alt.X(col, bin=alt.Bin(maxbins=30)),
                        alt.Y('count()')
                    )
                    st.altair_chart(chart, use_container_width=True)

    # Comparativo
    with s2:
        st.subheader("Relación con la variable objetivo")
        comp_vars = ["Contract", "InternetService", "PaymentMethod"]
        for var in comp_vars:
            if var in df_f.columns:
                chart = alt.Chart(df_f).mark_bar().encode(
                    x=var,
                    y='count()',
                    color='Churn:N'
                )
                st.altair_chart(chart, use_container_width=True)

    # Segmentación
    with s3:
        st.subheader("KPIs por segmento")
        seg_opts = [c for c in cat_vars if c in df_f.columns]
        if seg_opts:
            seg_col = st.selectbox("Segmentar por", seg_opts)
            seg_table = kpis_por_segmento(df_f, seg_col)
            st.dataframe(seg_table)
        else:
            st.info("No hay variables categóricas disponibles para segmentar.")

    # Insights
    with s4:
        st.subheader("Insights automáticos")
        bullets = []
        if "Contract" in df_f.columns:
            gr = df_f.groupby("Contract")["Churn"].mean().sort_values(ascending=False)*100
            bullets.append(f"- Mayor churn en contrato: {gr.index[0]} ({gr.iloc[0]:.2f}%).")
        if "InternetService" in df_f.columns:
            gr = df_f.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)*100
            bullets.append(f"- Mayor churn en servicio de internet: {gr.index[0]} ({gr.iloc[0]:.2f}%).")
        if bullets:
            st.write("\n".join(bullets))
        else:
            st.info("Ajusta los filtros para generar insights.")

# -------------------------------------------------------------------
# ----------- TAB 2: Inference (se llenará después) -----------------
# -------------------------------------------------------------------
with tab2:
    st.subheader("Inference - Predicción de Churn")

    st.write("Complete los datos del cliente para realizar la predicción.")

    # ========= FORMULARIO =========
    with st.form("form_inferencia"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("SeniorCitizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            depend = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (meses)", min_value=0, max_value=72, value=12, step=1)
            monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
            total_charges = st.text_input("TotalCharges (se calculará si está vacío)", "")

        with col2:
            internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
            onsec = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
            techs = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paper = st.selectbox("PaperlessBilling", ["Yes", "No"])
            paym = st.selectbox("PaymentMethod", [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ])

        submitted = st.form_submit_button("Generar predicción")

    # Aquí abajo tú después vas a armar raw_inputs y aplicar transform.py
    if submitted:
        st.write("Formulario enviado correctamente. Ahora puedes aplicar transform.py y el modelo aquí.")

# -------------------------------------------------------------------
# ----------- TAB 3: Dashboard (se llenará después) -----------------
# -------------------------------------------------------------------
with tab3:
    st.subheader("Dashboard")
    st.info("Aquí se incluirá la comparación de métricas, feature importance y confusion matrix.")

# -------------------------------------------------------------------
# ------- TAB 4: Business Impact (se llenará después) ---------------
# -------------------------------------------------------------------
with tab4:
    st.subheader("Business Impact")
    st.info("Aquí se incluirán KPIs de negocio como revenue en riesgo y retención.")
