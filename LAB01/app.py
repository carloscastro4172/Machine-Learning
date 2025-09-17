import streamlit as st
import pandas as pd
import altair as alt
import joblib

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(page_title="EDA - Titanic", layout="wide")
st.markdown("<h1 style='text-align:center;color:#1f4e79;'> Titanic Survival - EDA</h1>", unsafe_allow_html=True)

# ==============================
# CARGA DE DATOS
# ==============================
uploaded_file = st.file_uploader(" Cargar dataset CSV", type=["csv"])

if uploaded_file is None:
    st.info(" Sube un archivo CSV para comenzar el análisis.")
    st.stop()

df = pd.read_csv(uploaded_file)

st.subheader("Vista previa de los datos")
n_rows = st.slider("Número de filas a mostrar", min_value=5, max_value=len(df), value=5, step=5)
st.dataframe(df.head(n_rows))

# ==============================
# SIDEBAR: FILTROS
# ==============================
st.sidebar.header("Filtros de análisis")

# Categóricos
sel_pclass = st.sidebar.multiselect("Clase", sorted(df["Pclass"].unique()), default=sorted(df["Pclass"].unique()))
sel_sex = st.sidebar.multiselect("Sexo", sorted(df["Sex"].unique()), default=sorted(df["Sex"].unique()))
sel_embarked = st.sidebar.multiselect("Puerto de embarque", sorted(df["Embarked"].dropna().unique()), default=sorted(df["Embarked"].dropna().unique()))
solo_survived = st.sidebar.checkbox("Mostrar solo sobrevivientes", value=False)

# Numéricos
age_range = st.sidebar.slider("Edad", float(df["Age"].min()), float(df["Age"].max()), (float(df["Age"].min()), float(df["Age"].max())))
fare_range = st.sidebar.slider("Tarifa (Fare)", float(df["Fare"].min()), float(df["Fare"].max()), (float(df["Fare"].min()), float(df["Fare"].max())))

# Aplicar filtros
df_f = df.copy()
df_f = df_f[df_f["Pclass"].isin(sel_pclass)]
df_f = df_f[df_f["Sex"].isin(sel_sex)]
df_f = df_f[df_f["Embarked"].isin(sel_embarked)]
df_f = df_f[df_f["Age"].between(age_range[0], age_range[1])]
df_f = df_f[df_f["Fare"].between(fare_range[0], fare_range[1])]

if solo_survived:
    df_f = df_f[df_f["Survived"] == 1]

if df_f.empty:
    st.warning(" No hay datos para los filtros seleccionados.")
    st.stop()

# ==============================
# MAPA DE LABELS
# ==============================
label_map = {
    "Age": "Edad",
    "Fare": "Tarifa",
    "Pclass": "Clase",
    "Sex": "Sexo",
    "Embarked": "Puerto de embarque",
    "Survived": "Supervivencia",
    "SibSp": "Hermanos/Esposos a bordo",
    "Parch": "Padres/Hijos a bordo"
}

# ==============================
# KPIs PRINCIPALES
# ==============================
st.markdown("## Métricas Principales")
col1, col2, col3, col4 = st.columns(4)
surv_rate = df_f["Survived"].mean() * 100
avg_age = df_f["Age"].mean()
avg_fare = df_f["Fare"].mean()
total_passengers = len(df_f)

col1.metric("Tasa de Supervivencia", f"{surv_rate:.2f}%")
col2.metric("Edad Promedio", f"{avg_age:.1f} años")
col3.metric("Tarifa Promedio", f"${avg_fare:.2f}")
col4.metric("Total Pasajeros", f"{total_passengers:,}")

st.markdown("---")

# ==============================
# TABS
# ==============================
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Distribuciones", "Comparativo", "Correlaciones", "Insights", "Predicción"])

# ---- Tab 1: Distribuciones
with tab1:
    st.subheader("Distribución de Edad y Tarifa")
    for col in ["Age", "Fare"]:
        chart = alt.Chart(df_f).mark_bar(opacity=0.8).encode(
            alt.X(col, bin=alt.Bin(maxbins=30), title=label_map.get(col, col)),
            alt.Y('count()', title='Frecuencia'),
            tooltip=[col]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

    st.subheader("Boxplot de Edad por Categoría")
    x_cat = st.selectbox("Categoría", ["Pclass", "Sex", "Embarked"], index=0)
    chart = alt.Chart(df_f).mark_boxplot().encode(
        x=alt.X(x_cat, title=label_map.get(x_cat, x_cat)),
        y=alt.Y("Age", title="Edad"),
        color=x_cat
    ).properties(height=400)
    st.altair_chart(chart, use_container_width=True)

# ---- Tab 2: Comparativo
with tab2:
    st.subheader("Supervivencia por categoría")
    for var in ["Pclass", "Sex", "Embarked"]:
        chart = alt.Chart(df_f).mark_bar().encode(
            x=alt.X(var, title=label_map.get(var, var)),
            y=alt.Y("count()", title="Frecuencia"),
            color=alt.Color("Survived:N", scale=alt.Scale(scheme="set1"), title="Supervivencia"),
            tooltip=[var, "Survived"]
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# ---- Tab 3: Correlaciones
with tab3:
    st.subheader("Matriz de Correlación")
    num_vars = ["Age", "Fare", "SibSp", "Parch", "Survived"]
    corr = df_f[num_vars].corr()

    corr_long = corr.reset_index().melt("index")

    base = alt.Chart(corr_long).encode(
        x=alt.X("index:N", title="Variable"),
        y=alt.Y("variable:N", title="Variable")
    )

    rects = base.mark_rect().encode(
        color=alt.Color("value:Q", scale=alt.Scale(scheme="redblue", domain=(-1, 1)), title="Correlación")
    )

    text = base.mark_text(size=14, color="black").encode(
        text=alt.Text("value:Q", format=".0%")
    )

    corr_chart = rects + text
    corr_chart = corr_chart.properties(height=400)

    st.altair_chart(corr_chart, use_container_width=True)

# ---- Tab 4: Insights
with tab4:
    st.subheader("Insights clave")
    bullets = []
    if "Sex" in df_f.columns:
        surv_by_sex = df_f.groupby("Sex")["Survived"].mean() * 100
        bullets.append(f"• Mujeres: {surv_by_sex.get('female', 0):.2f}% supervivencia vs. Hombres: {surv_by_sex.get('male', 0):.2f}%.")
    if "Pclass" in df_f.columns:
        surv_by_class = df_f.groupby("Pclass")["Survived"].mean() * 100
        best_class = surv_by_class.idxmax()
        bullets.append(f"• La mayor supervivencia fue en la clase {best_class} ({surv_by_class[best_class]:.2f}%).")
    if "Age" in df_f.columns:
        young = df_f[df_f["Age"] <= 18]["Survived"].mean() * 100
        adult = df_f[df_f["Age"] > 18]["Survived"].mean() * 100
        bullets.append(f"• Pasajeros jóvenes (≤18) sobrevivieron {young:.2f}% vs. adultos {adult:.2f}%.")
    st.markdown("\n".join(bullets) if bullets else "No se generaron insights.")

# ---- Tab 5: Predicción
# ---- Tab 5: Predicción
with tab5:
    st.subheader("Predicción de Supervivencia (Decision Tree)")

    import joblib

    @st.cache_resource
    def load_model():
        obj = joblib.load("titanic_pipeline.pkl")
        # Soporta casos raros: dicts o GridSearch
        if hasattr(obj, "predict"):
            return obj
        if hasattr(obj, "best_estimator_") and hasattr(obj.best_estimator_, "predict"):
            return obj.best_estimator_
        if isinstance(obj, dict):
            for k in ["pipeline", "model", "clf", "best_estimator_"]:
                if k in obj and hasattr(obj[k], "predict"):
                    return obj[k]
        raise TypeError("El archivo .pkl no contiene un estimador con predict().")

    try:
        pipeline = load_model()
    except Exception as e:
        st.error("No pude cargar `titanic_pipeline.pkl`.")
        st.code(str(e))
        st.stop()

    with st.form("prediction_form"):
        pclass = st.selectbox("Clase", [1, 2, 3], index=0)
        sex = st.selectbox("Sexo", ["male", "female"], index=0)
        age = st.slider("Edad", min_value=0, max_value=90, value=30)
        sibsp = st.slider("Hermanos/Esposos (SibSp)", min_value=0, max_value=10, value=0)
        parch = st.slider("Padres/Hijos (Parch)", min_value=0, max_value=10, value=0)
        fare = st.slider("Tarifa (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
        embarked = st.selectbox("Puerto (Embarked)", ["C", "Q", "S"], index=2)

        submitted = st.form_submit_button("Predecir")

    if submitted:
        new_data = pd.DataFrame([{
            "Pclass": pclass, "Sex": sex, "Age": age,
            "SibSp": sibsp, "Parch": parch, "Fare": fare, "Embarked": embarked
        }])

        pred = pipeline.predict(new_data)[0]
        prob_txt = ""
        if hasattr(pipeline, "predict_proba"):
            prob = pipeline.predict_proba(new_data)[0][1] * 100
            prob_txt = f" (Prob. de sobrevivir: {prob:.2f}%)"

        if pred == 1:
            st.success("✅ El pasajero **sobrevive**" + prob_txt)
        else:
            st.error("❌ El pasajero **no sobrevive**" + (f" (Prob. de NO sobrevivir: {100-prob:.2f}%)" if prob_txt else ""))
