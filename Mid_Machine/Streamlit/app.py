import streamlit as st
import pandas as pd
import altair as alt
import numpy as np 
import sys, io, pickle
import pickle
import os, joblib
import Transformers as T 
from Transformers import DataFramePreparer

from EDA import clean_dt, kpis_por_segmento

# ==============================
# CONFIGURACIÓN DE PÁGINA
# ==============================
st.set_page_config(page_title="Telco Customer Churn", layout="wide")
st.title("Telco Customer Churn")



# ==============================
# TRANSFORMER 
# ==============================
class _RedirectingUnpickler(pickle.Unpickler):
    _MAP = {
        ("__main__", "NumericFeaturizer"): T.NumericFeaturizer,
        ("__main__", "CategoricalServiceCleaner"): T.CategoricalServiceCleaner,
        ("__main__", "CustomOneHotEncoder"): T.CustomOneHotEncoder,
        ("__main__", "DataFramePreparer"): T.DataFramePreparer,
    }
    def find_class(self, module, name):
        mapped = self._MAP.get((module, name))
        if mapped is not None:
            return mapped
        return super().find_class(module, name)

def load_pickle_with_redirect(src):
    # src puede ser ruta (str) o bytes (de file_uploader)
    if isinstance(src, (bytes, bytearray)):
        return _RedirectingUnpickler(io.BytesIO(src)).load()
    with open(src, "rb") as f:
        return _RedirectingUnpickler(f).load()

@st.cache_resource(show_spinner=True)
def load_preparer(preparer_source):
    return load_pickle_with_redirect(preparer_source)
PREPARER_PATH = "/home/carlos/Documents/8vo/MID_MACHINE/Mid_Machine/Streamlit/preparer.pkl"

# ==============================
# MODELOS Y UTILIDADES
# ==============================
MODELS_DIR = "/home/carlos/Documents/8vo/MID_MACHINE/Mid_Machine/models"  

MODEL_REGISTRY = {
    "Voting (full) – no SMOTE": os.path.join(MODELS_DIR, "voting_full_no_smote.pkl"),
    "Voting (full) – SMOTE":    os.path.join(MODELS_DIR, "voting_full_smote.pkl"),
    "Voting (feats) – no SMOTE":os.path.join(MODELS_DIR, "voting_feats_no_smote.pkl"),
    "Voting (feats) – SMOTE":   os.path.join(MODELS_DIR, "voting_feats_smote.pkl"),
    "Voting (PCA) – no SMOTE":  os.path.join(MODELS_DIR, "voting_pca_no_smote.pkl"),
    "Voting (PCA) – SMOTE":     os.path.join(MODELS_DIR, "voting_pca_smote.pkl"),
}

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    obj = joblib.load(path)   # <--- usa joblib como en tu guardado
    return obj

@st.cache_resource(show_spinner=True)
def load_all_models():
    loaded = {}
    for name, path in MODEL_REGISTRY.items():
        try:
            loaded[name] = load_model(path)
        except Exception as e:
            st.warning(f"No se pudo cargar '{name}' desde {path}: {e}")
    return loaded

def compute_total_charges(tenure, monthly):
    try:
        return float(tenure) * float(monthly)
    except Exception:
        return 0.0

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
# ---------------------------- TAB 2: Inference ---------------------
# -------------------------------------------------------------------
with tab2:
    st.subheader("Inference - Predicción de Churn")

    # ========== Helpers para bundles ==========
    def unwrap_bundle(obj):
        """
        Acepta:
          - dict bundle (recomendado)
          - estimador directo (por compatibilidad)
        Devuelve (bundle_dict, estimator)
        """
        if isinstance(obj, dict):
            est = obj.get("model", None)
            return obj, est
        # compatibilidad si alguien cargó solo el estimador
        return {
            "model": obj,
            "scenario": "unknown",
            "threshold": 0.5,
            "feature_names": None,
            "preprocessor": None,
            "classes_": getattr(obj, "classes_", None)
        }, obj

    def get_pos_index(bundle):
        """
        Índice de la clase positiva en predict_proba.
        Si hay clases y 'Yes' está, usamos ese índice; si no, 1.
        """
        classes_ = bundle.get("classes_", None)
        if classes_ is not None:
            classes_ = list(classes_)
            if "Yes" in classes_:
                return classes_.index("Yes")
        return 1

    def build_features_for_bundle(X_raw, preparer, bundle):
        """
        Flujo:
          - Siempre aplicamos preparer.transform(X_raw) -> X_full
          - Si scenario usa 'pca', aplicar preprocessor PCA: X_pca
          - Si hay 'feature_names', alinear columnas a ese orden
        """
        scenario = str(bundle.get("scenario", "unknown")).lower()
        feat_names = bundle.get("feature_names", None)
        preproc = bundle.get("preprocessor", None)

        # 1) features completas del preparador
        X_full = preparer.transform(X_raw)

        # 2) PCA (si aplica)
        if "pca" in scenario or ("pca" in str(preproc).lower() if preproc is not None else False):
            if preproc is None or not hasattr(preproc, "transform"):
                raise RuntimeError("El bundle indica PCA pero no trae 'preprocessor' válido.")
            X_b = preproc.transform(X_full)
            # salida PCA es ndarray (sin nombres)
            return X_b

        # 3) Selected (alinear al orden esperado)
        if feat_names is not None:
            # Añadir columnas faltantes con 0 y reordenar
            missing = [c for c in feat_names if c not in X_full.columns]
            if missing:
                for c in missing:
                    X_full[c] = 0
            X_b = X_full.reindex(columns=feat_names)
            return X_b

        # 4) Full (sin PCA): usar X_full tal cual
        return X_full

    def apply_threshold(p_pos, thr):
        return int(p_pos >= float(thr))

    # ========== Cargar transformer y modelos ==========
    try:
        preparer = load_preparer(PREPARER_PATH)
    except Exception as e:
        st.error(f"No se pudo cargar el transformer desde {PREPARER_PATH}. "
                 f"Asegúrate de exportar tu DataFramePreparer ya 'fit'. Detalle: {e}")
        st.stop()

    raw_models = load_all_models()
    if not raw_models:
        st.error("No se cargó ningún modelo. Verifica rutas en MODEL_REGISTRY.")
        st.stop()

    # desempaquetar bundles / validarlos
    bundles = {}
    bad = []
    for name, obj in raw_models.items():
        bndl, est = unwrap_bundle(obj)
        if est is None or not hasattr(est, "predict"):
            bad.append(name)
        else:
            bundles[name] = bndl
    if bad:
        with st.expander("Modelos que no son bundles válidos (o sin `.predict`):"):
            for nm in bad:
                st.warning(f"- {nm}")

    if not bundles:
        st.error("Ningún .pkl cargado contiene un bundle/estimador utilizable.")
        st.stop()

    model_name = st.selectbox("Selecciona el modelo", list(bundles.keys()))
    bundle = bundles[model_name]
    estimator = bundle["model"]
    thr = bundle.get("threshold", 0.5)
    pos_idx = get_pos_index(bundle)

    st.caption(f"Escenario: {bundle.get('scenario','?')} • Threshold F1-opt: {thr:.3f}")

    st.write("Complete los datos del cliente para realizar la predicción.")

    # ========= FORMULARIO =========
    with st.form("form_inferencia"):
        col1, col2 = st.columns(2)

        # -------- Columna izquierda --------
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("SeniorCitizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            depend = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (meses)", min_value=0, max_value=72, value=12, step=1)
            monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
            total_charges = st.text_input("TotalCharges (se calculará si está vacío)", "")

        # -------- Columna derecha (incluye campos que el preparer espera) --------
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

            phone = st.selectbox("PhoneService", ["Yes", "No"])
            if phone == "No":
                mult = "No phone service"
                st.caption("MultipleLines = No phone service (derivado de PhoneService = No)")
            else:
                mult = st.selectbox("MultipleLines", ["Yes", "No"])

            online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
            device_prot   = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
            stream_tv     = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
            stream_mov    = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

        submitted = st.form_submit_button("Generar predicción")

    if submitted:
        # 1) Inputs crudos
        if total_charges.strip() == "":
            total_val = compute_total_charges(tenure, monthly)
        else:
            try:
                total_val = float(total_charges)
            except Exception:
                total_val = compute_total_charges(tenure, monthly)

        raw = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": depend,
            "tenure": float(tenure),
            "MonthlyCharges": float(monthly),
            "TotalCharges": float(total_val),
            "InternetService": internet,
            "OnlineSecurity": onsec,
            "TechSupport": techs,
            "Contract": contract,
            "PaperlessBilling": paper,
            "PaymentMethod": paym,
            "PhoneService": phone,
            "MultipleLines": mult,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_prot,
            "StreamingTV": stream_tv,
            "StreamingMovies": stream_mov,
        }
        X_raw = pd.DataFrame([raw])

        # (opcional) blindaje si faltara alguna columna cruda esperada por el preparer
        exp_num = list(getattr(preparer, "_num_cols", []))
        exp_cat = list(getattr(preparer, "_cat_cols", []))
        exp_raw = set(exp_num + exp_cat)
        miss = [c for c in exp_raw if c not in X_raw.columns]
        if miss:
            for c in miss:
                if c in exp_num:
                    X_raw[c] = 0
                else:
                    if c == "MultipleLines" and ("PhoneService" in X_raw.columns) and (X_raw["PhoneService"].iloc[0] == "No"):
                        X_raw[c] = "No phone service"
                    else:
                        X_raw[c] = "No"

        # 2) Construir features según el tipo de bundle
        try:
            X_feat = build_features_for_bundle(X_raw, preparer, bundle)
        except Exception as e:
            st.error(f"Error construyendo features para el bundle '{bundle.get('scenario','?')}': {e}")
            st.stop()

        # 3) Probabilidades y clase usando threshold del bundle
        try:
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X_feat)
                p_yes = float(proba[:, pos_idx][0])
            elif hasattr(estimator, "decision_function"):
                score = estimator.decision_function(X_feat)
                # pasar a (0,1) con sigmoide
                s = float(score[0])
                p_yes = 1.0 / (1.0 + np.exp(-s))
            else:
                # como último recurso, usar predict y castear
                pred_raw = estimator.predict(X_feat)
                p_yes = float(pred_raw[0]) if isinstance(pred_raw[0], (int, float)) else (1.0 if str(pred_raw[0]).lower() in ("1","yes","true") else 0.0)

            y_hat = apply_threshold(p_yes, thr)
        except Exception as e:
            st.error(f"Error al predecir con el estimador del bundle '{model_name}': {e}")
            st.stop()

        # 4) Resultados
        c1, c2 = st.columns(2)
        c1.metric("Predicted class", "Churn" if y_hat == 1 else "No Churn")
        c2.metric("Probability (churn=1)", f"{100*p_yes:.2f}%")

        # 5) Reglas de acción
        if p_yes >= 0.70:
            st.warning("Alto riesgo de churn. Sugerencia: ofrecer descuento o servicio adicional.")
        elif p_yes >= 0.40:
            st.info("Riesgo medio. Recomendación: campaña de fidelización.")
        else:
            st.success("Riesgo bajo. Mantener comunicación estándar.")


# -------------------------------------------------------------------
# ---------------------------- TAB 3: Dashboard ---------------------
# -------------------------------------------------------------------
with tab3:
    st.subheader("Dashboard")

    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_auc_score, confusion_matrix,
        precision_score, recall_score
    )
    import altair as alt

    if "Churn" not in df.columns:
        st.error("El dataset no tiene la columna 'Churn'. Sube un CSV con la etiqueta para evaluar el dashboard.")
        st.stop()

    # ---------------- Helpers (reutiliza los tuyos) ----------------
    def unwrap_bundle(obj):
        if isinstance(obj, dict):
            est = obj.get("model", None)
            return obj, est
        return {
            "model": obj,
            "scenario": "unknown",
            "threshold": 0.5,
            "feature_names": None,
            "preprocessor": None,
            "classes_": getattr(obj, "classes_", None)
        }, obj

    def get_pos_index(bundle):
        classes_ = bundle.get("classes_", None)
        if classes_ is not None:
            classes_ = list(classes_)
            if "Yes" in classes_:
                return classes_.index("Yes")
        return 1

    def build_features_for_bundle(X_raw, preparer, bundle):
        scenario = str(bundle.get("scenario", "unknown")).lower()
        feat_names = bundle.get("feature_names", None)
        preproc = bundle.get("preprocessor", None)

        X_full = preparer.transform(X_raw)

        if "pca" in scenario or (preproc is not None and hasattr(preproc, "transform")):
            if preproc is None or not hasattr(preproc, "transform"):
                raise RuntimeError("El bundle indica PCA pero no trae 'preprocessor' válido.")
            X_pca = preproc.transform(X_full)
            return X_pca, [f"PC{i+1}" for i in range(X_pca.shape[1])]

        if feat_names is not None:
            missing = [c for c in feat_names if c not in X_full.columns]
            for c in missing:
                X_full[c] = 0
            X_sel = X_full.reindex(columns=feat_names)
            return X_sel, list(X_sel.columns)

        return X_full, list(X_full.columns)

    def to_binary_series(y):
        if pd.api.types.is_numeric_dtype(y):
            return y.astype(int).values
        y2 = y.astype(str).str.strip().str.lower().map({"yes":1,"1":1,"true":1,"no":0,"0":0,"false":0})
        return y2.fillna(0).astype(int).values

    def predict_proba_yes(estimator, X, pos_idx):
        if hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(X)
            return proba[:, pos_idx]
        elif hasattr(estimator, "decision_function"):
            s = estimator.decision_function(X)
            s = np.array(s).reshape(-1)
            return 1.0 / (1.0 + np.exp(-s))
        else:
            pred = estimator.predict(X)
            if isinstance(pred[0], (int, float, np.integer, np.floating)):
                return np.clip(np.array(pred, dtype=float), 0.0, 1.0)
            return pd.Series(pred).astype(str).str.lower().map({"yes":1.0,"1":1.0,"true":1.0}).fillna(0.0).values

    def get_feature_importance(estimator, feature_names):
        if hasattr(estimator, "feature_importances_"):
            vals = estimator.feature_importances_
            return pd.DataFrame({"feature": feature_names[:len(vals)], "importance": vals})
        if hasattr(estimator, "coef_"):
            coef = estimator.coef_
            vals = np.abs(coef) if coef.ndim == 1 else np.linalg.norm(coef, axis=0)
            return pd.DataFrame({"feature": feature_names[:len(vals)], "importance": vals})
        return None

    # ---------------- Cargar preparer + modelos ----------------
    try:
        preparer = load_preparer(PREPARER_PATH)
    except Exception as e:
        st.error(f"No se pudo cargar el transformer (preparer): {e}")
        st.stop()

    raw_models = load_all_models()
    if not raw_models:
        st.error("No se cargó ningún modelo.")
        st.stop()

    bundles = {}
    for name, obj in raw_models.items():
        b, est = unwrap_bundle(obj)
        if est is not None and hasattr(est, "predict"):
            bundles[name] = b

    if not bundles:
        st.error("Ningún .pkl contiene un estimador utilizable.")
        st.stop()

    # ---------------- UI de evaluación ----------------
    st.markdown("#### Configuración de evaluación")
    cols_top = st.columns([2,1,1])
    with cols_top[0]:
        sel_models = st.multiselect(
            "Modelos a comparar",
            list(bundles.keys()),
            default=list(bundles.keys())[:3]
        )
    with cols_top[1]:
        use_bundle_thr = st.checkbox("Usar threshold del bundle", value=True)
    with cols_top[2]:
        custom_thr = st.slider("Umbral personalizado", 0.0, 1.0, 0.5, 0.01, disabled=use_bundle_thr)

    if not sel_models:
        st.info("Selecciona al menos un modelo para comparar.")
        st.stop()

    # ---------------- Preparar X_raw e y_true ----------------
    y_true = to_binary_series(df["Churn"])

    exp_num = list(getattr(preparer, "_num_cols", []))
    exp_cat = list(getattr(preparer, "_cat_cols", []))
    needed_cols = list(dict.fromkeys(exp_num + exp_cat))

    missing_cols = [c for c in needed_cols if c not in df.columns]
    if missing_cols:
        st.warning(f"Faltan columnas en el CSV para transformar: {missing_cols}. Se completan con defaults.")
    X_raw = df.copy()
    for c in missing_cols:
        if c in exp_num:
            X_raw[c] = 0
        else:
            X_raw[c] = "No"

    # ---------------- Evaluación modelos -> res_df ----------------
    results = []
    preds_cache = {}
    for name in sel_models:
        bundle = bundles[name]
        est = bundle["model"]
        pos_idx = get_pos_index(bundle)
        try:
            X_feat, feat_names = build_features_for_bundle(X_raw, preparer, bundle)
            p_yes = predict_proba_yes(est, X_feat, pos_idx)
            thr = float(bundle.get("threshold", 0.5)) if use_bundle_thr else float(custom_thr)
            y_pred = (p_yes >= thr).astype(int)

            acc = accuracy_score(y_true, y_pred)
            f1  = f1_score(y_true, y_pred, zero_division=0)
            try:
                auc = roc_auc_score(y_true, p_yes)
            except Exception:
                auc = np.nan

            results.append({
                "Model": name,
                "Scenario": bundle.get("scenario", "?"),
                "Threshold": thr,
                "Accuracy": acc,
                "F1": f1,
                "AUC": auc
            })
            preds_cache[name] = (y_pred, p_yes, feat_names, est, bundle)
        except Exception as e:
            st.error(f"Error evaluando {name}: {e}")

    if not results:
        st.stop()

    res_df = pd.DataFrame(results).sort_values("F1", ascending=False)

    # ---------------- Tabla formateada (sin Styler) ----------------
    st.markdown("### Comparación de métricas")
    res_df_fmt = res_df.copy()
    for c, fmt in [("Accuracy","{:.3f}"),("F1","{:.3f}"),("AUC","{:.3f}"),("Threshold","{:.2f}")]:
        if c in res_df_fmt.columns:
            res_df_fmt[c] = res_df_fmt[c].apply(lambda v: "" if pd.isna(v) else fmt.format(v))
    st.dataframe(res_df_fmt, width='stretch')

    # ---------------- Barras F1 ----------------
    order_models = res_df.sort_values('F1', ascending=False)['Model'].tolist()
    chart_f1 = (
        alt.Chart(res_df)
        .mark_bar()
        .encode(
            x=alt.X('Model:N', sort=order_models, title="Modelo"),
            y=alt.Y('F1:Q', title='F1'),
            color=alt.Color('Scenario:N', title='Escenario'),
            tooltip=['Model','Scenario',
                     alt.Tooltip('Accuracy:Q', format=".3f"),
                     alt.Tooltip('F1:Q', format=".3f"),
                     alt.Tooltip('AUC:Q', format=".3f"),
                     alt.Tooltip('Threshold:Q', format=".2f")]
        )
        .properties(height=280)
    )
    st.altair_chart(chart_f1, use_container_width=True, theme=None)

    # ---------------- Matriz de confusión ----------------
    best_name = res_df.iloc[0]["Model"]
    y_pred_best, p_best, feat_names_best, est_best, bundle_best = preds_cache[best_name]
    cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])  # [[TN, FP],[FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    precision = precision_score(y_true, y_pred_best, zero_division=0)
    recall    = recall_score(y_true, y_pred_best, zero_division=0)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

    st.markdown(f"### Matriz de confusión – Mejor F1: **{best_name}**")

    cols_metrics = st.columns(5)
    cols_metrics[0].metric("Precision", f"{precision:.3f}")
    cols_metrics[1].metric("Recall (TPR)", f"{recall:.3f}")
    cols_metrics[2].metric("Specificity (TNR)", f"{specificity:.3f}")
    cols_metrics[3].metric("FPR", f"{fpr:.3f}")
    cols_metrics[4].metric("FNR", f"{fnr:.3f}")

    normalize = st.checkbox("Normalizar por fila (%)", value=True, key="cm_norm")

    # Preparar data
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])\
        .reset_index().rename(columns={"index":"Actual"})
    cm_long = cm_df.melt(id_vars="Actual", var_name="Pred", value_name="count")
    row_totals = cm_long.groupby("Actual")["count"].transform("sum")
    cm_long["row_pct"] = np.where(row_totals>0, cm_long["count"]/row_totals, 0.0)
    cm_long["text"] = cm_long.apply(lambda r: f"{r['count']}\n({r['row_pct']*100:.1f}%)", axis=1)

    value_field = "row_pct" if normalize else "count"
    value_title = "Porcentaje" if normalize else "Conteo"

    conf_matrix = (
        alt.Chart(cm_long)
        .mark_rect()
        .encode(
            x=alt.X("Pred:N", title="Predicción"),
            y=alt.Y("Actual:N", title="Valor real"),
            color=alt.Color(f"{value_field}:Q", title=value_title, scale=alt.Scale(scheme="blues")),
            tooltip=["Actual", "Pred", "count", alt.Tooltip("row_pct:Q", format=".1%")]
        )
        .properties(width=550, height=550)  
    )

    text = (
        alt.Chart(cm_long)
        .mark_text(fontSize=12, fontWeight="bold")
        .encode(
            x="Pred:N",
            y="Actual:N",
            text="text",
            color=alt.condition("datum.row_pct > 0.5", alt.value("white"), alt.value("black"))
        )
    )

    st.altair_chart(conf_matrix + text, use_container_width=False)

    # ---------------- Feature importance (rápida, agrupada y cacheada) ----------------
    st.markdown("### Feature Importance (modo rápido)")

    # Controles de rapidez
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_sample = st.slider("Muestra para importancia", 200, min(3000, len(df)), 800, 100)
    with c2:
        n_repeats = st.slider("Repeticiones", 1, 10, 3, 1)
    with c3:
        top_k_groups = st.slider("Top-K grupos a evaluar", 5, 40, 15, 1)
    with c4:
        scoring_choice = st.selectbox("Métrica para permutar", ["roc_auc", "f1", "neg_log_loss"], index=0)

    # 1) Reconstruir features y nombres del mejor bundle
    try:
        X_feat_best, feat_names_best = build_features_for_bundle(X_raw, preparer, bundle_best)
    except Exception:
        X_feat_best, feat_names_best = None, feat_names_best

    # Asegurar nombres
    def _fallback_feature_names(X, names):
        if names is not None:
            return list(names)
        if hasattr(X, "shape"):
            return [f"f{i}" for i in range(X.shape[1])]
        return []

    feat_names_best = _fallback_feature_names(X_feat_best, feat_names_best)
    pos_idx_best = get_pos_index(bundle_best)

    # 2) Agrupar columnas por "feature original"
    # Intentamos inferir el nombre original a partir del patrón común de One-Hot:
    #  - "feature=value"  -> grupo "feature"
    #  - "feature__value" -> grupo "feature"
    #  - "feature_value"  -> grupo "feature" (si hay diversidad de sufijos)
    import re
    def guess_group(name: str) -> str:
        # prioridad 1: "feature=value"
        if "=" in name:
            return name.split("=")[0]
        # prioridad 2: "feature__value"
        if "__" in name:
            return name.split("__")[0]
        # prioridad 3: "feature_value" (si parece categoría, no numérica)
        m = re.match(r"([A-Za-z][A-Za-z0-9]+)[_\|].+", name)
        if m:
            return m.group(1)
        return name  # si no se identifica, queda tal cual

    groups = {}
    for j, fname in enumerate(feat_names_best):
        g = guess_group(str(fname))
        groups.setdefault(g, []).append(j)

    # 3) Elegir una muestra (para acelerar) y preparar X / y
    rng = np.random.RandomState(42)
    n_total = len(y_true)
    n_sample = min(n_sample, n_total)
    idx = rng.choice(n_total, size=n_sample, replace=False)

    def _slice_X(X, idx):
        try:
            return X[idx]
        except Exception:
            return X.iloc[idx]

    Xp = _slice_X(X_feat_best, idx)
    yp = np.asarray(y_true)[idx]

    # 4) Heurística para seleccionar los Top-K grupos candidatos (reduce trabajo)
    #    - Para numéricos: alta varianza
    #    - Para dummies: alta prevalencia promedio
    import numpy as np
    from scipy import sparse

    def group_score_fast(X, cols):
        # estimador barato del "potencial" del grupo para priorizar
        # si sparse, usamos medias con operaciones vectorizadas
        if sparse.issparse(X):
            sub = X[:, cols]
            # proxy: densidad/media de la columna (si es dummy) o energía promedio
            m = np.asarray(sub.mean(axis=0)).ravel()
            v = m * (1 - m)  # varianza binaria aprox
            return float(np.nanmean(v)) if len(v) else 0.0
        else:
            sub = np.asarray(X)[:, cols]
            v = np.var(sub, axis=0)
            return float(np.nanmean(v)) if v.size else 0.0

    group_candidates = []
    for g, cols in groups.items():
        group_candidates.append((g, cols, group_score_fast(Xp, cols)))

    # ordenar por score descendente y tomar top-K
    group_candidates.sort(key=lambda t: t[2], reverse=True)
    group_candidates = group_candidates[:top_k_groups]

    # 5) Cachear la importancia para no recalcular si no cambian inputs
    @st.cache_data(show_spinner=True)
    def fast_group_permutation_importance_cached(
        model_name, scoring, n_repeats, idx_seed, y, group_list, feat_names, X_block, is_sparse
    ):
        """
        Calcula permutation importance por GRUPOS de columnas (más rápido).
        Retorna DataFrame con importancia media por grupo.
        """
        from sklearn.metrics import roc_auc_score, f1_score, log_loss

        rng_local = np.random.RandomState(idx_seed)

        # Funciones de probas y score
        def predict_proba_pos(Xloc):
            if hasattr(est_best, "predict_proba"):
                return est_best.predict_proba(Xloc)[:, pos_idx_best]
            elif hasattr(est_best, "decision_function"):
                s = est_best.decision_function(Xloc).ravel()
                return 1.0 / (1.0 + np.exp(-s))
            else:
                # 0/1
                return (est_best.predict(Xloc).astype(float)).clip(0, 1)

        def scorer(y_true_loc, p_loc):
            if scoring == "roc_auc":
                try:
                    return roc_auc_score(y_true_loc, p_loc)
                except Exception:
                    return np.nan
            elif scoring == "f1":
                y_pred_loc = (p_loc >= 0.5).astype(int)
                return f1_score(y_true_loc, y_pred_loc, zero_division=0)
            elif scoring == "neg_log_loss":
                try:
                    return -log_loss(y_true_loc, p_loc, labels=[0,1])
                except Exception:
                    return np.nan
            else:
                # por defecto AUC
                try:
                    return roc_auc_score(y_true_loc, p_loc)
                except Exception:
                    return np.nan

        # Predicción base
        p_base = predict_proba_pos(X_block)
        base_score = scorer(y, p_base)

        importances = []
        for g_name, cols, _gscore in group_list:
            scores = []
            for _ in range(n_repeats):
                if is_sparse:
                    # Copia en CSR y permuta columnas del grupo
                    Xperm = X_block.copy().tocsr()
                    # Para permutar en sparse de forma rápida: permutamos filas del sub-bloque columna a columna
                    # (más eficiente que densificar todo)
                    for c in cols:
                        col = Xperm.getcol(c).toarray().ravel()
                        rng_local.shuffle(col)
                        Xperm.data[Xperm.indptr[0]:]  # placeholder no usado; reconstruimos columna
                        # reinsertar columna permutada (convertir a formato COO para asignar rápido)
                        # Nota: la asignación directa en CSR a una columna es costosa; por rapidez total del loop,
                        # densificamos SOLO esas columnas y las reescribimos:
                        Xperm = Xperm.tolil()
                        Xperm[:, c] = col.reshape(-1, 1)
                        Xperm = Xperm.tocsr()
                else:
                    Xperm = np.asarray(X_block).copy()
                    # permutar todas las columnas del grupo a la vez (barato)
                    for c in cols:
                        rng_local.shuffle(Xperm[:, c])

                p = predict_proba_pos(Xperm)
                scores.append(scorer(y, p))

            # Δscore positivo = mayor importancia
            delta = np.nanmean(base_score - np.array(scores))
            importances.append((g_name, delta))

        imp_df_grp = (
            pd.DataFrame(importances, columns=["group", "importance"])
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )
        return imp_df_grp

    # 6) Preparar Xp compacto (evitar densificar todo)
    is_sparse = sparse.issparse(Xp)
    Xp_compact = Xp  # lo pasamos tal cual; la función maneja sparse/ndarray

    imp_grp_df = fast_group_permutation_importance_cached(
        model_name=str(best_name),
        scoring=scoring_choice,
        n_repeats=int(n_repeats),
        idx_seed=42,
        y=yp,
        group_list=group_candidates,
        feat_names=feat_names_best,
        X_block=Xp_compact,
        is_sparse=is_sparse
    )

    if imp_grp_df is None or imp_grp_df.empty:
        st.info("No fue posible calcular importancias rápidas (revisa tamaño de muestra o compatibilidad).")
    else:
        # Mostrar Top-K grupos más importantes
        chart_imp_grp = (
            alt.Chart(imp_grp_df.head(top_k_groups))
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title=f"Δ{scoring_choice} (↑ = mayor impacto)"),
                y=alt.Y("group:N", sort='-x', title="Feature original (grupo)"),
                tooltip=["group", alt.Tooltip("importance:Q", format=".5f")]
            )
            .properties(height=380)
        )
        st.altair_chart(chart_imp_grp, use_container_width=True)


# -------------------------------------------------------------------
# ------- TAB 4: Business Impact (se llenará después) ---------------
# -------------------------------------------------------------------
with tab4:
    st.subheader("Business Impact")
    st.info("Aquí se incluirán KPIs de negocio como revenue en riesgo y retención.")
