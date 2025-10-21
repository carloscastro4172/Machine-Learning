import streamlit as st
import pandas as pd
import numpy as np
from utils.loader import load_all_models, load_preparer

# =====================================================
# 🧠 PAGE TITLE
# =====================================================
st.set_page_config(page_title="Inference & Prediction", layout="wide")
st.title("Inference & Prediction")

# =====================================================
# 🔍 Check if necessary components are loaded
# =====================================================
if "models" not in st.session_state or "preparer" not in st.session_state:
    st.error("Models or preparer not loaded. Please return to the Home page.")
    st.stop()

preparer = st.session_state["preparer"]
models = st.session_state["models"]

# =====================================================
# 🧩 ADVANCED INFERENCE – MANUAL FORM
# =====================================================
st.header("Manual Customer Prediction")

st.markdown("""
Use this section to perform **manual churn prediction** for any customer scenario.  
You can choose a model, enter the customer details, and the system will estimate churn probability
and suggest retention actions.
""")

# ======== Helper functions ========
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

    if "pca" in scenario or ("pca" in str(preproc).lower() if preproc is not None else False):
        if preproc is None or not hasattr(preproc, "transform"):
            raise RuntimeError("Bundle indicates PCA but has no valid 'preprocessor'.")
        return preproc.transform(X_full)

    if feat_names is not None:
        missing = [c for c in feat_names if c not in X_full.columns]
        for c in missing:
            X_full[c] = 0
        return X_full.reindex(columns=feat_names)

    return X_full

def apply_threshold(p_pos, thr):
    return int(p_pos >= float(thr))

def compute_total_charges(tenure, monthly):
    try:
        return float(tenure) * float(monthly)
    except Exception:
        return 0.0


# ======== Prepare valid bundles ========
bundles = {}
bad = []
for name, obj in models.items():
    bndl, est = unwrap_bundle(obj)
    if est is None or not hasattr(est, "predict"):
        bad.append(name)
    else:
        bundles[name] = bndl

if bad:
    with st.expander("Invalid or incomplete model bundles:"):
        for nm in bad:
            st.warning(f"- {nm}")

if not bundles:
    st.error("No valid models available. Check your model .pkl files.")
    st.stop()

model_name = st.selectbox("Select model", list(bundles.keys()))
bundle = bundles[model_name]
estimator = bundle["model"]
thr = bundle.get("threshold", 0.5)
pos_idx = get_pos_index(bundle)

st.caption(f"Scenario: {bundle.get('scenario','?')} • Threshold (F1-opt): {thr:.3f}")

st.markdown("### Fill Customer Details")

# ========= MANUAL FORM =========
with st.form("manual_inference_form"):
    col1, col2 = st.columns(2)

    # -------- Left Column --------
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("SeniorCitizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        depend = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
        monthly = st.number_input("MonthlyCharges", min_value=0.0, max_value=150.0, value=70.0, step=1.0)
        total_charges = st.text_input("TotalCharges (auto-calculated if empty)", "")

    # -------- Right Column --------
    with col2:
        internet = st.selectbox("InternetService", ["DSL", "Fiber optic", "No"])
        onsec = st.selectbox("OnlineSecurity", ["Yes", "No", "No internet service"])
        techs = st.selectbox("TechSupport", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paper = st.selectbox("PaperlessBilling", ["Yes", "No"])
        paym = st.selectbox(
            "PaymentMethod",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        phone = st.selectbox("PhoneService", ["Yes", "No"])
        if phone == "No":
            mult = "No phone service"
            st.caption("MultipleLines = No phone service (derived from PhoneService = No)")
        else:
            mult = st.selectbox("MultipleLines", ["Yes", "No"])

        online_backup = st.selectbox("OnlineBackup", ["Yes", "No", "No internet service"])
        device_prot = st.selectbox("DeviceProtection", ["Yes", "No", "No internet service"])
        stream_tv = st.selectbox("StreamingTV", ["Yes", "No", "No internet service"])
        stream_mov = st.selectbox("StreamingMovies", ["Yes", "No", "No internet service"])

    submitted = st.form_submit_button("Generate Prediction")

# ========= HANDLE SUBMIT =========
if submitted:
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

    exp_num = list(getattr(preparer, "_num_cols", []))
    exp_cat = list(getattr(preparer, "_cat_cols", []))
    exp_raw = set(exp_num + exp_cat)
    miss = [c for c in exp_raw if c not in X_raw.columns]
    for c in miss:
        if c in exp_num:
            X_raw[c] = 0
        else:
            if c == "MultipleLines" and X_raw["PhoneService"].iloc[0] == "No":
                X_raw[c] = "No phone service"
            else:
                X_raw[c] = "No"

    # Generate prediction
    try:
        X_feat = build_features_for_bundle(X_raw, preparer, bundle)
        if hasattr(estimator, "predict_proba"):
            p_yes = float(estimator.predict_proba(X_feat)[:, pos_idx][0])
        elif hasattr(estimator, "decision_function"):
            score = float(estimator.decision_function(X_feat)[0])
            p_yes = 1.0 / (1.0 + np.exp(-score))
        else:
            pred_raw = estimator.predict(X_feat)
            p_yes = float(pred_raw[0]) if isinstance(pred_raw[0], (int, float)) else (
                1.0 if str(pred_raw[0]).lower() in ("1", "yes", "true") else 0.0
            )
        y_hat = apply_threshold(p_yes, thr)
    except Exception as e:
        st.error(f"Prediction error for '{model_name}': {e}")
        st.stop()

    # ===== Results =====
    c1, c2 = st.columns(2)
    c1.metric("Predicted Class", "Churn" if y_hat == 1 else "No Churn")
    c2.metric("Probability (churn=1)", f"{100*p_yes:.2f}%")

    # ===== Recommendations =====
    if p_yes >= 0.70:
        st.warning("⚠️ High churn risk – recommend offering discounts or personalized retention offers.")
    elif p_yes >= 0.40:
        st.info("Medium churn risk – consider customer engagement or satisfaction surveys.")
    else:
        st.success("✅ Low churn risk – maintain standard retention strategies.")
