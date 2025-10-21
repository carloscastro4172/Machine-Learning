import streamlit as st
import pandas as pd
import numpy as np

def run_inference_form(preparer, models):
    st.subheader("Customer Data Input")

    # Form input
    with st.form("inference_form"):
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges", 0.0, 150.0, 70.0)
        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    # Simple example
    X_raw = pd.DataFrame([{"gender": gender, "tenure": tenure, "MonthlyCharges": monthly}])
    model_name = list(models.keys())[0]
    estimator = models[model_name]["model"]

    X_feat = preparer.transform(X_raw)
    prob = estimator.predict_proba(X_feat)[0][1]
    y_hat = int(prob >= 0.5)

    st.metric("Predicted", "Churn" if y_hat else "No Churn")
    st.metric("Probability", f"{prob*100:.2f}%")
