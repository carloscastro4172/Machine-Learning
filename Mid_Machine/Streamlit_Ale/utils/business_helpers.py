import pandas as pd
import numpy as np
import streamlit as st

# ============================================================
# 🧠 MODEL SELECTION
# ============================================================

def get_best_model(models_dict):
    """
    Safely returns the best model object available in the app session.
    Priority:
    1. Model stored as st.session_state["best_model"]
    2. Model with 'best' in its name
    3. First valid model in the dict
    """
    # 1️⃣ Use directly stored best model (Dashboard already saves it)
    if "best_model" in st.session_state:
        return st.session_state["best_model"]

    # 2️⃣ Try to find one labeled as 'best' in its name
    if isinstance(models_dict, dict):
        for name, model in models_dict.items():
            if "best" in name.lower():
                return model.get("model", model)

        # 3️⃣ Fallback to first available
        if len(models_dict) > 0:
            first = list(models_dict.values())[0]
            return first.get("model", first)

    raise ValueError("❌ No valid models available to select the best one.")


# ============================================================
# 🧩 DATA PREPARATION AND PREDICTION
# ============================================================

def prepare_business_data(df, preparer, models):
    """
    Transforms the dataset while preserving customerID and
    generates churn probabilities using the best available model.
    """
    # ✅ Check for required ID column
    if "customerID" not in df.columns:
        raise ValueError("The dataset must contain 'customerID' for Business Impact.")

    # ✅ Keep IDs safe
    ids = df["customerID"].copy().reset_index(drop=True)

    # ✅ Transform features
    X = preparer.transform(df)

    # ✅ Get best model (handles dicts or single models)
    if isinstance(models, dict):
        best_model = get_best_model(models)
    else:
        best_model = models

    # ✅ Predict churn probability safely
    probs = best_model.predict_proba(np.asarray(X))[:, 1]

    # ✅ Return combined DataFrame
    df_out = pd.DataFrame({
        "customerID": ids,
        "churn_probability": probs
    })

    return df_out


# ============================================================
# 💬 RECOMMENDATION SYSTEM
# ============================================================

def recommend_action(churn_probability, threshold_high=0.7, threshold_mid=0.4):
    """
    Generates a simple business recommendation based on churn probability.
    """
    if churn_probability >= threshold_high:
        return (
            "⚠️ **High churn risk** — Offer a personalized discount, "
            "loyalty bonus, or proactive call to retain the customer."
        )

    elif churn_probability >= threshold_mid:
        return (
            "🟠 **Medium churn risk** — Keep the customer engaged via "
            "email campaigns, rewards, or additional service offers."
        )

    else:
        return (
            "🟢 **Low churn risk** — Customer is stable. Maintain service quality "
            "and customer satisfaction."
        )


# ============================================================
# 🧾 KPI CALCULATION (OPTIONAL)
# ============================================================

def compute_business_kpis(df, prob_col="churn_probability", charge_col="MonthlyCharges"):
    """
    Computes core KPIs: churn rate, revenue at risk, potential savings.
    Returns a tuple: (churn_rate, revenue_at_risk, potential_savings)
    """
    if prob_col not in df.columns or charge_col not in df.columns:
        raise ValueError("Missing required columns for KPI computation.")

    churn_rate = (df[prob_col] > 0.5).mean() * 100
    avg_charge = df[charge_col].mean()
    revenue_at_risk = (df[prob_col] > 0.5).sum() * avg_charge
    potential_savings = 0.2 * revenue_at_risk  # Assuming 20% retention impact

    return churn_rate, revenue_at_risk, potential_savings
