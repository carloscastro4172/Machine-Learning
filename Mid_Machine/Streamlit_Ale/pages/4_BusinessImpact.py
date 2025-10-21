import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from utils.business_helpers import prepare_business_data, recommend_action

# ============================================================
#  PAGE CONFIG
# ============================================================
st.title("Business Impact")

# ============================================================
#  LOAD DATA & MODELS
# ============================================================
try:
    df = st.session_state["data"]
    preparer = st.session_state["preparer"]
    models = st.session_state["models"]
    best_model = st.session_state.get("best_model")
except KeyError:
    st.error("Missing data or model. Please run the Dashboard tab first.")
    st.stop()

if df is None or preparer is None or not models:
    st.error("Data or model not found. Please return to the Dashboard tab.")
    st.stop()

# ============================================================
#  RESTORE REAL CUSTOMER ID IF AVAILABLE
# ============================================================
if "customerID" not in df.columns:
    if "customer_ids" in st.session_state:
        # Restore true customer IDs saved from EDA.py
        df = df.copy()
        df.insert(0, "customerID", st.session_state["customer_ids"])
        st.success("Real customer IDs restored from original dataset.")
    else:
        # Generate temporary IDs if none exist  
        df = df.copy()
        df.insert(0, "customerID", [f"CUST_{i+1:05d}" for i in range(len(df))])
        st.warning("No original IDs found — temporary IDs generated for display.")




# ============================================================
#  1. CUSTOMER-LEVEL IMPACT ANALYSIS
# ============================================================
st.subheader("1️. Customer-Level Prediction")

try:
    df_business = prepare_business_data(df, preparer, models)
    customer_id = st.selectbox("Select a Customer ID:", df_business["customerID"].unique())

    selected = df_business[df_business["customerID"] == customer_id].iloc[0]
    prob = selected["churn_probability"]
    action = recommend_action(prob)

    st.metric("Predicted Churn Probability", f"{prob:.2%}")
    st.info(action)

except Exception as e:
    st.error(f"Error generating customer prediction: {e}")
    st.stop()

# ============================================================
#  2️ SIMULATED BUSINESS KPIs
# ============================================================
st.subheader("2️. Simulated Business KPIs")

try:
    df_business["churn_label"] = np.where(df_business["churn_probability"] > 0.5, "At Risk", "Safe")

    churn_rate = (df_business["churn_probability"] > 0.5).mean() * 100
    avg_charge = df["MonthlyCharges"].mean()
    revenue_at_risk = (df_business["churn_probability"] > 0.5).sum() * avg_charge
    potential_savings = 0.2 * revenue_at_risk

    col1, col2, col3 = st.columns(3)
    col1.metric(" % Customers at Risk", f"{churn_rate:.1f}%")
    col2.metric(" Revenue at Risk", f"${revenue_at_risk:,.0f}")
    col3.metric(" Potential Savings (20%)", f"${potential_savings:,.0f}")

    st.caption("Simulated KPIs assume retention of 20% of at-risk customers.")
except Exception as e:
    st.warning(f"Could not calculate KPIs: {e}")

# ============================================================
#  3️ STRATEGIC VISUALIZATION (Top 5 Important Features)
# ============================================================
st.subheader(" Strategic Visualization – Churn Distribution by Key Features")

st.markdown("""
Upload the **original Telco dataset** to visualize churn distribution across the most important features.  
These visualizations reveal which customer groups are more likely to churn.
""")

uploaded_file = st.file_uploader("📂 Upload the same Telco dataset", type=["csv"])

if uploaded_file is not None:
    df_original = pd.read_csv(uploaded_file)

    # === Ensure Churn column exists ===
    if "Churn" not in df_original.columns:
        st.error("❌ The dataset must include a 'Churn' column with values 'Yes' or 'No'.")
        st.stop()

    # === Filter to top 8 features ===
    if "top_features" in st.session_state:
        top_feats = st.session_state["top_features"]
        keep_cols = ["Churn"] + [f for f in top_feats if f in df_original.columns]
        df_filtered = df_original[keep_cols].copy()
        st.success(f" Showing visualization for Top 5 important features: {', '.join(top_feats)}")
    else:
        st.warning(" No top features found in session. Run the Dashboard tab first.")
        df_filtered = df_original

    # ============================================================
    #  BARPLOT: Churn Rate per Category
    # ============================================================
    st.markdown("###  Churn Rate by Categorical Features")

    cat_features = [col for col in df_filtered.columns if df_filtered[col].dtype == 'object' and col != "Churn"]

    if len(cat_features) == 0:
        st.info("No categorical features available for visualization.")
    else:
        selected_cat = st.selectbox("Select a categorical feature to analyze:", cat_features)

        try:
            churn_rate = (
                df_filtered.groupby(selected_cat)["Churn"]
                .value_counts(normalize=True)
                .rename("Proportion")
                .reset_index()
            )

            bar_chart = (
                alt.Chart(churn_rate)
                .mark_bar()
                .encode(
                    x=alt.X(f"{selected_cat}:N", title=selected_cat),
                    y=alt.Y("Proportion:Q", title="Proportion of Customers"),
                    color=alt.Color("Churn:N", scale=alt.Scale(domain=["Yes", "No"], range=["#E35D6A", "#4C9BE8"])),
                    tooltip=[selected_cat, "Churn", alt.Tooltip("Proportion:Q", format=".2%")]
                )
                .properties(height=400, width="container", title=f"Churn Distribution by {selected_cat}")
            )

            st.altair_chart(bar_chart, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Visualization error: {e}")

    # ============================================================
    #  NUMERICAL FEATURES: AVERAGE CHURN RATES
    # ============================================================
    st.markdown("### 🔹 Average Churn Rate by Numerical Features")

    num_features = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if len(num_features) > 0:
        selected_num = st.selectbox("Select a numerical feature to analyze:", num_features)

        try:
            avg_churn = (
                df_filtered.groupby("Churn")[selected_num]
                .mean()
                .reset_index()
                .rename(columns={selected_num: "Average"})
            )

            bar_num = (
                alt.Chart(avg_churn)
                .mark_bar(size=70)
                .encode(
                    x=alt.X("Churn:N", title="Churn"),
                    y=alt.Y("Average:Q", title=f"Average {selected_num}"),
                    color=alt.Color("Churn:N", scale=alt.Scale(domain=["Yes", "No"], range=["#E35D6A", "#4C9BE8"])),
                    tooltip=["Churn", alt.Tooltip("Average:Q", format=".2f")]
                )
                .properties(height=400, width="container", title=f"Average {selected_num} by Churn Status")
            )

            st.altair_chart(bar_num, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ Could not plot numeric feature: {e}")
    else:
        st.info("No numerical features found for average churn analysis.")

    st.markdown("""
    **Interpretation Tips:**  
    - Red bars = customers who churned, Blue = those who stayed.  
    - The higher the red proportion in a category, the higher its churn risk.  
    - Combine this with your Business KPIs to design targeted retention offers.
    """)

else:
    st.info("⬆️ Upload the dataset to begin visualizing top features.")


# ============================================================
#  AUTOMATIC RECOMMENDATION — PERSONALIZED STRATEGY
# ============================================================
st.subheader("4️. Automatic Personalized Recommendations")

st.markdown("""
If **churn probability > 70%**, display **personalized retention suggestions** based on missing services.
""")

try:
    # Select a customer
    customer_id = st.selectbox(
        "Select a customer for automatic analysis:",
        df_business["customerID"].unique(),
        key="auto_recommend"
    )
    
    customer_row = df_business[df_business["customerID"] == customer_id].iloc[0]
    prob = customer_row["churn_probability"]
    
    # Merge original df to access service info
    original_row = df[df["customerID"] == customer_id].iloc[0]
    
    # Identify missing or “No” services
    service_cols = [
        "PhoneService", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    missing_services = [
        col for col in service_cols if str(original_row.get(col, "Yes")).lower() in ["no", "none", "nan"]
    ]
    
    # Generate custom recommendation
    if prob > 0.7:
        base_msg = f" {customer_id} — High churn probability ({prob*100:.1f}%)"
        st.error(base_msg)
        
        if missing_services:
            st.info(f" Missing services: **{', '.join(missing_services)}**")
            if "OnlineSecurity" in missing_services or "TechSupport" in missing_services:
                st.markdown("> Offer **security & support bundles** — these customers often have 25% lower churn.")
            elif "InternetService" in missing_services:
                st.markdown("> Upsell **InternetService** — customers with active plans show stronger retention.")
            elif "StreamingTV" in missing_services or "StreamingMovies" in missing_services:
                st.markdown("> Suggest **entertainment add-ons** — bundled plans improve satisfaction.")
            else:
                st.markdown("> Contact the customer directly with **loyalty offers or discounts**.")
        else:
            st.markdown("> This customer has all core services. Consider a **loyalty bonus** or **thank-you promotion**.")
    
    elif prob > 0.4:
        st.warning(f" {customer_id} — Medium churn risk ({prob*100:.1f}%)")
        st.markdown("> Encourage engagement through **email follow-ups** or small perks (e.g., free month of service).")
    
    else:
        st.success(f"✅ {customer_id} — Low churn risk ({prob*100:.1f}%)")
        st.markdown("> Maintain service quality and customer satisfaction.")
        
except Exception as e:
    st.warning(f"Recommendation error: {e}")


# ============================================================
#  ✨ OPTIONAL: ADDITIONAL METRICS
# ============================================================
st.markdown("---")
st.markdown("###  Extended KPIs")

try:
    total_customers = len(df_business)
    avg_tenure = df["tenure"].mean()
    avg_charges_risk = df.loc[df_business["churn_label"] == "At Risk", "MonthlyCharges"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Customers", f"{total_customers:,}")
    col2.metric(" Avg Tenure", f"{avg_tenure:.1f} months")
    col3.metric(" Avg Monthly Charge (At-Risk)", f"${avg_charges_risk:,.2f}")
except Exception as e:
    st.warning(f"Additional KPI error: {e}")
