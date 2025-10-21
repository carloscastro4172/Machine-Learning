import streamlit as st
import altair as alt
import pandas as pd
from EDA import clean_dt, kpis_por_segmento


# ====================================================
# EDA PAGE
# ====================================================
st.title("Exploratory Data Analysis")

# ---------------- Upload dataset ----------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
if uploaded_file is None:
    st.info("Upload a dataset to start.")
    st.stop()

raw_df = pd.read_csv(uploaded_file)

# 🔒 Save original customer IDs before cleaning
if "customerID" in raw_df.columns:
    st.session_state["customer_ids"] = raw_df["customerID"].copy()

# 🧼 Then clean your data normally
df = clean_dt(raw_df)

# 💾 Store cleaned dataframe for other pages
st.session_state["df"] = df
# Share with other pages

# ---------------- Preview ----------------
st.subheader("Preview")
n_rows = st.slider("Rows to display", 5, len(df), 5, step=5)
st.dataframe(df.head(n_rows), use_container_width=True)

# ---------------- Basic KPI ----------------
if "Churn" in df.columns:
    churn_rate = df["Churn"].mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.2f}%")

# ====================================================
# FILTER SIDEBAR
# ====================================================
st.sidebar.header("Filter Options")

num_vars = [c for c in ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"] if c in df.columns]
cat_vars = [c for c in df.columns if c not in num_vars]

def opts(col):
    return sorted([v for v in df[col].dropna().unique()]) if col in df.columns else []

# --- Categorical Filters ---
with st.sidebar.expander("Categorical Variables", expanded=True):
    sel_contract = st.multiselect("Contract", opts("Contract"), default=opts("Contract"))
    sel_internet = st.multiselect("InternetService", opts("InternetService"), default=opts("InternetService"))
    sel_payment  = st.multiselect("PaymentMethod", opts("PaymentMethod"), default=opts("PaymentMethod"))
    sel_gender   = st.multiselect("Gender", opts("gender"), default=opts("gender"))
    sel_senior   = st.multiselect(
        "SeniorCitizen",
        sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else [],
        default=sorted(df["SeniorCitizen"].dropna().unique().tolist()) if "SeniorCitizen" in df.columns else []
    )
    solo_churn = st.checkbox("Show only Churn = 1 customers", value=False)

# --- Numeric Filters ---
with st.sidebar.expander("Numeric Variables", expanded=True):
    def range_slider_for(col, step=1.0):
        vmin = float(df[col].min()) if col in df.columns else 0.0
        vmax = float(df[col].max()) if col in df.columns else 0.0
        # Ensure all have same type
        step = float(step)
        return st.slider(
            f"{col}",
            min_value=vmin,
            max_value=vmax,
            value=(vmin, vmax),
            step=step
        )

    ten_r = range_slider_for("tenure", step=1.0)
    mon_r = range_slider_for("MonthlyCharges", step=0.5)
    tot_r = range_slider_for("TotalCharges", step=1.0)


# ====================================================
# APPLY FILTERS
# ====================================================
df_f = df.copy()
if sel_contract: df_f = df_f[df_f["Contract"].isin(sel_contract)]
if sel_internet: df_f = df_f[df_f["InternetService"].isin(sel_internet)]
if sel_payment:  df_f = df_f[df_f["PaymentMethod"].isin(sel_payment)]
if sel_gender:   df_f = df_f[df_f["gender"].isin(sel_gender)]
if sel_senior:   df_f = df_f[df_f["SeniorCitizen"].isin(sel_senior)]
df_f = df_f[df_f["tenure"].between(ten_r[0], ten_r[1])]
df_f = df_f[df_f["MonthlyCharges"].between(mon_r[0], mon_r[1])]
df_f = df_f[df_f["TotalCharges"].between(tot_r[0], tot_r[1])]
if solo_churn:   df_f = df_f[df_f["Churn"] == 1]

if df_f.empty:
    st.warning("No data for the selected filters. Adjust filters and try again.")
    st.stop()

# ====================================================
# MAIN KPIs
# ====================================================
st.subheader("Key Metrics")
col1, col2, col3, col4 = st.columns(4)
churn_rate = (df_f["Churn"].mean() * 100) if "Churn" in df_f.columns else 0.0
avg_charge = df_f["MonthlyCharges"].mean()
avg_tenure = df_f["tenure"].mean()
total_clients = len(df_f)

col1.metric("Churn Rate", f"{churn_rate:.2f}%")
col2.metric("Average Charge", f"${avg_charge:.2f}")
col3.metric("Avg Tenure", f"{avg_tenure:.1f} months")
col4.metric("Total Clients", f"{total_clients:,}")

st.markdown("---")

# ====================================================
# EDA SUBTABS
# ====================================================
s1, s2, s3, s4 = st.tabs(["Distributions", "Comparative", "Segmentation", "Insights"])

# --- Distributions ---
with s1:
    st.subheader("Numeric Distributions")
    for i in range(0, len(num_vars), 2):
        cols = st.columns(2)
        for j, col in enumerate(num_vars[i:i+2]):
            with cols[j]:
                chart = alt.Chart(df_f).mark_bar(opacity=0.8).encode(
                    alt.X(col, bin=alt.Bin(maxbins=30)),
                    alt.Y('count()')
                )
                st.altair_chart(chart, use_container_width=True)

# --- Comparative ---
with s2:
    st.subheader("Relationship with Target Variable (Churn)")
    comp_vars = ["Contract", "InternetService", "PaymentMethod"]
    for var in comp_vars:
        if var in df_f.columns:
            chart = alt.Chart(df_f).mark_bar().encode(
                x=var, y='count()', color='Churn:N'
            )
            st.altair_chart(chart, use_container_width=True)

# --- Segmentation ---
with s3:
    st.subheader("KPIs by Segment")
    seg_opts = [c for c in cat_vars if c in df_f.columns]
    if seg_opts:
        seg_col = st.selectbox("Segment by", seg_opts)
        seg_table = kpis_por_segmento(df_f, seg_col)
        st.dataframe(seg_table, use_container_width=True)
    else:
        st.info("No categorical variables available for segmentation.")

# --- Insights ---
with s4:
    st.subheader("Automatic Insights")
    bullets = []
    if "Contract" in df_f.columns:
        gr = df_f.groupby("Contract")["Churn"].mean().sort_values(ascending=False)*100
        bullets.append(f"- Highest churn in contract: {gr.index[0]} ({gr.iloc[0]:.2f}%).")
    if "InternetService" in df_f.columns:
        gr = df_f.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)*100
        bullets.append(f"- Highest churn in internet service: {gr.index[0]} ({gr.iloc[0]:.2f}%).")
    if bullets:
        st.write("\n".join(bullets))
    else:
        st.info("Adjust filters to generate insights.")
