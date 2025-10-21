import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix,
    precision_score, recall_score
)
from utils.dashboard_helpers import (
    unwrap_bundle,
    get_pos_index,
    build_features_for_bundle,
    to_binary_series,
    predict_proba_yes,
    fast_group_permutation_importance_cached,
    guess_group,
    group_score_fast
)
from utils.loader import load_preparer, load_all_models

# =====================================================
#  PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Model Dashboard – Telco Customer Churn")

# =====================================================
#  Load Data
# =====================================================
if "df" not in st.session_state:
    st.error("Dataset not loaded. Please upload the CSV on the EDA page.")
    st.stop()
df = st.session_state["df"]

if "Churn" not in df.columns:
    st.error("The dataset does not contain the 'Churn' column.")
    st.stop()

# =====================================================
#  Load Preparer and Models
# =====================================================
try:
    preparer = st.session_state.get("preparer", load_preparer())
except Exception as e:
    st.error(f"The transformer could not be loaded. (preparer): {e}")
    st.stop()

raw_models = st.session_state.get("models", load_all_models())
if not raw_models:
    st.error("No valid model found.")
    st.stop()

# =====================================================
#  Prepare Models
# =====================================================
bundles = {}
for name, obj in raw_models.items():
    b, est = unwrap_bundle(obj)
    if est is not None and hasattr(est, "predict"):
        bundles[name] = b

if not bundles:
    st.error("None of the loaded models are valid.")
    st.stop()

# =====================================================
#  Evaluation Configuration
# =====================================================
st.markdown("### Evalutation Settings")
cols_top = st.columns([2,1,1])
with cols_top[0]:
    sel_models = st.multiselect("Models to compare", list(bundles.keys()), default=list(bundles.keys())[:3])
with cols_top[1]:
    use_bundle_thr = st.checkbox("Models to compare", value=True)
with cols_top[2]:
    custom_thr = st.slider("Custom threshold", 0.0, 1.0, 0.5, 0.01, disabled=use_bundle_thr)

if not sel_models:
    st.info("Select at least one model to compare.")
    st.stop()

# =====================================================
#  Data Preparation
# =====================================================
y_true = to_binary_series(df["Churn"])
exp_num = list(getattr(preparer, "_num_cols", []))
exp_cat = list(getattr(preparer, "_cat_cols", []))
needed_cols = exp_num + exp_cat

X_raw = df.copy()
for c in needed_cols:
    if c not in X_raw.columns:
        X_raw[c] = 0 if c in exp_num else "No"

# =====================================================
#  Evaluate Models
# =====================================================
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

        results.append({
            "Model": name,
            "Scenario": bundle.get("scenario", "?"),
            "Threshold": thr,
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "AUC": roc_auc_score(y_true, p_yes)
        })
        preds_cache[name] = (y_pred, p_yes, feat_names, est, bundle)
    except Exception as e:
        st.error(f"Error evaluating {name}: {e}")

if not results:
    st.stop()

res_df = pd.DataFrame(results).sort_values("F1", ascending=False)

# =====================================================
#  Comparison Table + F1 Chart
# =====================================================
st.markdown("### Metrics Comparison")
st.dataframe(res_df.style.format({"Accuracy":"{:.3f}","F1":"{:.3f}","AUC":"{:.3f}","Threshold":"{:.2f}"}))

chart_f1 = (
    alt.Chart(res_df)
    .mark_bar()
    .encode(
        x=alt.X('Model:N', sort='-y', title="Model"),
        y=alt.Y('F1:Q', title='F1 Score'),
        color=alt.Color('Scenario:N', title='Scenario'),
        tooltip=['Model','Scenario','Accuracy','F1','AUC','Threshold']
    )
    .properties(height=280)
)
st.altair_chart(chart_f1, use_container_width=True)

# =====================================================
#  Confusion Matrix for Best Model
# =====================================================
best_name = res_df.iloc[0]["Model"]
y_pred_best, p_best, feat_names_best, est_best, bundle_best = preds_cache[best_name]
cm = confusion_matrix(y_true, y_pred_best, labels=[0, 1])
TN, FP, FN, TP = cm.ravel()

precision = precision_score(y_true, y_pred_best, zero_division=0)
recall    = recall_score(y_true, y_pred_best, zero_division=0)
specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
fnr = FN / (FN + TP) if (FN + TP) > 0 else 0.0

st.markdown(f"### Confussion Matrix – Best F1: **{best_name}**")
cols_metrics = st.columns(5)
for i, (label, val) in enumerate([
    ("Precision", precision),
    ("Recall (TPR)", recall),
    ("Specificity (TNR)", specificity),
    ("FPR", fpr),
    ("FNR", fnr),
]):
    cols_metrics[i].metric(label, f"{val:.3f}")

# Confusion matrix heatmap
normalize = st.checkbox("Normalize by row (%)", value=True)
cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]).reset_index().rename(columns={"index":"Actual"})
cm_long = cm_df.melt(id_vars="Actual", var_name="Pred", value_name="count")
cm_long["row_pct"] = cm_long.groupby("Actual")["count"].transform(lambda x: x / x.sum())
cm_long["text"] = cm_long.apply(lambda r: f"{r['count']}\n({r['row_pct']*100:.1f}%)", axis=1)
value_field = "row_pct" if normalize else "count"

conf_matrix = (
    alt.Chart(cm_long)
    .mark_rect()
    .encode(
        x=alt.X("Pred:N", title="Prediction"),
        y=alt.Y("Actual:N", title="Actual Value"),
        color=alt.Color(f"{value_field}:Q", title="Value", scale=alt.Scale(scheme="blues")),
        tooltip=["Actual","Pred","count",alt.Tooltip("row_pct:Q",format=".1%")]
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

# =====================================================
#  Feature Importance (modo rápido)
# =====================================================
st.markdown("### Feature Importance (modo rápido)")
c1, c2, c3, c4 = st.columns(4)
with c1:
    n_sample = st.slider("Feature Importance", 200, min(3000, len(df)), 800, 100)
with c2:
    n_repeats = st.slider("Repetitions", 1, 10, 3, 1)
with c3:
    top_k_groups = st.slider("Top-K groups to evaluate", 5, 40, 15, 1)
with c4:
    scoring_choice = st.selectbox("Metric to permute", ["roc_auc","f1","neg_log_loss"], index=0)

try:
    X_feat_best, feat_names_best = build_features_for_bundle(X_raw, preparer, bundle_best)
except Exception:
    st.error("Error building features for the best model.")
    st.stop()

# Agrupar nombres
groups = {}
for j, fname in enumerate(feat_names_best):
    g = guess_group(str(fname))
    groups.setdefault(g, []).append(j)

rng = np.random.RandomState(42)
idx = rng.choice(len(y_true), size=min(n_sample, len(y_true)), replace=False)
if hasattr(X_feat_best, "iloc"):
    Xp = X_feat_best.iloc[idx]
else:
    Xp = X_feat_best[idx]

yp = np.asarray(y_true)[idx]

group_candidates = [(g, cols, group_score_fast(Xp, cols)) for g, cols in groups.items()]
group_candidates.sort(key=lambda t: t[2], reverse=True)
group_candidates = group_candidates[:top_k_groups]

is_sparse = hasattr(Xp, "tocsr")
imp_grp_df = fast_group_permutation_importance_cached(
    model_name=str(best_name),
    scoring=scoring_choice,
    n_repeats=int(n_repeats),
    idx_seed=42,
    y=yp,
    group_list=group_candidates,
    feat_names=feat_names_best,
    X_block=Xp,
    is_sparse=is_sparse
)

if imp_grp_df is None or imp_grp_df.empty:
    st.info("It was not possible to calculate quick importances.")
else:
    chart_imp_grp = (
        alt.Chart(imp_grp_df.head(top_k_groups))
        .mark_bar()
        .encode(
            x=alt.X("importance:Q", title=f"Δ{scoring_choice} (↑ = mayor impacto)"),
            y=alt.Y("group:N", sort='-x', title="Feature original (group)"),
            tooltip=["group", alt.Tooltip("importance:Q", format=".5f")]
        )
        .properties(height=380)
    )
    st.altair_chart(chart_imp_grp, use_container_width=True)


# =====================================================
#  SAVE OBJECTS FOR OTHER TABS (Business Impact)
# =====================================================
st.session_state["df"] = df                     # keep as backup
st.session_state["data"] = df                   # for Business Impact tab
st.session_state["preparer"] = preparer         # fitted transformer
st.session_state["models"] = raw_models         # all models loaded
st.session_state["best_model"] = est_best       # best trained model
st.session_state["best_model_name"] = best_name # name of best model
st.success(" Models, data, and transformer stored successfully. You can now open the Business Impact tab!")

top_features = imp_grp_df.sort_values("importance", ascending=False).head(5)["group"].tolist()
st.session_state["top_features"] = top_features
st.success(f" Top 5 important features saved: {', '.join(top_features)}")
