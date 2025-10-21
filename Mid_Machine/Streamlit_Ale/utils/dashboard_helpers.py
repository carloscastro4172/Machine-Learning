import numpy as np
import pandas as pd
from scipy import sparse
import re
import streamlit as st

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
        return estimator.predict_proba(X)[:, pos_idx]
    elif hasattr(estimator, "decision_function"):
        s = estimator.decision_function(X).ravel()
        return 1.0 / (1.0 + np.exp(-s))
    else:
        pred = estimator.predict(X)
        return np.clip(np.array(pred, dtype=float), 0.0, 1.0)

def guess_group(name):
    if "=" in name:
        return name.split("=")[0]
    if "__" in name:
        return name.split("__")[0]
    m = re.match(r"([A-Za-z][A-Za-z0-9]+)[_\|].+", name)
    if m:
        return m.group(1)
    return name

def group_score_fast(X, cols):
    if sparse.issparse(X):
        sub = X[:, cols]
        m = np.asarray(sub.mean(axis=0)).ravel()
        v = m * (1 - m)
        return float(np.nanmean(v)) if len(v) else 0.0
    else:
        sub = np.asarray(X)[:, cols]
        v = np.var(sub, axis=0)
        return float(np.nanmean(v)) if v.size else 0.0

@st.cache_data(show_spinner=True)
def fast_group_permutation_importance_cached(
    model_name, scoring, n_repeats, idx_seed, y, group_list, feat_names, X_block, is_sparse
):
    from sklearn.metrics import roc_auc_score, f1_score, log_loss
    rng_local = np.random.RandomState(idx_seed)

    def predict_proba_pos(Xloc, est):
        if hasattr(est, "predict_proba"):
            return est.predict_proba(Xloc)[:, 1]
        elif hasattr(est, "decision_function"):
            s = est.decision_function(Xloc).ravel()
            return 1.0 / (1.0 + np.exp(-s))
        else:
            return (est.predict(Xloc).astype(float)).clip(0, 1)

    est_best = st.session_state["models"][model_name]["model"]
    p_base = predict_proba_pos(X_block, est_best)

    def scorer(y_true_loc, p_loc):
        if scoring == "roc_auc":
            return roc_auc_score(y_true_loc, p_loc)
        elif scoring == "f1":
            y_pred_loc = (p_loc >= 0.5).astype(int)
            return f1_score(y_true_loc, y_pred_loc, zero_division=0)
        elif scoring == "neg_log_loss":
            return -log_loss(y_true_loc, p_loc, labels=[0,1])
        else:
            return roc_auc_score(y_true_loc, p_loc)

    base_score = scorer(y, p_base)
    importances = []
    for g_name, cols, _ in group_list:
        scores = []
        for _ in range(n_repeats):
            if is_sparse:
                Xperm = X_block.copy().tolil()
                for c in cols:
                    col = np.asarray(Xperm[:, c].todense()).ravel()
                    rng_local.shuffle(col)
                    Xperm[:, c] = col.reshape(-1, 1)
                Xperm = Xperm.tocsr()
            else:
                Xperm = np.asarray(X_block).copy()
                for c in cols:
                    rng_local.shuffle(Xperm[:, c])
            p = predict_proba_pos(Xperm, est_best)
            scores.append(scorer(y, p))
        delta = np.nanmean(base_score - np.array(scores))
        importances.append((g_name, delta))

    imp_df_grp = (
        pd.DataFrame(importances, columns=["group","importance"])
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return imp_df_grp
