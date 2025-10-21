

import os, io, pickle, joblib
import streamlit as st
import Transformers as T

# =====================================================
# 🔧 PATH CONFIGURATION
# =====================================================
# Root folder where your Streamlit app runs
BASE_DIR_1 = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR_1 = os.path.abspath(os.path.join(BASE_DIR_1, "..")) 
PREPARER_PATH = os.path.join(ROOT_DIR_1, "preparer.pkl")


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../.."))  
MODELS_DIR = os.path.join(ROOT_DIR, "models")


# =====================================================
# 🔄 LOADING LOGIC
# =====================================================
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
    if isinstance(src, (bytes, bytearray)):
        return _RedirectingUnpickler(io.BytesIO(src)).load()
    with open(src, "rb") as f:
        return _RedirectingUnpickler(f).load()

@st.cache_resource(show_spinner=True)
def load_preparer():
    return load_pickle_with_redirect(PREPARER_PATH)

@st.cache_resource(show_spinner=False)
def load_model(path: str):
    return joblib.load(path)

@st.cache_resource(show_spinner=True)
def load_all_models():
    MODEL_REGISTRY = {
        "Voting (full) – no SMOTE": os.path.join(MODELS_DIR, "voting_full_no_smote.pkl"),
        "Voting (full) – SMOTE":    os.path.join(MODELS_DIR, "voting_full_smote.pkl"),
        "Voting (feats) – no SMOTE":os.path.join(MODELS_DIR, "voting_feats_no_smote.pkl"),
        "Voting (feats) – SMOTE":   os.path.join(MODELS_DIR, "voting_feats_smote.pkl"),
        "Voting (PCA) – no SMOTE":  os.path.join(MODELS_DIR, "voting_pca_no_smote.pkl"),
        "Voting (PCA) – SMOTE":     os.path.join(MODELS_DIR, "voting_pca_smote.pkl"),
    }
    loaded = {}
    for name, path in MODEL_REGISTRY.items():
        try:
            loaded[name] = load_model(path)
        except Exception as e:
            st.warning(f"Could not load '{name}' from {path}: {e}")
    return loaded
