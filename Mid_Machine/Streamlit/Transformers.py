#Transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer

class CategoricalServiceCleaner(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._cols = None

    def fit(self, X, y=None):
        self._cols = list(X.columns)
        return self

    def transform(self, X, y=None):
        Xc = X.copy()
        for c in self._cols:
            Xc[c] = (
                Xc[c]
                .astype(str)
                .str.strip()
                .replace({"No internet service": "No", "No phone service": "No"})
            )
        return Xc

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="if_binary"
        )
        self._cat_cols = None
        self._columns_out = None

    def fit(self, X, y=None):
        Xc = X.astype(str)
        self._cat_cols = list(Xc.columns)
        self._oh.fit(Xc)
        self._columns_out = list(self._oh.get_feature_names_out(self._cat_cols))
        return self

    def transform(self, X, y=None):
        Xc = X.astype(str)
        return self._oh.transform(Xc)

    def get_feature_names_out(self, input_features=None):
        cols = input_features if input_features is not None else self._cat_cols
        return self._oh.get_feature_names_out(cols)

class NumericFeaturizer(BaseEstimator, TransformerMixin):
    def __init__(self, add_interaction=True, drop_totalcharges=False):
        self.add_interaction = add_interaction
        self.drop_totalcharges = drop_totalcharges
        self._cols = None

    def fit(self, X, y=None):
        self._cols = list(X.columns)
        return self

    def transform(self, X, y=None):
        Xn = X.copy()
        if "TotalCharges" in Xn.columns:
            Xn["TotalCharges"] = pd.to_numeric(Xn["TotalCharges"], errors="coerce")
        if self.add_interaction and {"tenure", "MonthlyCharges"}.issubset(Xn.columns):
            Xn["tenure_x_monthly"] = Xn["tenure"] * Xn["MonthlyCharges"]
        if self.drop_totalcharges and "TotalCharges" in Xn.columns:
            Xn = Xn.drop(columns=["TotalCharges"])
        return Xn

class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self, add_interaction=True, drop_totalcharges=False, drop_cols=None):
        self.add_interaction = add_interaction
        self.drop_totalcharges = drop_totalcharges
        self.drop_cols = drop_cols if drop_cols is not None else ["customerID"]
        self._ct = None
        self._num_cols = None
        self._cat_cols = None
        self._columns_out = None

    def fit(self, X, y=None):
        Xw = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")

        num_cols = list(Xw.select_dtypes(include=["int64", "float64"]).columns)
        if "TotalCharges" in Xw.columns and "TotalCharges" not in num_cols:
            num_cols.append("TotalCharges")
        self._num_cols = num_cols
        self._cat_cols = [c for c in Xw.columns if c not in self._num_cols]

        num_pipeline = Pipeline([
            ("feats", NumericFeaturizer(add_interaction=self.add_interaction,
                                        drop_totalcharges=self.drop_totalcharges)),
            ("imputer", SimpleImputer(strategy="median")),
            ("rbst_scaler", RobustScaler())
        ])

        cat_pipeline = Pipeline([
            ("clean", CategoricalServiceCleaner()),
            ("oh_df", CustomOneHotEncoder())
        ])

        self._ct = ColumnTransformer([
            ("num", num_pipeline, self._num_cols),
            ("cat", cat_pipeline, self._cat_cols)
        ], remainder="drop")

        self._ct.fit(Xw)

        num_out = list(self._num_cols)
        if self.add_interaction and {"tenure", "MonthlyCharges"}.issubset(self._num_cols):
            num_out.append("tenure_x_monthly")
        if self.drop_totalcharges and "TotalCharges" in num_out:
            num_out.remove("TotalCharges")

        oh = self._ct.named_transformers_["cat"].named_steps["oh_df"]
        cat_out = list(oh.get_feature_names_out(self._cat_cols))
        self._columns_out = num_out + cat_out
        return self

    def transform(self, X, y=None):
        Xw = X.drop(columns=[c for c in self.drop_cols if c in X.columns], errors="ignore")
        Xt = self._ct.transform(Xw)
        return pd.DataFrame(Xt, columns=self._columns_out, index=X.index)
