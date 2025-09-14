import pandas as pd
from sklearn.impute import SimpleImputer

# ============
# LIMPIEZA
# ============
def clean_dt(df):
    if "customerID" in df.columns:
        df = df.drop(['customerID'], axis=1)

    # Convertir TotalCharges a numérico
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

    # Columnas binarias: Yes/No -> 1/0
    colum = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for c in colum:
        if c in df.columns:
            df[c] = df[c].replace({'Yes': 1, 'No': 0}).astype("Int64")

    # Asegurar que otras categóricas se queden como string
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")

    return df

# ============
# MÉTRICAS
# ============
def metric(df, column):
    series = df[column]
    if set(series.dropna().unique()) <= {0, 1}:  # binaria
        count_1 = series.sum()
        count_0 = len(series) - count_1
        return pd.Series({
            'Total': len(series),
            '1s': int(count_1),
            '0s': int(count_0),
            'Porcentaje 1': round(count_1 / len(series) * 100, 2),
            'Porcentaje 0': round(count_0 / len(series) * 100, 2)
        })
    else:  # numérica o categórica
        return series.describe()
