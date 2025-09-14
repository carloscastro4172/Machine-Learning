import pandas as pd
from sklearn.impute import SimpleImputer

def clean_dt(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia y tipa el dataset Telco para EDA."""
    if "customerID" in df.columns:
        df = df.drop(['customerID'], axis=1)

    # TotalCharges a numérico + imputación
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    imputer = SimpleImputer(strategy='median')
    df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

    # Binarias Yes/No -> 1/0 (incluye Churn)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].replace({'Yes': 1, 'No': 0}).astype("Int64")

    # Tratar servicios "sin internet/teléfono" como "No"
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for c in service_cols:
        if c in df.columns:
            df[c] = df[c].replace({'No internet service': 'No', 'No phone service': 'No'})

    # Asegurar categóricas como string (evita errores con Arrow/Streamlit)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype("string")

    return df

def metric(df: pd.DataFrame, column: str) -> pd.Series:
    """Métricas descriptivas. Si es binaria (0/1), devuelve conteos/porcentajes."""
    series = df[column]
    if set(series.dropna().unique()) <= {0, 1}:  # binaria
        count_1 = int(series.sum())
        total = int(series.count())
        count_0 = total - count_1
        return pd.Series({
            'Total': total,
            '1s': count_1,
            '0s': count_0,
            'Porcentaje 1': round((count_1 / total) * 100, 2) if total else 0.0,
            'Porcentaje 0': round((count_0 / total) * 100, 2) if total else 0.0
        })
    else:
        return series.describe()

def kpis_por_segmento(df: pd.DataFrame, segment_col: str) -> pd.DataFrame:
    """KPIs por segmento: tasa de churn, promedios y tamaño del grupo."""
    if segment_col not in df.columns:
        return pd.DataFrame()
    g = df.groupby(segment_col, dropna=False)
    out = pd.DataFrame({
        'Clientes': g.size(),
        'Tasa_Churn_%': (g['Churn'].mean() * 100).round(2),
        'Tenure_Prom': g['tenure'].mean().round(2),
        'Monthly_Prom': g['MonthlyCharges'].mean().round(2),
        'Total_Prom': g['TotalCharges'].mean().round(2),
    }).sort_values('Tasa_Churn_%', ascending=False)
    return out.reset_index()
