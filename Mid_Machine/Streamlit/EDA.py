from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import RobustScaler 
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import RobustScaler 
from sklearn.impute import SimpleImputer 
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay 
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score 
from sklearn.metrics import f1_score 
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import PrecisionRecallDisplay

import matplotlib.pyplot as plt 
import pandas as pd

#=========== Cargamos la base de datos ===============
#df = pd.read_csv("/home/carlos/Documents/8vo/MID_MACHINE/Mid_Machine/WA_Fn-UseC_-Telco-Customer-Churn.csv")
#Alessa
df = pd.read_csv(r"C:\Users\Ale\Documents\Yachay\Sexto\Machine Learning\Machine-Learning\Mid_Machine\WA_Fn-UseC_-Telco-Customer-Churn.csv")
#Aldrin
#df = pd.read_csv("")
#Brandon
#df = pd.read_csv("")

#Eliminamos la columna Customer_ID 
df = df.drop(['customerID'], axis=1)
#Esta columna esta en strings
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

#Vamos a imputar
imputer = SimpleImputer(strategy='median')
df['TotalCharges'] = imputer.fit_transform(df[['TotalCharges']])

#Reemplazamos los Yes/No por 1, 0
colum = ['Partner', 'Dependents', 'PhoneService', '' ]
print(df.info())
print(df.head())
#Reemplazamos los Yes/No por 1, 0
colum = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling' ]
df = df[colum].replace({'Yes': 1, 'No':0 })