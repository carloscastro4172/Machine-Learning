Streamlit/
│
├── app.py                     ← main entry point (streamlit run app.py)
│
├── utils/
│   ├── __init__.py
│   ├── loader.py              ← loads models, transformers, etc.
│   ├── inference_helpers.py   ← shared prediction helpers
│   ├── dashboard_helpers.py   ← metrics, charts, etc.
│   ├── business_helpers.py
|
├── pages/
│   ├── 1_EDA.py               ← Tab 1
│   ├── 2_Inference.py         ← Tab 2
│   ├── 3_Dashboard.py         ← Tab 3
│   ├── 4_BusinessImpact.py    ← Tab 4
│
├── EDA.py                     ← your existing cleaning functions
├── Transformers.py            ← your custom classes
└── models/                    ← pkl files

