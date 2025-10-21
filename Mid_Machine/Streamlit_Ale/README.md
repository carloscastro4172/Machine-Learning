#  Telco Customer Churn – Streamlit Application

This repository contains an interactive **Streamlit web application** developed for analyzing and predicting customer churn using machine learning models.  
The project integrates **EDA**, **model inference**, **evaluation dashboards**, and **business impact simulations** — all inside one modular, easy-to-navigate interface.

---

##  Quick Start

###  1. Clone the repository
```bash
git clone https://github.com/yourusername/telco-churn-streamlit.git
cd telco-churn-streamlit
````

###  2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate     # macOS/Linux
venv\Scripts\activate        # Windows
```

###  3. Install dependencies

Make sure you have Streamlit and other required libraries:

```bash
pip install -r requirements.txt
```

---

##  4. Run the application

The **main entry point** is the file:

```
app.py
```

To start the web app, open a terminal inside the project folder and run:

```bash
streamlit run app.py
```

Once it’s running, Streamlit will automatically open your default browser.
If not, visit:

```
http://localhost:8501
```

---

##  Project Architecture

```
Streamlit/
│
├── app.py                     ← main entry point (launches the app)
│
├── utils/
│   ├── __init__.py
│   ├── loader.py              ← loads models, transformers, etc.
│   ├── inference_helpers.py   ← shared prediction helpers
│   ├── dashboard_helpers.py   ← metrics, charts, etc.
│   ├── business_helpers.py    ← business logic & KPIs
│
├── pages/
│   ├── 1_EDA.py               ← Tab 1: Exploratory Data Analysis
│   ├── 2_Inference.py         ← Tab 2: Prediction & Inference
│   ├── 3_Dashboard.py         ← Tab 3: Model Dashboard & Metrics
│   ├── 4_BusinessImpact.py    ← Tab 4: Business Impact Simulation
│
├── EDA.py                     ← data cleaning and utility functions
├── Transformers.py            ← custom preprocessing classes
└── models/                    ← serialized trained models (.pkl files)
```

---

## App Features

### 1️. EDA (Exploratory Data Analysis)

* Upload and clean the **Telco Customer Churn** dataset.
* Visualize class imbalance, customer metrics, and key insights.

### 2️. Inference

* Perform churn probability predictions with pre-trained models.
* Supports manual input forms and scenario testing.

### 3️. Dashboard

* Compare multiple models (F1, AUC, Accuracy).
* View confusion matrices and top feature importances.

### 4️. Business Impact

* Simulate KPIs (Revenue at Risk, Potential Savings).
* Generate personalized retention recommendations.

---

##  Modularity

Each module is independent and reusable:

* `utils/` handles preprocessing, inference, and visualization logic.
* `pages/` defines Streamlit tabs for user interaction.
* `Transformers.py` stores custom Scikit-learn-compatible classes.

This design allows **easy scaling** and **clean code management**.


## Authors

**Developed by:**
Aldrin Chávez, Brandon Jiménez, Carlos Castro, and Alessa Melo
 *School of Mathematical and Computational Sciences – Yachay Tech University*
