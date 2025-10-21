import streamlit as st
from utils.loader import load_preparer, load_all_models

# =====================================================
# 🎨 PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    layout="wide",
    page_icon="📶",
    initial_sidebar_state="expanded",
)

# =====================================================
# 🎨 CUSTOM PLDT-STYLE THEME
# =====================================================
st.markdown("""
<style>
:root {
    --burgundy: #8B0D32;
    --burgundy-dark: #6C0A27;
    --black-banner: #000000;
    --white: #FFFFFF;
    --text-gray: #4A4A4A;
    --light-bg: #FAFAFA;
}

/* ===== General Page ===== */
[data-testid="stAppViewContainer"] {
    background-color: var(--light-bg);
}

.stTitle {
    color: var(--white);
    font-weight: 800;
    text-align: center;
    letter-spacing: 1px;
}

/* ===== Banner ===== */
.banner {
    background-color: var(--black-banner);
    color: white;
    text-align: center;
    padding: 60px 20px;
    border-radius: 0;
    margin-top: -70px;
    margin-bottom: 30px;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.4);
}

/* ===== Sidebar ===== */
[data-testid="stSidebar"] {
    background-color: var(--burgundy);
    color: var(--white);
    padding-top: 40px !important;
}

[data-testid="stSidebarNav"] a {
    font-size: 15px;
    font-weight: 600;
    text-transform: uppercase;
    color: var(--white) !important;
    letter-spacing: 0.8px;
    padding: 12px 18px;
    display: block;
    margin-bottom: 6px;
    border-radius: 8px;
    transition: all 0.3s ease;
}

[data-testid="stSidebarNav"] a:hover {
    background-color: var(--burgundy-dark);
    color: #FFFFFF !important;
    transform: translateX(3px);
}

[data-testid="stSidebarNav"] a[selected="true"] {
    background-color: #B91942 !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
    font-weight: 500;
}

/* ===== Cards ===== */
.section {
    background-color: white;
    border-left: 6px solid var(--burgundy);
    padding: 25px 35px;
    margin-top: 1.5rem;
    border-radius: 12px;
    box-shadow: 0px 3px 8px rgba(0,0,0,0.08);
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# 🧠 BLACK BANNER
# =====================================================
st.markdown("""
<div class="banner">
    <h1>TELCO CUSTOMER CHURN</h1>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 📄 INTRODUCTION
# =====================================================
st.markdown("""
<div style="text-align:center; max-width:800px; margin:auto; color:#4A4A4A; font-size:1.1rem; line-height:1.6;">
Welcome to the <b>Telco Customer Churn Analysis Platform</b>, a professional environment for understanding 
customer retention dynamics. This dashboard provides insights into how contract length, internet service type, 
and payment preferences affect customer churn rates.
</div>
""", unsafe_allow_html=True)

# =====================================================
# 💡 ABOUT SECTION
# =====================================================
st.markdown("""
<div class="section">
<h3> About the Dataset</h3>
<p style='color:#555; font-size:1.05rem;'>
The <strong>Telco Customer Churn dataset</strong> represents customer behavior within a telecommunications company, including
service subscriptions, tenure, contract types, billing preferences, and churn decisions. It’s commonly used to 
build predictive models for identifying customers likely to leave.
</p>

<ul style='color:#666; line-height:1.7; font-size:1rem;'>
<li><strong>Rows:</strong> Approximately 7,000 customer entries</li>
<li><strong>Features:</strong> Demographics, subscription details, billing methods, and internet services</li>
<li><strong>Target:</strong> <em>Churn</em> — whether a customer left the company</li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 🧩 LOAD OBJECTS
# =====================================================
if "preparer" not in st.session_state:
    PREPARER_PATH = "./preparer.pkl"
    st.session_state["preparer"] = load_preparer()

if "models" not in st.session_state:
    st.session_state["models"] = load_all_models()

# =====================================================
#  STATUS SECTION
# =====================================================
st.markdown("""
<div class="section" style='border-left-color:#6C0A27; background-color:#FFF6F8;'>
<h3> Environment Status</h3>
<p style='font-size:1rem; color:#5A5656;'>
The environment is ready. Data preparer and machine learning models are loaded.  
Use the sidebar to explore <b>EDA</b>, <b>Inference</b>, <b>Dashboard</b>, and <b>Business Impact</b> sections.
</p>
</div>
""", unsafe_allow_html=True)
