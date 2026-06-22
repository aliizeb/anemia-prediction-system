import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

st.set_page_config(page_title="Anemia Prediction", page_icon="🩸", layout="wide", initial_sidebar_state="expanded")


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

header[data-testid="stHeader"] { display: none !important; }
section.main > div { padding-top: 0.5rem !important; }
.block-container { padding-top: 0.5rem !important; padding-bottom: 1rem !important; }

section[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    width: 21rem !important;
    min-width: 21rem !important;
}

section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
}
         
.top-bar {
    background: linear-gradient(90deg, #0f766e, #0d9488);
    border-radius: 12px;
    padding: 0.7rem 1.4rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.top-bar h1 { font-size: 1.90rem !important; font-weight: 700 !important; color: #fff !important; margin: 0 !important; }
.top-bar p  { font-size: 1.00rem; color: rgba(255,255,255,0.75); margin: 0; }

.result-banner {
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-anemia { background: rgba(239,68,68,0.1);  border: 1.5px solid #ef4444; }
.result-normal { background: rgba(20,184,166,0.1); border: 1.5px solid #14b8a6; }
.r-title { font-size: 1.35rem; font-weight: 700; }
.result-anemia .r-title { color: #ef4444; }
.result-normal .r-title { color: #14b8a6; }
.r-sub  { font-size: 0.75rem; color: #94a3b8; margin-top: 3px; }
.r-conf { font-size: 1.7rem; font-weight: 700; }
.result-anemia .r-conf { color: #ef4444; }
.result-normal .r-conf { color: #14b8a6; }

.model-row { display: flex; gap: 8px; margin-bottom: 0.8rem; }
.model-pill {
    flex: 1; border-radius: 10px; padding: 0.65rem 1rem;
    border: 1.5px solid; text-align: center;
}
.pill-anemia { border-color: #ef4444; background: rgba(239,68,68,0.07); }
.pill-normal { border-color: #14b8a6; background: rgba(20,184,166,0.07); }
.m-name { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; }
.m-pred { font-size: 0.95rem; font-weight: 600; margin: 3px 0; }
.pill-anemia .m-pred { color: #ef4444; }
.pill-normal .m-pred { color: #14b8a6; }
.m-conf { font-size: 0.72rem; color: #64748b; }

.sec-title {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; color: #0d9488;
    border-bottom: 1.5px solid #0d9488;
    padding-bottom: 3px; margin: 0.85rem 0 0.55rem;
}

.param-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.param-chip {
    border-radius: 8px; padding: 0.5rem 0.75rem;
    border: 1px solid; display: flex;
    justify-content: space-between; align-items: center;
}
.chip-ok   { border-color: rgba(20,184,166,0.3);  background: rgba(20,184,166,0.06); }
.chip-low  { border-color: rgba(251,146,60,0.3);  background: rgba(251,146,60,0.06); }
.chip-high { border-color: rgba(239,68,68,0.3);   background: rgba(239,68,68,0.06);  }
.chip-name { font-size: 0.72rem; font-weight: 600; color: #cbd5e1; }
.chip-val  { font-size: 0.8rem;  font-weight: 700; color: #f1f5f9; margin-top: 1px; }
.chip-badge {
    font-size: 0.62rem; font-weight: 700; padding: 2px 7px;
    border-radius: 4px; text-transform: uppercase; letter-spacing: 0.05em;
}
.badge-ok   { background: rgba(20,184,166,0.18);  color: #14b8a6; }
.badge-low  { background: rgba(251,146,60,0.18);  color: #fb923c; }
.badge-high { background: rgba(239,68,68,0.18);   color: #ef4444; }

div[data-testid="stSidebar"] > div:first-child { padding-top: 0.5rem; }
div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size: 0.78rem !important; }
            
.main .block-container {
    max-width: 100% !important;
    width: 100% !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] ~ .main .block-container {
    max-width: 100% !important;
}


</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_models():
    rf     = joblib.load('rf_model.pkl')
    xgb    = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return rf, xgb, scaler


all_features = [
    'Hemoglobin(g/dL)', 'Hematocrit(fL)', 'RBC(mil/uL)',
    'TIBC(ug/dL)', 'RDW(fL)', 'Ferritin(ng/mL)',
    'Serum_Iron(ug/dL)', 'MCH(pg)', 'Platelets(10*3/uL)',
    'MCV(fL)', 'WBC(10*3/uL)', 'Vitamin_B12(pg/mL)',
    'Folate(ng/mL)', 'MCHC(g/dL)'
]

healthy_ranges = {
    'Hemoglobin(g/dL)':(12.0,18.0), 'Hematocrit(fL)':(36.0,50.0),
    'RBC(mil/uL)':(4.0,6.0),        'TIBC(ug/dL)':(250.0,400.0),
    'RDW(fL)':(11.5,14.5),          'Ferritin(ng/mL)':(30.0,300.0),
    'Serum_Iron(ug/dL)':(60.0,170.0),'MCH(pg)':(27.0,33.0),
    'Platelets(10*3/uL)':(150.0,400.0),'MCV(fL)':(80.0,100.0),
    'WBC(10*3/uL)':(4.5,11.0),      'Vitamin_B12(pg/mL)':(200.0,900.0),
    'Folate(ng/mL)':(2.7,17.0),     'MCHC(g/dL)':(32.0,36.0)
}

data_ranges = {
    'Hemoglobin(g/dL)':(6.0,18.0,0.1),    'Hematocrit(fL)':(3.6,39.3,0.1),
    'RBC(mil/uL)':(2.0,7.2,0.01),          'TIBC(ug/dL)':(200.0,450.0,1.0),
    'RDW(fL)':(6.8,18.2,0.1),              'Ferritin(ng/mL)':(5.0,400.0,1.0),
    'Serum_Iron(ug/dL)':(20.0,200.0,1.0),  'MCH(pg)':(20.0,38.0,0.1),
    'Platelets(10*3/uL)':(120.0,512.0,1.0),'MCV(fL)':(65.0,110.0,0.1),
    'WBC(10*3/uL)':(3.0,16.1,0.1),         'Vitamin_B12(pg/mL)':(120.0,900.0,1.0),
    'Folate(ng/mL)':(2.0,19.4,0.1),        'MCHC(g/dL)':(28.0,38.0,0.1)
}

defaults = {
    'Hemoglobin(g/dL)':10.5,  'Hematocrit(fL)':12.3,  'RBC(mil/uL)':3.4,
    'TIBC(ug/dL)':350.0,      'RDW(fL)':15.2,          'Ferritin(ng/mL)':25.0,
    'Serum_Iron(ug/dL)':45.0, 'MCH(pg)':24.0,          'Platelets(10*3/uL)':180.0,
    'MCV(fL)':72.0,           'WBC(10*3/uL)':6.5,      'Vitamin_B12(pg/mL)':250.0,
    'Folate(ng/mL)':3.5,      'MCHC(g/dL)':30.0
}

short_names = {
    'Hemoglobin(g/dL)':'Hemoglobin',   'Hematocrit(fL)':'Hematocrit',
    'RBC(mil/uL)':'RBC',               'TIBC(ug/dL)':'TIBC',
    'RDW(fL)':'RDW',                   'Ferritin(ng/mL)':'Ferritin',
    'Serum_Iron(ug/dL)':'Serum Iron',  'MCH(pg)':'MCH',
    'Platelets(10*3/uL)':'Platelets',  'MCV(fL)':'MCV',
    'WBC(10*3/uL)':'WBC',             'Vitamin_B12(pg/mL)':'Vitamin B12',
    'Folate(ng/mL)':'Folate',          'MCHC(g/dL)':'MCHC'
}

rf, xgb, scaler = train_models()

st.markdown("""
<div class="top-bar">
    <div>
        <h1>🩸 Anemia Prediction System</h1>
        <p>\tAdjust blood parameters in the sidebar — results update instantly</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### 🔬 Blood Parameters")
    user_input = {}
    for feature in all_features:
        mn, mx, step = data_ranges[feature]
        lo, hi = healthy_ranges[feature]
        user_input[feature] = st.slider(
            f"{short_names[feature]}  ·  {lo}–{hi}",
            min_value=float(mn), max_value=float(mx),
            value=float(defaults[feature]), step=float(step), key=feature
        )

input_df     = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

rf_pred   = rf.predict(input_scaled)[0]
rf_proba  = rf.predict_proba(input_scaled)[0]
xgb_pred  = xgb.predict(input_scaled)[0]
xgb_proba = xgb.predict_proba(input_scaled)[0]

ensemble_pred = 1 if (rf_pred + xgb_pred) >= 1 else 0
ensemble_conf = (max(rf_proba) + max(xgb_proba)) / 2 * 100
rf_conf       = max(rf_proba) * 100
xgb_conf      = max(xgb_proba) * 100

col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    r_class = "result-anemia" if ensemble_pred == 1 else "result-normal"
    r_title = "Anemia Detected"    if ensemble_pred == 1 else "No Anemia Detected"
    r_sub   = "Please consult a healthcare professional" if ensemble_pred == 1 else "Blood parameters appear normal"

    st.markdown(f"""
    <div class="result-banner {r_class}">
        <div>
            <div class="r-title">{r_title}</div>
            <div class="r-sub">{r_sub}</div>
        </div>
        <div class="r-conf">{ensemble_conf:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    rf_cls  = "pill-anemia" if rf_pred  == 1 else "pill-normal"
    xgb_cls = "pill-anemia" if xgb_pred == 1 else "pill-normal"
    rf_lbl  = "Anemia" if rf_pred  == 1 else "No Anemia"
    xgb_lbl = "Anemia" if xgb_pred == 1 else "No Anemia"

    st.markdown(f"""
    <div class="model-row">
        <div class="model-pill {rf_cls}">
            <div class="m-name">Random Forest</div>
            <div class="m-pred">{rf_lbl}</div>
            <div class="m-conf">{rf_conf:.1f}% confidence</div>
        </div>
        <div class="model-pill {xgb_cls}">
            <div class="m-name">XGBoost</div>
            <div class="m-pred">{xgb_lbl}</div>
            <div class="m-conf">{xgb_conf:.1f}% confidence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">Parameter Status</div>', unsafe_allow_html=True)

    chips = '<div class="param-grid">'
    for feature in all_features:
        val    = user_input[feature]
        lo, hi = healthy_ranges[feature]
        if val < lo:
            cc, bc, bt = "chip-low",  "badge-low",  "Low"
        elif val > hi:
            cc, bc, bt = "chip-high", "badge-high", "High"
        else:
            cc, bc, bt = "chip-ok",   "badge-ok",   "OK"
        chips += f"""
        <div class="param-chip {cc}">
            <div>
                <div class="chip-name">{short_names[feature]}</div>
                <div class="chip-val">{val}</div>
            </div>
            <span class="chip-badge {bc}">{bt}</span>
        </div>"""
    chips += '</div>'
    st.markdown(chips, unsafe_allow_html=True)

with col_right:
    BG   = "#0e1117"
    FG   = "#e2e8f0"
    GRID = "#1e293b"
    TEAL = "#14b8a6"
    RED  = "#ef4444"
    SLATE= "#475569"

    st.markdown('<div class="sec-title">Confidence Chart</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(4, 2.2))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    m_labels = ['Random\nForest', 'XGBoost', 'Ensemble']
    m_preds  = [rf_pred, xgb_pred, ensemble_pred]
    m_confs  = [rf_conf, xgb_conf, ensemble_conf]
    m_colors = [RED if p == 1 else TEAL for p in m_preds]

    bars = ax.barh(m_labels, m_confs, color=m_colors, height=0.45, edgecolor='none')
    ax.set_xlim(0, 115)
    ax.set_xlabel('Confidence (%)', fontsize=8, color=FG)
    ax.tick_params(colors=FG, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for bar, conf in zip(bars, m_confs):
        ax.text(conf + 1.5, bar.get_y() + bar.get_height()/2,
                f'{conf:.1f}%', va='center', fontsize=8, color=FG, fontweight='600')

    leg = ax.legend(
        handles=[mpatches.Patch(color=RED, label='Anemia'), mpatches.Patch(color=TEAL, label='No Anemia')],
        fontsize=7, loc='lower right', facecolor=BG, edgecolor=GRID, labelcolor=FG
    )
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    st.markdown('<div class="sec-title">Feature Importance</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(4, 4))
    fig2.patch.set_facecolor(BG)
    ax2.set_facecolor(BG)

    imp_df = pd.DataFrame({
        'Feature': [short_names[f] for f in all_features],
        'Importance': rf.feature_importances_
    }).sort_values('Importance')

    med = imp_df['Importance'].median()
    bar_colors = [TEAL if v >= med else SLATE for v in imp_df['Importance']]

    ax2.barh(imp_df['Feature'], imp_df['Importance'], color=bar_colors, edgecolor='none', height=0.6)
    ax2.set_xlabel('Importance', fontsize=8, color=FG)
    ax2.tick_params(colors=FG, labelsize=7.5)
    for spine in ax2.spines.values():
        spine.set_edgecolor(GRID)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.tight_layout(pad=0.4)
    st.pyplot(fig2, use_container_width=True)
    plt.close()