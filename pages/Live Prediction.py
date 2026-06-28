import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib


def inject_lp_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 60%, #0a1628 100%); }
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0a1628 0%, #112240 100%);
        border-bottom: 2px solid #19CA9C;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
        border-right: 1px solid #1a2e45;
    }
    .main .block-container { padding-top:2rem; padding-left:2.5rem; padding-right:2.5rem; max-width:1400px; }
    h1 {
        font-size:1.9rem !important; font-weight:700 !important; color:#ffffff !important;
        border-bottom:3px solid #19CA9C; padding-bottom:0.4rem; margin-bottom:0.6rem !important;
    }
    h2 { font-size:1.3rem !important; font-weight:600 !important; color:#e2e8f0 !important;
         border-left:4px solid #19CA9C; padding-left:10px; }
    h3 { font-size:1.05rem !important; font-weight:500 !important; color:#a8c0d6 !important; }
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #112240 0%, #1a2e45 100%);
        border:1px solid #1e3a5f; border-top:3px solid #19CA9C;
        border-radius:10px; padding:16px 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover { transform:translateY(-2px); box-shadow:0 6px 24px rgba(25,202,156,0.15); }
    [data-testid="stMetricLabel"] { font-size:11px !important; font-weight:500 !important; color:#8899aa !important; text-transform:uppercase; letter-spacing:0.6px; }
    [data-testid="stMetricValue"] { font-size:2rem !important; font-weight:700 !important; color:#19CA9C !important; }
    [data-testid="stMetricDelta"] { font-size:13px !important; }
    .stButton > button {
        background: linear-gradient(135deg, #19CA9C 0%, #0fa87d 100%);
        color:#0a0f1e; font-weight:700; font-size:16px; border:none;
        border-radius:8px; padding:12px 40px; letter-spacing:0.3px;
        transition:all 0.25s; box-shadow:0 4px 14px rgba(25,202,156,0.3);
        width:100%;
    }
    .stButton > button:hover {
        box-shadow:0 8px 24px rgba(25,202,156,0.5); transform:translateY(-1px);
    }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input { background:#112240 !important; border:1px solid #1e3a5f !important; border-radius:6px !important; color:#e2e8f0 !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] { background:#19CA9C !important; }
    [data-testid="stExpander"] { background:#0d1b2a; border:1px solid #1a2e45; border-radius:8px; }
    [data-testid="stExpander"] summary { color:#a8c0d6 !important; font-weight:500; }
    [data-testid="stInfo"] { background:rgba(25,202,156,0.08); border-left:4px solid #19CA9C; border-radius:6px; }
    [data-testid="stWarning"] { background:rgba(255,180,0,0.08); border-left:4px solid #f0a500; border-radius:6px; }
    [data-testid="stError"]   { background:rgba(220,50,50,0.08);  border-left:4px solid #dc3232; border-radius:6px; }
    [data-testid="stSuccess"] { background:rgba(25,202,156,0.10); border-left:4px solid #19CA9C; border-radius:6px; }
    hr { border:none; border-top:1px solid #1a2e45; margin:1.5rem 0; }
    ::-webkit-scrollbar { width:6px; height:6px; }
    ::-webkit-scrollbar-track { background:#0d1b2a; }
    ::-webkit-scrollbar-thumb { background:#1e3a5f; border-radius:3px; }
    ::-webkit-scrollbar-thumb:hover { background:#19CA9C; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] p { color:#a8c0d6 !important; font-size:13px; }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="Live Water Cut Prediction", layout="wide", page_icon="🔮")

inject_lp_css()

# ─────────────────────────────────────────────
#  Load models
# ─────────────────────────────────────────────
MODEL_REGISTRY = [
    ("Random Forest",     "water_cut_model.pkl"),
    ("Gradient Boosting", "water_cut_model_gb.pkl"),
    ("Linear Regression", "water_cut_model_lr.pkl"),
]

available_models = {name: path for name, path in MODEL_REGISTRY if os.path.exists(path)}

try:    metadata = joblib.load("water_cut_metadata.pkl")
except: metadata = None

if not available_models or metadata is None:
    st.error("Model files not found. Ensure .pkl files are in the project root.")
    st.stop()

features     = metadata["features"]
well_columns = [c for c in metadata["well_columns"] if c.startswith("WELL_") and "/" in c]
well_names   = [c.replace("WELL_", "") for c in well_columns]

# ─────────────────────────────────────────────
#  Sidebar — model selector
# ─────────────────────────────────────────────
st.sidebar.header("⚙️ Model Configuration")
selected_model_name = st.sidebar.selectbox(
    "Choose Prediction Model",
    list(available_models.keys()),
    help=(
        "Random Forest → lowest MAE (4.37%), best average accuracy.\n\n"
        "Gradient Boosting → better for high water-cut values (≥ 80%).\n\n"
        "Linear Regression → simplest baseline."
    ),
)
model = joblib.load(available_models[selected_model_name])

if selected_model_name == "Random Forest":
    st.sidebar.warning(
        "When last month's water cut is ≥ 80%, Random Forest may underestimate. "
        "Switch to **Gradient Boosting** for high water-cut wells."
    )

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.title("🔮 Live Water Cut Prediction")
st.markdown(
    "Enter current-month production data for any Volve well to predict "
    "**next month's water cut (%)** using the ML model trained on Volve field history."
)
st.info(f"🤖 Active model: **{selected_model_name}**")

st.markdown("---")

# ─────────────────────────────────────────────
#  Input form — 4-column layout
# ─────────────────────────────────────────────
st.subheader("Well Input Data")

# Default values — Track B (mid-life, safe for all three models)
r1c1, r1c2, r1c3, r1c4 = st.columns(4)

with r1c1:
    selected_well  = st.selectbox("Wellbore", well_names)
    on_stream      = st.number_input("ON_STREAM (hrs)", min_value=0.0, max_value=744.0, value=620.0, step=1.0)
    oil            = st.number_input("OIL this month (m³)", min_value=0.0, value=35000.0, step=100.0)

with r1c2:
    gas            = st.number_input("GAS this month (m³)", min_value=0.0, value=1500000.0, step=1000.0)
    well_age       = st.number_input("WELL_AGE_MONTHS", min_value=1, value=36, step=1)
    cum_oil        = st.number_input("CUM_OIL (m³)", min_value=0.0, value=900000.0, step=1000.0)

with r1c3:
    water_cut_lag1 = st.slider(
        "Last month's WATER_CUT (%)",
        min_value=0.0, max_value=100.0, value=45.0, step=0.5,
        help="Most important feature (~92% importance). If ≥ 80% switch model to Gradient Boosting.",
    )
    oil_lag1       = st.number_input("Last month's OIL (m³)", min_value=0.0, value=38000.0, step=100.0)

with r1c4:
    st.markdown("**Injection (optional)**")
    gi = st.number_input("GI — Gas Injection (m³)",   min_value=0.0, value=0.0, step=100.0)
    wi = st.number_input("WI — Water Injection (m³)", min_value=0.0, value=20000.0, step=100.0)

    # Scenario quick-load guide
    with st.expander("Scenario reference values"):
        st.markdown("""
| Scenario | WC Lag | Well Age | CUM_OIL | Model |
|---|---|---|---|---|
| 🟢 Early life | 18% | 18 mo | 350,000 | RF |
| 🟡 Mid life | 45% | 36 mo | 900,000 | RF |
| 🔵 Stable | 45% | 36 mo | 700,000 | LR |
| 🔴 Late life | 82% | 72 mo | 2,000,000 | GB |
""")

# High water-cut warning
if selected_model_name == "Random Forest" and water_cut_lag1 >= 80:
    st.warning(
        f"Last month's water cut is **{water_cut_lag1:.0f}%**. "
        "Random Forest may underestimate in this range — consider switching to **Gradient Boosting**."
    )

st.markdown("---")

# ─────────────────────────────────────────────
#  Build input dataframe
# ─────────────────────────────────────────────
gor = 0.0 if oil == 0 else gas / oil

input_row = {
    "ON_STREAM":       on_stream,
    "OIL":             oil,
    "GAS":             gas,
    "GI":              gi,
    "WI":              wi,
    "GOR":             gor,
    "CUM_OIL":         cum_oil,
    "WELL_AGE_MONTHS": well_age,
    "WATER_CUT_LAG1":  water_cut_lag1,
    "OIL_LAG1":        oil_lag1,
}

for col in well_columns:
    col_normalized = col.replace("WELL_", "").strip()
    input_row[col] = 1 if col_normalized == selected_well.strip() else 0

input_df = pd.DataFrame(
    [[input_row.get(f, 0) for f in features]],
    columns=features,
)

# ─────────────────────────────────────────────
#  Predict
# ─────────────────────────────────────────────
if st.button("🔮 Predict Water Cut", type="primary", use_container_width=False):

    raw  = model.predict(input_df)[0]
    pred = float(np.clip(raw, 0, 100))

    # ── Result cards ──
    st.markdown("### Prediction Result")
    rc1, rc2, rc3 = st.columns([1, 1, 2])

    with rc1:
        st.metric(
            label=f"Predicted Water Cut — {selected_model_name}",
            value=f"{pred:.1f}%",
            delta=f"{pred - water_cut_lag1:+.1f}% vs last month",
        )

    with rc2:
        gor_display = f"{gor:,.0f}" if oil > 0 else "N/A (no oil)"
        st.metric("GOR (computed)", gor_display)
        st.metric("Well", selected_well)

    with rc3:
        if pred >= 70:
            st.error(
                f"**High water cut predicted ({pred:.1f}%)**\n\n"
                "Recommended action: review water shut-off options or evaluate EOR candidacy for this well. "
                "Water processing costs are likely significant at this level."
            )
        elif pred >= 40:
            st.warning(
                f"**Moderate water cut ({pred:.1f}%)**\n\n"
                "Continue routine monitoring. Consider tracking trend over the next 2–3 months "
                "to determine whether water cut is stabilising or accelerating."
            )
        else:
            st.success(
                f"**Low water cut ({pred:.1f}%)**\n\n"
                "Well is producing efficiently. No immediate water management action required."
            )

        if selected_model_name == "Random Forest" and water_cut_lag1 >= 80:
            st.info(
                "Tip: Random Forest may be underestimating here. "
                "Re-run with **Gradient Boosting** to compare results."
            )

    # ── Technical details (hidden by default) ──
    with st.expander("Technical details — input sent to model"):
        st.caption(f"GOR computed from inputs: {gor:,.2f} m³/m³")
        st.caption(f"Raw model output (before clip): {raw:.4f}%")
        st.dataframe(input_df, use_container_width=True)

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;padding:12px;opacity:0.75;'>
        <p style='font-size:13px;margin-bottom:4px;'>Developed by</p>
        <p style='font-size:16px;font-weight:bold;margin-bottom:2px;'>
            Mohamed Mustafa Abdelnabi Elbashir
        </p>
        <p style='font-size:13px;'>
            Petroleum Engineer &nbsp;|&nbsp; ML Engineer &nbsp;|&nbsp; Data Scientist
        </p>
        <p style='font-size:11px;margin-top:6px;color:#888;'>
            Models: RF MAE 4.37 | GB MAE 5.69 | LR MAE 5.10 &nbsp;·&nbsp;
            TimeSeriesSplit cross-validation on Volve monthly data
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
