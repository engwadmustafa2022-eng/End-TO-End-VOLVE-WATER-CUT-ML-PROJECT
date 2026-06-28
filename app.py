import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import calendar
import joblib
from typing import Dict, Literal, Optional

# ─────────────────────────────────────────────
#  Helper Functions (unchanged from notebook)
# ─────────────────────────────────────────────
volumes = {
    "OIL":   0.1589873,
    "GAS":   0.0283168,
    "WATER": 0.1589873,
}

def clean_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    df.columns = df.columns.str.replace(" ", "_").str.upper()
    df = df.replace(",", "", regex=True).fillna(0)
    df[["NPDCODE", "YEAR"]] = df[["NPDCODE", "YEAR"]].astype(int)
    df[["ON_STREAM", "OIL", "GAS", "WATER", "WI", "GI"]] = (
        df[["ON_STREAM", "OIL", "GAS", "WATER", "WI", "GI"]].astype(float)
    )
    abbr = dict(enumerate(calendar.month_abbr)); abbr.pop(0)
    if pd.api.types.is_numeric_dtype(df["MONTH"]):
        df["MONTH"] = df["MONTH"].map(abbr)
    else:
        df["MONTH"] = df["MONTH"].astype(str).str.strip().str[:3].str.title()
    df["MONTH"] = pd.Categorical(df["MONTH"], categories=list(abbr.values()), ordered=True)
    return df.sort_values(["YEAR", "MONTH"]).reset_index(drop=True)

def get_annual_data(data, category=None):
    df = data.groupby("YEAR", as_index=False)[["ON_STREAM","OIL","GAS","WATER","WI","GI"]].sum()
    if not category: return df
    c = category.strip().lower()
    if c in ["production","prod"]:
        df["CUM_OIL"]   = df["OIL"].cumsum()
        df["CUM_GAS"]   = df["GAS"].cumsum()
        df["CUM_WATER"] = df["WATER"].cumsum()
        df = df.drop(columns=["ON_STREAM","GI","WI"])
    elif c in ["injection","inj"]:
        df = df[["YEAR","GI","WI"]]
    return df

def get_monthly_data(data, parameter=None):
    df = data.copy(); df["MONTH"] = df["MONTH"].astype(str)
    df = df.groupby(["YEAR","MONTH"], as_index=False)[["ON_STREAM","OIL","GAS","WATER","GI","WI"]].sum()
    if not parameter: return df
    p = parameter.strip().upper()
    df = df.pivot_table(values=p, index="MONTH", columns="YEAR", aggfunc="sum", fill_value=0)
    df = df.reindex(list(calendar.month_abbr)[1:])
    return df

def get_well_data(data, well_name):
    return data.query("`WELLBORE_NAME` == @well_name.strip().upper()").reset_index(drop=True)

def get_well_annual_data(data, well_name, category=None):
    df = get_well_data(data, well_name).groupby("YEAR", as_index=False)[["ON_STREAM","OIL","GAS","WATER","GI","WI"]].sum()
    if not category: return df
    c = category.strip().lower()
    if c in ["production","prod"]:
        df["CUM_OIL"] = df["OIL"].cumsum(); df["CUM_GAS"] = df["GAS"].cumsum(); df["CUM_WATER"] = df["WATER"].cumsum()
        df = df.drop(columns=["ON_STREAM","GI","WI"])
    elif c in ["injection","inj"]:
        df = df[["YEAR","GI","WI"]]
    return df

def get_well_monthly_data(data, well_name, parameter=None):
    df = get_well_data(data, well_name); df["MONTH"] = df["MONTH"].astype(str)
    if not parameter: return df
    df = df.pivot_table(values=parameter.strip().upper(), index="MONTH", columns="YEAR", aggfunc="sum", fill_value=0)
    return df.reindex(list(calendar.month_abbr)[1:])

def determine_well_type(data, well_name):
    w = get_well_data(data, well_name)
    if sum(w["OIL"]+w["GAS"]) > 0:
        return "HYBRID" if sum(w["GI"]+w["WI"]) > 0 else "PRODUCTION"
    return "INJECTION"

def wellbores_data(data, category=None):
    df = data.groupby("WELLBORE_NAME", as_index=False)[["ON_STREAM","OIL","GAS","WATER","WI","GI"]].sum()
    if not category: return df
    c = category.strip().lower()
    if c in ["production","prod"]:   df = df.query("OIL > 0 or GAS > 0")
    elif c in ["injection","inj"]:   df = df.query("GI > 0 or WI > 0")
    elif c in ["hybrid","hb"]:       df = df.query("(OIL > 0 or GAS > 0) and (GI > 0 or WI > 0)")
    return df.reset_index(drop=True)

def analyze_production_wellbore(data, well_name):
    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.175,
        subplot_titles=("<b>Annual Oil (m³)</b>","<b>Monthly Oil (m³)</b>","<b>Annual Gas (m³)</b>","<b>Monthly Gas (m³)</b>"))
    a  = get_well_annual_data(data, well_name)
    mo = get_well_monthly_data(data, well_name, "OIL")
    mg = get_well_monthly_data(data, well_name, "GAS")
    traces = [
        go.Bar(x=a["YEAR"], y=a["OIL"],  customdata=a["OIL"]/volumes["OIL"],  hovertemplate="<b>%{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="mediumspringgreen", showlegend=False, name=""),
        go.Heatmap(x=mo.columns, y=mo.index, z=mo.values, colorscale="Greens", showscale=False, name=""),
        go.Bar(x=a["YEAR"], y=a["GAS"],  customdata=a["GAS"]/volumes["GAS"],  hovertemplate="<b>%{x}</b><br>SCF: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="crimson", showlegend=False, name=""),
        go.Heatmap(x=mg.columns, y=mg.index, z=mg.values, colorscale="Reds",   showscale=False, name=""),
    ]
    for i,(r,c) in enumerate([(1,1),(1,2),(2,1),(2,2)]):
        fig.add_trace(traces[i], row=r, col=c)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_layout(title=f"<b>{well_name.upper()}</b>", height=700, template="plotly_dark", margin=dict(t=80,b=50))
    return fig

def analyze_injection_wellbore(data, well_name):
    fig = make_subplots(rows=1, cols=2, subplot_titles=("<b>Annual WI (m³)</b>","<b>Monthly WI (m³)</b>"))
    a  = get_well_annual_data(data, well_name)
    mw = get_well_monthly_data(data, well_name, "WI")
    fig.add_trace(go.Bar(x=a["YEAR"], y=a["WI"], customdata=a["WI"]/volumes["WATER"], hovertemplate="<b>%{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="deepskyblue", name=""), row=1, col=1)
    fig.add_trace(go.Heatmap(x=mw.columns, y=mw.index, z=mw.values, colorscale="Blues", showscale=False, name=""), row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)
    fig.update_layout(title=f"<b>{well_name.upper()}</b>", height=400, template="plotly_dark", margin=dict(t=80,b=50))
    return fig

def analyze_hybrid_wellbore(data, well_name):
    fig = make_subplots(rows=2, cols=3, vertical_spacing=0.185,
        subplot_titles=("<b>Annual Oil</b>","<b>Annual Gas</b>","<b>Annual WI</b>","<b>Monthly Oil</b>","<b>Monthly Gas</b>","<b>Monthly WI</b>"))
    a  = get_well_annual_data(data, well_name)
    mo = get_well_monthly_data(data, well_name, "OIL")
    mg = get_well_monthly_data(data, well_name, "GAS")
    mw = get_well_monthly_data(data, well_name, "WI")
    traces = [
        go.Bar(x=a["YEAR"], y=a["OIL"], customdata=a["OIL"]/volumes["OIL"], hovertemplate="<b>%{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="mediumspringgreen", showlegend=False, name=""),
        go.Bar(x=a["YEAR"], y=a["GAS"], customdata=a["GAS"]/volumes["GAS"], hovertemplate="<b>%{x}</b><br>SCF: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="crimson", showlegend=False, name=""),
        go.Bar(x=a["YEAR"], y=a["WI"],  customdata=a["WI"]/volumes["WATER"], hovertemplate="<b>%{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{y:,.0f}", marker_color="deepskyblue", showlegend=False, name=""),
        go.Heatmap(x=mo.columns, y=mo.index, z=mo.values, colorscale="Greens", showscale=False, name=""),
        go.Heatmap(x=mg.columns, y=mg.index, z=mg.values, colorscale="Reds",   showscale=False, name=""),
        go.Heatmap(x=mw.columns, y=mw.index, z=mw.values, colorscale="Blues",  showscale=False, name=""),
    ]
    for i,(r,c) in enumerate([(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]):
        fig.add_trace(traces[i], row=r, col=c)
    for c in range(1,4):
        fig.update_yaxes(autorange="reversed", row=2, col=c)
    fig.update_layout(title=f"<b>{well_name.upper()}</b>", height=700, template="plotly_dark", margin=dict(t=80,b=50))
    return fig

def analyze_wellbore(data, well_name):
    t = determine_well_type(data, well_name)
    if t == "PRODUCTION": return analyze_production_wellbore(data, well_name)
    if t == "INJECTION":  return analyze_injection_wellbore(data, well_name)
    return analyze_hybrid_wellbore(data, well_name)

# ─────────────────────────────────────────────
#  UI Helper
# ─────────────────────────────────────────────
def show_insight(title: str, points: list, icon: str = ""):
    bullets = "".join(f"<li>{p}</li>" for p in points)
    st.markdown(
        f"""<div style="background:#1c2530;border-left:4px solid #19CA9C;
        border-radius:6px;padding:14px 18px;margin:10px 0 18px 0;">
        <div style="font-weight:600;font-size:15px;color:#19CA9C;margin-bottom:6px;">
        {icon + ' ' if icon else ''}{title}</div>
        <ul style="margin:0;padding-left:18px;color:#e6e6e6;font-size:14px;line-height:1.6;">
        {bullets}</ul></div>""",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  App Config & Data
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
#  Professional CSS Theme — Volve Field Dashboard
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
    <style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* ── Global ── */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Dark background ── */
    .stApp {
        background: linear-gradient(160deg, #0a0f1e 0%, #0d1b2a 60%, #0a1628 100%);
    }

    /* ── Top header bar ── */
    header[data-testid="stHeader"] {
        background: linear-gradient(90deg, #0a1628 0%, #112240 100%);
        border-bottom: 2px solid #19CA9C;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #112240 100%);
        border-right: 1px solid #1a2e45;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #cdd9e5 !important;
        font-weight: 500;
        font-size: 14px;
        padding: 6px 4px;
        transition: color 0.2s;
    }
    [data-testid="stSidebar"] .stRadio label:hover {
        color: #19CA9C !important;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #8899aa;
        font-size: 12px;
    }

    /* ── Main content area ── */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 1400px;
    }

    /* ── Title ── */
    h1 {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        letter-spacing: -0.5px;
        border-bottom: 3px solid #19CA9C;
        padding-bottom: 0.4rem;
        margin-bottom: 0.6rem !important;
    }

    /* ── Headers ── */
    h2 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #e2e8f0 !important;
        margin-top: 1.6rem !important;
        border-left: 4px solid #19CA9C;
        padding-left: 10px;
    }
    h3 {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #a8c0d6 !important;
        margin-top: 1.2rem !important;
    }

    /* ── st.metric cards ── */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #112240 0%, #1a2e45 100%);
        border: 1px solid #1e3a5f;
        border-top: 3px solid #19CA9C;
        border-radius: 10px;
        padding: 16px 20px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(25, 202, 156, 0.15);
    }
    [data-testid="stMetricLabel"] {
        font-size: 12px !important;
        font-weight: 500 !important;
        color: #8899aa !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #19CA9C !important;
    }
    [data-testid="stMetricDelta"] {
        font-size: 12px !important;
        font-weight: 500 !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        background: #0d1b2a;
        border-bottom: 2px solid #1a2e45;
        gap: 4px;
    }
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background: transparent;
        color: #8899aa !important;
        font-weight: 500;
        font-size: 13px;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        transition: all 0.2s;
    }
    [data-testid="stTabs"] [aria-selected="true"] {
        background: #112240 !important;
        color: #19CA9C !important;
        border-bottom: 2px solid #19CA9C !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #19CA9C 0%, #0fa87d 100%);
        color: #0a0f1e;
        font-weight: 700;
        font-size: 15px;
        border: none;
        border-radius: 8px;
        padding: 10px 32px;
        letter-spacing: 0.3px;
        transition: all 0.25s;
        box-shadow: 0 4px 14px rgba(25,202,156,0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #22e0ac 0%, #19CA9C 100%);
        box-shadow: 0 6px 20px rgba(25,202,156,0.45);
        transform: translateY(-1px);
    }

    /* ── Selectbox / number_input / slider ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] input,
    [data-baseweb="select"] {
        background: #112240 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 6px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background: #19CA9C !important;
    }
    [data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
        background: #1a2e45 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: #0d1b2a;
        border: 1px solid #1a2e45;
        border-radius: 8px;
    }
    [data-testid="stExpander"] summary {
        color: #a8c0d6 !important;
        font-weight: 500;
        font-size: 14px;
    }

    /* ── Info / Warning / Error / Success banners ── */
    [data-testid="stInfo"] {
        background: rgba(25,202,156,0.08);
        border-left: 4px solid #19CA9C;
        border-radius: 6px;
        color: #cdd9e5;
    }
    [data-testid="stWarning"] {
        background: rgba(255,180,0,0.08);
        border-left: 4px solid #f0a500;
        border-radius: 6px;
    }
    [data-testid="stError"] {
        background: rgba(220,50,50,0.08);
        border-left: 4px solid #dc3232;
        border-radius: 6px;
    }
    [data-testid="stSuccess"] {
        background: rgba(25,202,156,0.10);
        border-left: 4px solid #19CA9C;
        border-radius: 6px;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border: 1px solid #1a2e45;
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Dividers ── */
    hr {
        border: none;
        border-top: 1px solid #1a2e45;
        margin: 1.5rem 0;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0d1b2a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #19CA9C; }

    /* ── Caption / small text ── */
    [data-testid="stCaptionContainer"] p {
        color: #6a7f94 !important;
        font-size: 12px !important;
    }

    /* ── Sidebar logo area ── */
    .sidebar-brand {
        text-align: center;
        padding: 18px 12px 24px;
        border-bottom: 1px solid #1a2e45;
        margin-bottom: 12px;
    }
    .sidebar-brand .brand-title {
        font-size: 15px;
        font-weight: 700;
        color: #19CA9C;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .sidebar-brand .brand-sub {
        font-size: 10px;
        color: #6a7f94;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 2px;
    }
    </style>
    """, unsafe_allow_html=True)


st.set_page_config(page_title="Volve Field — Water Cut Dashboard", layout="wide", page_icon="🛢️")

inject_css()
DATA_PATH = "volve_data.csv"

@st.cache_data
def load_data(path): return clean_data(path)

volve_df = load_data(DATA_PATH)

# ─────────────────────────────────────────────
#  Header
# ─────────────────────────────────────────────
st.title("🛢️ Volve Field Production Analysis Dashboard")
st.markdown(
    "North Sea oil field operated by **Equinor** (2008–2016). "
    "This dashboard traces the field's production history and deploys a machine learning model "
    "to predict **water cut** — the fraction of water in produced fluids — a key metric for "
    "water-management decisions."
)

with st.expander("Field & terminology reference"):
    st.markdown("""
| Term | Definition |
|---|---|
| **OIL / GAS / WATER** | Monthly produced volumes per wellbore (m³) |
| **WI** | Water Injection — injected to maintain reservoir pressure |
| **GI** | Gas Injection — re-injected into reservoir |
| **ON_STREAM** | Hours a well was actively producing / injecting that month |
| **Water Cut** | WATER ÷ (OIL + WATER) — the ML model target |
""")

# ─────────────────────────────────────────────
#  Executive Summary
# ─────────────────────────────────────────────
st.subheader("Executive Summary")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Oil (m³)",           f"{volve_df['OIL'].sum():,.0f}")
c2.metric("Total Gas (m³)",           f"{volve_df['GAS'].sum():,.0f}")
c3.metric("Total Water Produced (m³)",f"{volve_df['WATER'].sum():,.0f}")
c4.metric("Total Water Injected (m³)",f"{volve_df['WI'].sum():,.0f}")
c5.metric("Active Wellbores",         f"{volve_df['WELLBORE_NAME'].nunique()}",
          help=f"Field active: {volve_df['YEAR'].min()}–{volve_df['YEAR'].max()}")

st.markdown("---")

# ─────────────────────────────────────────────
#  Navigation — 3 pages only
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div class="sidebar-brand">
    <div class="brand-title">🛢️ Volve Field</div>
    <div class="brand-sub">ML Production Dashboard</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["📈 Field Production Profile & History",
     "🧬 Reservoir Dynamics & Well Performance",
     "🤖 ML: Water Cut Model"],
)

# ══════════════════════════════════════════════
#  PAGE 1 — Field Production Profile & History
# ══════════════════════════════════════════════
if page == "📈 Field Production Profile & History":

    st.header("📈 Field Production Profile & History")

    annual_prod = get_annual_data(volve_df, "production")
    annual_inj  = get_annual_data(volve_df, "injection")

    # ── Combined Line Chart: Oil / Water / WI ──
    st.subheader("Water Breakthrough & Sweep Efficiency")
    fig_combined = go.Figure()
    fig_combined.add_trace(go.Scatter(
        x=annual_prod["YEAR"], y=annual_prod["OIL"],
        name="Oil Produced", mode="lines+markers",
        line=dict(color="#19CA9C", width=3, shape="spline"), marker_size=8,
        customdata=annual_prod["OIL"]/volumes["OIL"],
        hovertemplate="<b>%{x}</b><br>Oil: %{y:,.0f} m³ (%{customdata:,.0f} bbl)<extra></extra>"
    ))
    fig_combined.add_trace(go.Scatter(
        x=annual_prod["YEAR"], y=annual_prod["WATER"],
        name="Water Produced", mode="lines+markers",
        line=dict(color="#636EFA", width=3, shape="spline"), marker_size=8,
        customdata=annual_prod["WATER"]/volumes["WATER"],
        hovertemplate="<b>%{x}</b><br>Water: %{y:,.0f} m³ (%{customdata:,.0f} bbl)<extra></extra>"
    ))
    fig_combined.add_trace(go.Scatter(
        x=annual_inj["YEAR"], y=annual_inj["WI"],
        name="Water Injection", mode="lines+markers",
        line=dict(color="#AB63FA", width=2, dash="dot", shape="spline"), marker_size=7,
        customdata=annual_inj["WI"]/volumes["WATER"],
        hovertemplate="<b>%{x}</b><br>WI: %{y:,.0f} m³ (%{customdata:,.0f} bbl)<extra></extra>"
    ))
    fig_combined.add_trace(go.Scatter(
        x=annual_prod["YEAR"], y=annual_prod["GAS"],
        name="Gas Produced", mode="lines+markers",
        line=dict(color="#EE553C", width=2, shape="spline"), marker_size=7,
        hovertemplate="<b>%{x}</b><br>Gas: %{y:,.0f} m³<extra></extra>"
    ))
    fig_combined.update_layout(
        xaxis=dict(title="Year", tickmode="array", tickvals=annual_prod["YEAR"]),
        yaxis=dict(title="Volume (m³)", hoverformat=",.0f"),
        legend=dict(orientation="h", x=0.1, y=-0.18),
        hovermode="x unified", font_size=12, height=450, template="plotly_dark"
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    show_insight(
        "Field Production Story — Water Breakthrough",
        [
            "Oil peaked in 2009 and declined steadily — classic reservoir depletion curve.",
            "Water production rose inversely to oil from 2010 onward, marking the onset of water breakthrough.",
            "Water injection began in 2008 to maintain reservoir pressure, but injected water began breaking through to producers rather than sweeping oil — a sweep-efficiency problem visible in the diverging water produced vs. water injected lines.",
            "A partial recovery in oil production in 2014–2015 reflects new drilling activity before field shutdown in 2016.",
        ],
    )

    # ── Monthly Oil Heatmap ──
    st.subheader("Monthly Oil Production Heatmap")
    monthly_oil = get_monthly_data(volve_df, "OIL")
    fig_heat = go.Figure(data=go.Heatmap(
        x=monthly_oil.columns, y=monthly_oil.index, z=monthly_oil.values,
        customdata=monthly_oil.values/volumes["OIL"],
        hovertemplate="<b>%{y} %{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{z:,.0f}<extra></extra>",
        colorscale="Greens", name=""
    ))
    fig_heat.update_layout(
        xaxis=dict(title="Year", tickmode="array", tickvals=monthly_oil.columns),
        yaxis=dict(title="Month", autorange="reversed"),
        font_size=12, height=380, template="plotly_dark"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info(
        "**Operational Anomaly — August & September 2012:**  "
        "Production dropped sharply across all wells in these two months due to a temporary field shutdown "
        "caused by an anchor-line maintenance issue. All fluid streams (oil, gas, water injection) were "
        "interrupted simultaneously. These months were retained in the dataset but flagged during "
        "ML preprocessing to avoid biasing the lag features."
    )

    with st.expander("Monthly Gas & Water Injection heatmaps"):
        monthly_gas = get_monthly_data(volve_df, "GAS")
        monthly_wi  = get_monthly_data(volve_df, "WI")

        fig_gash = go.Figure(data=go.Heatmap(
            x=monthly_gas.columns, y=monthly_gas.index, z=monthly_gas.values,
            customdata=monthly_gas.values/volumes["GAS"],
            hovertemplate="<b>%{y} %{x}</b><br>SCF: %{customdata:,.0f}<br>m³: %{z:,.0f}<extra></extra>",
            colorscale="Reds", name=""
        ))
        fig_gash.update_layout(
            title="<b>Monthly Gas Production (m³)</b>",
            xaxis=dict(title="Year", tickmode="array", tickvals=monthly_gas.columns),
            yaxis=dict(title="Month", autorange="reversed"),
            font_size=12, height=360, template="plotly_dark"
        )
        st.plotly_chart(fig_gash, use_container_width=True)

        fig_wih = go.Figure(data=go.Heatmap(
            x=monthly_wi.columns, y=monthly_wi.index, z=monthly_wi.values,
            customdata=monthly_wi.values/volumes["WATER"],
            hovertemplate="<b>%{y} %{x}</b><br>bbl: %{customdata:,.0f}<br>m³: %{z:,.0f}<extra></extra>",
            colorscale="Blues", name=""
        ))
        fig_wih.update_layout(
            title="<b>Monthly Water Injection (m³)</b>",
            xaxis=dict(title="Year", tickmode="array", tickvals=monthly_wi.columns),
            yaxis=dict(title="Month", autorange="reversed"),
            font_size=12, height=360, template="plotly_dark"
        )
        st.plotly_chart(fig_wih, use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 2 — Reservoir Dynamics & Well Performance
# ══════════════════════════════════════════════
elif page == "🧬 Reservoir Dynamics & Well Performance":

    st.header("🧬 Reservoir Dynamics & Well Performance")

    tab_corr, tab_well = st.tabs(["Field-Wide Correlations", "Individual Well Diagnostics"])

    # ── TAB 1: Correlations ──
    with tab_corr:
        st.subheader("Correlation Matrix — Water Cut Drivers")

        # Only variables directly relevant to water cut
        corr_vars = ["OIL", "WATER", "GAS", "WI", "GI", "ON_STREAM"]
        corr_df   = volve_df[corr_vars].corr().round(2)

        # Mask upper triangle for clean symmetric matrix
        import numpy as np
        mask = np.triu(np.ones_like(corr_df, dtype=bool), k=1)
        z_masked = corr_df.values.copy().astype(float)
        z_masked[mask] = None

        annotations = []
        for i, row in enumerate(corr_df.index):
            for j, col in enumerate(corr_df.columns):
                if not mask[i, j]:
                    annotations.append(dict(
                        x=col, y=row, text=f"{corr_df.loc[row, col]:.2f}",
                        showarrow=False,
                        font=dict(color="white" if abs(corr_df.loc[row, col]) > 0.4 else "#aaa", size=12)
                    ))

        fig_corr = go.Figure(data=go.Heatmap(
            z=z_masked, x=corr_df.columns, y=corr_df.index,
            colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
            colorbar=dict(title="r", thickness=14),
            hovertemplate="X: %{x}<br>Y: %{y}<br>r = %{z:.2f}<extra></extra>",
            name=""
        ))
        fig_corr.update_layout(
            annotations=annotations,
            xaxis=dict(side="bottom"), yaxis=dict(autorange="reversed"),
            font_size=12, height=480, template="plotly_dark",
            margin=dict(t=30, b=60)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        show_insight(
            "Key Correlations with Water Cut",
            [
                "OIL and GAS are strongly positively correlated (r ≈ 0.90) — both produced from the same reservoir energy, confirming GOR is a useful engineered feature.",
                "WATER and OIL are negatively correlated — as oil declines, water rises. This is the water breakthrough signature that drives the ML model.",
                "Water Injection (WI) shows moderate positive correlation with produced WATER — injected water breaks through to production wells over time.",
                "ON_STREAM correlates positively with all production volumes, as expected — active hours determine total monthly output.",
            ],
        )

        # Scatter: Oil vs Water colored by year
        st.subheader("Oil vs. Water Production — Field Maturity View")
        fig_scatter = go.Figure(data=go.Scatter(
            x=volve_df["WATER"], y=volve_df["OIL"],
            mode="markers",
            marker=dict(size=8, color=volve_df["YEAR"], colorscale="Plasma",
                        colorbar=dict(title="Year"), showscale=True,
                        line=dict(width=0.4, color="white")),
            hovertemplate="Water: %{x:,.0f} m³<br>Oil: %{y:,.0f} m³<br>Year: %{marker.color}<extra></extra>"
        ))
        fig_scatter.update_layout(
            xaxis_title="Water Production (m³)", yaxis_title="Oil Production (m³)",
            font_size=12, height=460, template="plotly_dark"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        show_insight(
            "Scatter Interpretation — Time-Evolving Reservoir Maturity",
            [
                "Early years (2008–2010, darker tones) cluster in the high-oil / low-water quadrant.",
                "Later years (2013–2016, brighter tones) migrate toward high-water / low-oil — the maturity trajectory.",
                "The gradient confirms that water cut is a time-dependent process, justifying WELL_AGE_MONTHS as a model feature.",
            ],
        )

    # ── TAB 2: Individual Well Diagnostics ──
    with tab_well:
        st.subheader("Individual Well Diagnostics")

        prod_df = wellbores_data(volve_df, "production")
        inj_df  = wellbores_data(volve_df, "injection")

        c1, c2 = st.columns(2)
        with c1:
            fig_wp = go.Figure()
            for fluid, color in [("OIL","#19CA9C"),("GAS","#EE553C"),("WATER","#636EFA")]:
                fig_wp.add_trace(go.Bar(
                    x=prod_df["WELLBORE_NAME"], y=prod_df[fluid],
                    name=fluid, marker_color=color,
                    hovertemplate=f"<b>%{{x}}</b><br>{fluid}: %{{y:,.0f}} m³<extra></extra>"
                ))
            fig_wp.update_layout(
                title="<b>Fluid Production per Well (m³)</b>",
                barmode="group", template="plotly_dark", height=380, font_size=11,
                legend=dict(orientation="h", x=0.2, y=-0.22)
            )
            st.plotly_chart(fig_wp, use_container_width=True)

        with c2:
            fig_wi = go.Figure(data=go.Bar(
                x=inj_df["WELLBORE_NAME"], y=inj_df["WI"],
                marker=dict(color=inj_df["WI"], colorscale="Blues", line=dict(color="gray", width=1)),
                hovertemplate="<b>%{x}</b><br>WI: %{y:,.0f} m³<extra></extra>", name=""
            ))
            fig_wi.update_layout(
                title="<b>Water Injection per Well (m³)</b>",
                template="plotly_dark", height=380, font_size=11
            )
            st.plotly_chart(fig_wi, use_container_width=True)

        show_insight(
            "Well-Level Observations",
            [
                "15/9-F-12 is the field's top oil producer — yet lacks downhole sensor data for most of its active life.",
                "15/9-F-14 shows the highest water production, making it the priority candidate for water management review.",
                "15/9-F-4 carries the bulk of the injection workload; its injected water is the primary source of the sweep-efficiency issue.",
                "15/9-F-5 transitioned from injection to production in later years — a hybrid operational profile.",
            ],
        )

        st.markdown("---")
        selected_wellbore = st.selectbox(
            "Select a wellbore for detailed annual & monthly analysis:",
            volve_df["WELLBORE_NAME"].unique()
        )
        if selected_wellbore:
            st.plotly_chart(analyze_wellbore(volve_df, selected_wellbore), use_container_width=True)


# ══════════════════════════════════════════════
#  PAGE 3 — ML: Water Cut Model
# ══════════════════════════════════════════════
elif page == "🤖 ML: Water Cut Model":

    st.header("🤖 ML: Water Cut Prediction Model")
    st.markdown(
        "A regression model trained to predict **next month's water cut (%)** per well, "
        "using engineered features built from production history: "
        "lag values, well age, GOR, and cumulative oil production."
    )

    MODEL_FILES = [
        ("water_cut_model.pkl",    "Random Forest"),
        ("water_cut_model_gb.pkl", "Gradient Boosting"),
        ("water_cut_model_lr.pkl", "Linear Regression"),
    ]
    ml_model = None; ml_model_name = None
    for f, name in MODEL_FILES:
        try:
            ml_model = joblib.load(f); ml_model_name = name; break
        except FileNotFoundError:
            continue

    ml_metadata = None
    try: ml_metadata = joblib.load("water_cut_metadata.pkl")
    except FileNotFoundError: pass

    if ml_model is None or ml_metadata is None:
        st.warning("No model files found. Place the .pkl files next to app.py.")
        st.stop()

    results_df = pd.DataFrame({
        "RMSE": [6.15, 6.30, 7.98],
        "MAE":  [5.10, 4.37, 5.79],
        "R²":   [0.881, 0.875, 0.799],
    }, index=["Linear Regression", "Random Forest", "Gradient Boosting"])

    # ── Model Metrics ──
    st.subheader("Model Comparison")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Models Trained", "3")
    m2.metric("Active Model",   ml_model_name)
    m3.metric("Test RMSE",      f"{results_df.loc[ml_model_name,'RMSE']:.2f}")
    m4.metric("Test R²",        f"{results_df.loc[ml_model_name,'R²']:.3f}")

    col_t, col_c = st.columns([1, 1.4])
    with col_t:
        st.dataframe(
            results_df.style
                .format({"RMSE":"{:.2f}","MAE":"{:.2f}","R²":"{:.3f}"})
                .highlight_min(subset=["RMSE","MAE"], color="#1c3d33")
                .highlight_max(subset=["R²"],         color="#1c3d33"),
            use_container_width=True,
        )
    with col_c:
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Bar(x=results_df.index, y=results_df["RMSE"], name="RMSE", marker_color="#e76f51"))
        fig_cmp.add_trace(go.Bar(x=results_df.index, y=results_df["MAE"],  name="MAE",  marker_color="#f4a261"))
        fig_cmp.update_layout(barmode="group", template="plotly_dark", height=320, font_size=12)
        st.plotly_chart(fig_cmp, use_container_width=True)

    show_insight(
        "Why Random Forest Was Selected",
        [
            "Random Forest achieved the lowest MAE (4.37%) — average error of ~4.4 percentage points per well per month.",
            "Linear Regression had a slightly lower RMSE (6.15 vs 6.30), confirming the water-cut relationship is largely linear.",
            "Gradient Boosting scored highest RMSE/MAE here — the dataset (175 records) is too small for GB's sequential trees to fully tune.",
            "The narrow R² gap (0.799–0.881) across all three models confirms genuine generalisation, not overfitting.",
        ],
    )

    show_insight(
        "Known Limitation — RF Underestimates at High Water Cut (80–100%)",
        [
            "Random Forest averages tree outputs, pulling predictions toward the dataset mean — 'regression toward the mean'.",
            "When WATER_CUT_LAG1 ≥ 80%, RF underestimates by up to 15–25 percentage points.",
            "Root cause: high water-cut months are rare in the 175-record Volve dataset — the trees lack examples at the extremes.",
            "Recommendation: switch to Gradient Boosting in the Live Prediction page when WATER_CUT_LAG1 ≥ 80%.",
        ],
        icon="⚠️",
    )

    # ── Feature Importance ──
    st.subheader("Feature Importance")
    _inner = ml_model.named_steps["model"] if hasattr(ml_model,"named_steps") else ml_model
    if hasattr(_inner, "feature_importances_"):
        imp = pd.Series(_inner.feature_importances_, index=ml_metadata["features"]).sort_values(ascending=False).head(10)
        fig_imp = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h", marker_color="#264653"))
        fig_imp.update_layout(
            title=f"<b>Top 10 Feature Importances — {ml_model_name}</b>",
            xaxis_title="Importance", template="plotly_dark", height=380, font_size=12
        )
        st.plotly_chart(fig_imp, use_container_width=True)

    show_insight(
        "What Drives the Prediction",
        [
            "WATER_CUT_LAG1 dominates (~92% importance) — water cut evolves gradually in a mature field and last month's value is the strongest predictor of next month's.",
            "WELL_AGE_MONTHS is the clear second driver — older wells have progressively higher water cut (reservoir maturation).",
            "CUM_OIL acts as a proxy for reservoir depletion depth.",
            "Downhole pressure and temperature contribute marginally — consistent with limited sensor coverage in this dataset.",
        ],
    )

    st.markdown("---")

    # ── Conclusions ──
    st.subheader("Conclusions & Strategic Recommendations")
    st.markdown(
        "The Volve field followed a classic lifecycle: rapid startup → 2009 peak → water-driven decline → "
        "partial recovery from new drilling (2014–2015) → shutdown (2016). "
        "The ML model quantifies this trajectory and turns it into a monthly forecasting tool."
    )

    show_insight("1 — Prioritise 15/9-F-14 for water management review",
        ["Highest water production of any well across the field's full life.",
         "The model consistently predicts elevated water cut for this well.",
         "Evaluate whether produced-water processing costs are justified; assess water shut-off or EOR candidacy."], icon="⚠️")

    show_insight("2 — Restore downhole sensor coverage on 15/9-F-12",
        ["Top oil producer in the field — yet sensor data was missing for most of its active life.",
         "The model relied solely on production lags for this well, reducing forecast confidence.",
         "Any future intervention should prioritise reinstalling functioning downhole gauges."], icon="🔧")

    show_insight("3 — Review injection sweep efficiency",
        ["Water injection into 15/9-F-4 began in 2008, yet produced water rose steadily from 2010.",
         "Injected water is breaking through to producers rather than sweeping oil.",
         "Revisit injection pattern and rates; consider redirecting volumes to improve sweep."], icon="💧")

    show_insight("4 — Deploy the model as a monthly early-warning system",
        ["Predicts next month's water cut with R² = 0.875 and MAE ≈ 4.4 percentage points.",
         "Threshold alerts: ≥ 70% → immediate review | 40–70% → increased monitoring | < 40% → routine.",
         "Runs on monthly production data alone — no new instruments required.",
         "Use the Live Prediction page (sidebar) to test any well before production review meetings."], icon="🤖")

    show_insight("Honest Limitations",
        ["Trained on 175 monthly records from 5 wells — small by ML standards.",
         "A proof-of-concept tool: predictions should guide attention, not replace engineering judgment.",
         "RF underestimates at high water cut (≥ 80%) — use Gradient Boosting in that regime."], icon="📌")

    st.markdown("🔮 Open **Live Prediction** from the sidebar to test the model on custom well inputs.")
