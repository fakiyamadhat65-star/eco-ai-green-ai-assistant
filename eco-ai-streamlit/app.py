"""
ECO-AI Streamlit App
Green Grid Intelligence — ML-powered, real datasets, interactive dashboards
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECO-AI · Green Grid Intelligence",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Theme CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700&family=Rajdhani:wght@400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Rajdhani', sans-serif; }

.main { background: #050a0e; }

.stApp { background: #050a0e; color: #c8dde8; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0a1520 !important;
    border-right: 1px solid #1a3a50;
}
section[data-testid="stSidebar"] * { color: #c8dde8 !important; }

/* Cards */
div[data-testid="metric-container"] {
    background: #0d1e2e;
    border: 1px solid #1a3a50;
    border-radius: 8px;
    padding: 12px;
    border-top: 2px solid #00ff87;
}

/* Inputs */
.stSlider > div > div { background: #1a3a50; }
.stSelectbox > div > div { background: #0d1e2e; border: 1px solid #1a3a50; }
.stNumberInput > div > div { background: #0d1e2e; border: 1px solid #1a3a50; }

/* Headers */
h1, h2, h3 { font-family: 'Orbitron', monospace !important; color: #00ff87 !important; }
h1 { font-size: 1.6rem !important; }
h3 { font-size: 1rem !important; color: #00c8ff !important; }

/* Prediction box */
.pred-box {
    background: #0d1e2e;
    border: 1px solid #1a3a50;
    border-radius: 8px;
    padding: 20px;
    text-align: center;
    border-top: 3px solid #00ff87;
}
.pred-class {
    font-family: 'Orbitron', monospace;
    font-size: 2rem;
    font-weight: 700;
}
.rec-good { background:#0d2a1a; border-left:4px solid #00ff87; padding:10px 14px; border-radius:4px; color:#00ff87; }
.rec-warn { background:#2a1f0d; border-left:4px solid #ffd700; padding:10px 14px; border-radius:4px; color:#ffd700; }
.rec-bad  { background:#2a0d0d; border-left:4px solid #ff4444; padding:10px 14px; border-radius:4px; color:#ff4444; }

.impact-card {
    background:#0d1e2e; border:1px solid #1a3a50; border-radius:8px;
    padding:16px; text-align:center;
}
.impact-val { font-family:'Orbitron',monospace; font-size:1.5rem; font-weight:700; color:#ffd700; }
.impact-lbl { font-size:0.75rem; color:#4a6a7a; font-family:'Share Tech Mono',monospace; }

.section-header {
    font-family:'Orbitron',monospace; font-size:0.7rem; letter-spacing:3px;
    text-transform:uppercase; color:#4a6a7a; margin-bottom:4px;
    border-bottom:1px solid #1a3a50; padding-bottom:6px;
}

div[data-testid="stMetric"] label { color: #4a6a7a !important; font-size:0.7rem !important; font-family:'Share Tech Mono',monospace !important; }
div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00ff87 !important; font-family:'Orbitron',monospace !important; font-size:1.4rem !important; }
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] { font-size:0.75rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Data path ─────────────────────────────────────────────────────────────────
BASE = Path(__file__).parent
DATA = BASE / "data"

# ── Load & cache data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    grid  = pd.read_csv(DATA / "portugal_grid_data.csv",  parse_dates=["timestamp"])
    runs  = pd.read_csv(DATA / "training_runs_log.csv",   parse_dates=["timestamp"])
    cc    = pd.read_csv(DATA / "country_comparison.csv",  parse_dates=["timestamp"])
    budg  = pd.read_csv(DATA / "carbon_budget.csv")
    return grid, runs, cc, budg

@st.cache_resource
def train_models(grid, runs):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score

    grid = grid.copy().sort_values("timestamp").reset_index(drop=True)
    grid["ci_lag1"]  = grid["carbon_intensity"].shift(1)
    grid["ci_lag3"]  = grid["carbon_intensity"].shift(3)
    grid["ci_lag6"]  = grid["carbon_intensity"].shift(6)
    grid["rp_lag1"]  = grid["renewable_pct"].shift(1)
    grid["rp_lag3"]  = grid["renewable_pct"].shift(3)
    grid["is_night"] = ((grid["hour"]>=22)|(grid["hour"]<=5)).astype(int)
    grid["is_peak"]  = ((grid["hour"]>=8) &(grid["hour"]<=20)).astype(int)
    grid["ci_trend"] = grid["carbon_intensity"] - grid["ci_lag3"]
    grid = grid.dropna().reset_index(drop=True)

    def ci_label(v):
        if v<50:   return "Very Low"
        if v<100:  return "Low"
        if v<200:  return "Moderate"
        if v<350:  return "High"
        if v<500:  return "Very High"
        return "Extreme"

    def rp_label(v):
        if v<20:  return "Very Low"
        if v<40:  return "Low"
        if v<60:  return "Moderate"
        if v<75:  return "High"
        if v<90:  return "Very High"
        return "Exceptional"

    def gs_label(v):
        if v>=90: return "Excellent"
        if v>=75: return "Very Good"
        if v>=55: return "Good"
        if v>=35: return "Fair"
        return "Poor"

    grid["ci_class"] = grid["carbon_intensity"].apply(ci_label)
    grid["rp_class"] = grid["renewable_pct"].apply(rp_label)
    grid["gs_class"] = grid["green_score"].apply(gs_label)

    models, encoders, stats = {}, {}, {}

    # 1. CI Classifier
    f1 = ["ci_lag1","ci_lag3","ci_lag6","ci_trend","renewable_pct","temperature_c",
          "wind_speed_ms","hour","month","day_of_week","grid_balance_mw","is_night","is_peak"]
    le1 = LabelEncoder()
    y1  = le1.fit_transform(grid["ci_class"])
    X1  = grid[f1]
    Xt1,Xv1,yt1,yv1 = train_test_split(X1,y1,test_size=0.2,random_state=42)
    m1 = RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1)
    m1.fit(Xt1,yt1)
    models["ci"]=m1; encoders["ci"]=le1
    stats["ci"]={"acc":accuracy_score(yv1,m1.predict(Xv1))*100,"features":f1,"classes":list(le1.classes_)}

    # 2. RP Classifier
    f2 = ["rp_lag1","rp_lag3","carbon_intensity","wind_speed_ms","cloud_cover_pct",
          "solar_mw","wind_mw","hour","month","is_night"]
    le2 = LabelEncoder()
    y2  = le2.fit_transform(grid["rp_class"])
    X2  = grid[f2]
    Xt2,Xv2,yt2,yv2 = train_test_split(X2,y2,test_size=0.2,random_state=42)
    m2 = RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1)
    m2.fit(Xt2,yt2)
    models["rp"]=m2; encoders["rp"]=le2
    stats["rp"]={"acc":accuracy_score(yv2,m2.predict(Xv2))*100,"features":f2,"classes":list(le2.classes_)}

    # 3. GS Classifier
    f3 = ["carbon_intensity","renewable_pct","wind_speed_ms","temperature_c",
          "hour","month","is_night","is_peak","grid_balance_mw"]
    le3 = LabelEncoder()
    y3  = le3.fit_transform(grid["gs_class"])
    X3  = grid[f3]
    Xt3,Xv3,yt3,yv3 = train_test_split(X3,y3,test_size=0.2,random_state=42)
    m3 = RandomForestClassifier(n_estimators=150,max_depth=12,random_state=42,n_jobs=-1)
    m3.fit(Xt3,yt3)
    models["gs"]=m3; encoders["gs"]=le3
    stats["gs"]={"acc":accuracy_score(yv3,m3.predict(Xv3))*100,"features":f3,"classes":list(le3.classes_)}

    # 4. CO2 Regressor
    f4 = ["gpu_count","gpu_tdp_w","duration_h","pue","carbon_intensity"]
    runs_c = runs.dropna(subset=f4+["co2_kg"])
    X4=runs_c[f4]; y4=runs_c["co2_kg"]
    Xt4,Xv4,yt4,yv4 = train_test_split(X4,y4,test_size=0.2,random_state=42)
    m4 = GradientBoostingRegressor(n_estimators=200,max_depth=5,random_state=42)
    m4.fit(Xt4,yt4)
    p4 = m4.predict(Xv4)
    models["co2"]=m4
    stats["co2"]={"mae":mean_absolute_error(yv4,p4),"r2":r2_score(yv4,p4),"features":f4}

    models["grid_engineered"] = grid
    return models, encoders, stats

# ── Plotly theme helper ────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#050a0e", plot_bgcolor="#0d1e2e",
    font=dict(color="#c8dde8", family="Rajdhani"),
    margin=dict(l=40,r=20,t=30,b=40)
)

def apply_theme(fig):
    fig.update_xaxes(gridcolor="#1a3a50", zerolinecolor="#1a3a50")
    fig.update_yaxes(gridcolor="#1a3a50", zerolinecolor="#1a3a50")
    return fig

CI_COLORS = {
    "Very Low":"#00ff87","Low":"#7fff00","Moderate":"#ffd700",
    "High":"#ff8c00","Very High":"#ff4444","Extreme":"#ff0066"
}
RP_COLORS = {
    "Very Low":"#ff4444","Low":"#ff8c00","Moderate":"#ffd700",
    "High":"#7fff00","Very High":"#00ff87","Exceptional":"#00ffff"
}
GS_COLORS = {
    "Poor":"#ff4444","Fair":"#ff8c00","Good":"#ffd700",
    "Very Good":"#7fff00","Excellent":"#00ff87"
}

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:0 0 16px'>
      <div style='font-family:Orbitron,monospace;font-size:1.2rem;font-weight:800;color:#00ff87;letter-spacing:3px'>ECO-AI</div>
      <div style='font-family:"Share Tech Mono",monospace;font-size:0.65rem;color:#4a6a7a;letter-spacing:2px'>GREEN GRID INTELLIGENCE</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.selectbox("", [
        "🏠  Overview Dashboard",
        "🌡  Carbon Intensity",
        "🌿  Renewable % Predictor",
        "🏅  Green Score",
        "💨  CO₂ Predictor",
        "⏰  Smart Scheduler",
        "🔍  Anomaly Detector",
        "🌍  Country Comparator",
        "📊  Model Performance",
        "📈  Dataset Explorer",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-family:"Share Tech Mono",monospace;font-size:0.65rem;color:#4a6a7a;line-height:2'>
    <div>Grid Records: <span style='color:#00ff87'>17,520</span></div>
    <div>ML Models: <span style='color:#00ff87'>4 trained</span></div>
    <div>Countries: <span style='color:#00ff87'>8</span></div>
    <div>Training Runs: <span style='color:#00ff87'>300</span></div>
    <div style='margin-top:8px'>Status: <span style='color:#00ff87'>● LIVE</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Load data + train ──────────────────────────────────────────────────────────
with st.spinner("⚙ Loading datasets & training ML models..."):
    grid, runs, cc, budg = load_data()
    models, encoders, stats = train_models(grid, runs)
    grid_eng = models["grid_engineered"]

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: OVERVIEW DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown("## 🌍 ECO-AI · Overview Dashboard")
    st.markdown("<div class='section-header'>REAL-TIME GRID INTELLIGENCE · PORTUGAL 2023–2024</div>", unsafe_allow_html=True)

    # KPI Row
    col1,col2,col3,col4,col5 = st.columns(5)
    col1.metric("Avg Carbon Intensity", f"{grid.carbon_intensity.mean():.0f} gCO₂/kWh", f"σ={grid.carbon_intensity.std():.0f}")
    col2.metric("Avg Renewable %",      f"{grid.renewable_pct.mean():.1f}%",             f"max={grid.renewable_pct.max():.0f}%")
    col3.metric("Total Grid Records",   f"{len(grid):,}",                                "17,520 hours")
    col4.metric("Training Runs",        f"{len(runs)}",                                  f"avg CO₂={runs.co2_kg.mean():.1f}kg")
    col5.metric("CI Classifier Acc",    f"{stats['ci']['acc']:.1f}%",                    "RandomForest")

    st.markdown("---")
    col_l, col_r = st.columns([2,1])

    with col_l:
        # Monthly CI trend
        monthly = grid.groupby("month").agg(
            ci=("carbon_intensity","mean"), rp=("renewable_pct","mean")
        ).reset_index()
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
                       7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
        monthly["month_name"] = monthly["month"].map(month_names)

        fig = make_subplots(specs=[[{"secondary_y":True}]])
        fig.add_trace(go.Scatter(x=monthly.month_name, y=monthly.ci, name="Carbon Intensity",
            fill="tozeroy", line=dict(color="#ff4444",width=2),
            fillcolor="rgba(255,68,68,0.1)"), secondary_y=False)
        fig.add_trace(go.Scatter(x=monthly.month_name, y=monthly.rp, name="Renewable %",
            fill="tozeroy", line=dict(color="#00ff87",width=2),
            fillcolor="rgba(0,255,135,0.08)"), secondary_y=True)
        fig.update_layout(**PLOTLY_LAYOUT, title="Monthly Carbon Intensity vs Renewable %",
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        fig.update_yaxes(title_text="gCO₂/kWh", secondary_y=False, gridcolor="#1a3a50")
        fig.update_yaxes(title_text="Renewable %", secondary_y=True, gridcolor="#1a3a50")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        # CI class distribution pie
        ci_dist = grid_eng["ci_class"].value_counts().reset_index()
        ci_dist.columns = ["class","count"]
        ci_dist["color"] = ci_dist["class"].map(CI_COLORS)
        fig2 = go.Figure(go.Pie(
            labels=ci_dist["class"], values=ci_dist["count"],
            marker_colors=ci_dist["color"], hole=0.5,
            textfont=dict(family="Rajdhani"), showlegend=True
        ))
        fig2.update_layout(**PLOTLY_LAYOUT, title="CI Class Distribution",
            legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(size=10)))
        st.plotly_chart(fig2, use_container_width=True)

    # 24h hourly profile
    hourly = grid.groupby("hour").agg(
        ci=("carbon_intensity","mean"), rp=("renewable_pct","mean"),
        solar=("solar_mw","mean"), wind=("wind_mw","mean")
    ).reset_index()

    fig3 = make_subplots(rows=1,cols=2,subplot_titles=["Avg CI by Hour","Avg Renewable by Hour"])
    fig3.add_trace(go.Bar(x=hourly.hour, y=hourly.ci, name="CI",
        marker_color=[("#00ff87" if v<70 else "#ffd700" if v<100 else "#ff4444") for v in hourly.ci]), row=1,col=1)
    fig3.add_trace(go.Bar(x=hourly.hour, y=hourly.rp, name="Renewable %",
        marker_color="rgba(0,255,135,0.6)"), row=1,col=2)
    fig3.update_layout(**PLOTLY_LAYOUT, showlegend=False, title="24-Hour Grid Profile (Portugal Average)")
    for ax in ["xaxis","xaxis2"]:
        fig3.update_layout(**{ax:dict(gridcolor="#1a3a50",tickfont=dict(size=9))})
    st.plotly_chart(fig3, use_container_width=True)

    # Model accuracy summary
    st.markdown("#### ML Models Trained on Real Data")
    mc1,mc2,mc3,mc4 = st.columns(4)
    mc1.metric("CI Classifier",    f"{stats['ci']['acc']:.2f}%",  "Accuracy")
    mc2.metric("RP Classifier",    f"{stats['rp']['acc']:.2f}%",  "Accuracy")
    mc3.metric("GS Classifier",    f"{stats['gs']['acc']:.2f}%",  "Accuracy")
    mc4.metric("CO₂ Regressor",    f"{stats['co2']['mae']:.2f}kg","MAE")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CARBON INTENSITY
# ═══════════════════════════════════════════════════════════════════════════════
elif "Carbon Intensity" in page:
    st.markdown("## 🌡 Carbon Intensity Predictor")
    st.markdown("<div class='section-header'>ENTER GRID CONDITIONS → LIVE ML PREDICTION</div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("#### Grid Conditions")
        ci_now  = st.number_input("Carbon Intensity Now (gCO₂/kWh)", 0.0, 900.0, 160.0, 1.0)
        ci_3h   = st.number_input("Carbon Intensity 3h Ago",          0.0, 900.0, 150.0, 1.0)
        ci_6h   = st.number_input("Carbon Intensity 6h Ago",          0.0, 900.0, 145.0, 1.0)
        renew   = st.slider("Renewable %", 0.0, 100.0, 55.0, 0.5)
        temp    = st.slider("Temperature (°C)", -10.0, 50.0, 18.0, 0.5)
        wind    = st.slider("Wind Speed (m/s)", 0.0, 30.0, 4.0, 0.1)
        hour    = st.slider("Hour of Day", 0, 23, 12)
        month   = st.slider("Month", 1, 12, 6)
        dow     = st.slider("Day of Week (0=Mon)", 0, 6, 2)
        gb      = st.number_input("Grid Balance (MW)", -2000.0, 2000.0, -50.0, 10.0)

    ci_trend = ci_now - ci_3h
    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    is_peak  = 1 if (8 <= hour <= 20) else 0

    feat = np.array([[ci_now, ci_3h, ci_6h, ci_trend, renew, temp, wind, hour, month, dow, gb, is_night, is_peak]])
    pred_idx   = models["ci"].predict(feat)[0]
    pred_class = encoders["ci"].inverse_transform([pred_idx])[0]
    probs      = models["ci"].predict_proba(feat)[0]
    confidence = probs[pred_idx] * 100
    prob_df    = pd.DataFrame({"Class": encoders["ci"].classes_, "Probability": probs * 100})

    with col_out:
        # Big prediction result
        color = CI_COLORS.get(pred_class, "#c8dde8")
        rec = ("✅ Great time to run ML training! Grid is very clean." if pred_class in ("Very Low","Low")
               else "⚠️ Acceptable. Consider off-peak scheduling for intensive jobs." if pred_class == "Moderate"
               else "❌ High carbon period. Delay training to reduce CO₂.")
        rec_cls = "rec-good" if pred_class in ("Very Low","Low") else "rec-warn" if pred_class=="Moderate" else "rec-bad"

        st.markdown(f"""
        <div class='pred-box' style='border-top-color:{color}'>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#4a6a7a;letter-spacing:2px;margin-bottom:8px'>ML PREDICTION</div>
          <div class='pred-class' style='color:{color}'>{pred_class}</div>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.8rem;color:#4a6a7a;margin-top:6px'>Confidence: {confidence:.1f}%</div>
          <div style='font-size:0.7rem;color:#4a6a7a;margin-top:4px'>Based on 17,520 real grid records · RandomForest · Acc={stats['ci']['acc']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div class='{rec_cls}' style='margin-top:12px'>{rec}</div>", unsafe_allow_html=True)

        # Trend indicator
        trend_col, ci_col = st.columns(2)
        trend_col.metric("CI Trend (vs 3h ago)", f"{ci_trend:+.1f}", "↑ Rising" if ci_trend>5 else "↓ Falling" if ci_trend<-5 else "→ Stable")
        ci_col.metric("Is Night / Peak", f"{'Night' if is_night else 'Peak' if is_peak else 'Off-peak'}", f"Hour {hour}:00")

        # Probability bar chart
        prob_df["Color"] = prob_df["Class"].map(CI_COLORS)
        fig_p = go.Figure(go.Bar(
            x=prob_df["Probability"], y=prob_df["Class"], orientation="h",
            marker_color=prob_df["Color"], text=prob_df["Probability"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside"
        ))
        fig_p.update_layout(**PLOTLY_LAYOUT, title="Class Probabilities", height=260)
        fig_p.update_xaxes(range=[0,105], gridcolor="#1a3a50")
        fig_p.update_yaxes(gridcolor="#1a3a50")
        st.plotly_chart(fig_p, use_container_width=True)

    # 24h chart with current hour highlighted
    hourly = grid.groupby("hour")["carbon_intensity"].mean().reset_index()
    hourly.columns = ["hour","avg_ci"]
    hourly["is_selected"] = hourly["hour"] == hour
    fig_h = go.Figure()
    fig_h.add_trace(go.Bar(x=hourly.hour, y=hourly.avg_ci,
        marker_color=["#00c8ff" if h==hour else ("#00ff87" if v<70 else "#ffd700" if v<100 else "#ff4444")
                      for h,v in zip(hourly.hour, hourly.avg_ci)],
        name="Avg CI"))
    fig_h.add_vline(x=hour, line_color="#ffffff", line_dash="dash", line_width=1.5,
        annotation_text=f" Hour {hour}:00 →", annotation_font_color="#00c8ff")
    fig_h.update_layout(**PLOTLY_LAYOUT, title=f"24-Hour Carbon Intensity Profile · Hour {hour}:00 Highlighted", height=220)
    st.plotly_chart(fig_h, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: RENEWABLE % PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "Renewable" in page:
    st.markdown("## 🌿 Renewable % Predictor")
    st.markdown("<div class='section-header'>WEATHER + GRID → RENEWABLE ENERGY LEVEL PREDICTION</div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("#### Conditions")
        rp_1h   = st.number_input("Renewable % 1h Ago", 0.0, 100.0, 53.0, 0.5)
        rp_3h   = st.number_input("Renewable % 3h Ago", 0.0, 100.0, 50.0, 0.5)
        ci_in   = st.number_input("Carbon Intensity (gCO₂/kWh)", 0.0, 900.0, 160.0, 1.0)
        wind    = st.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.1)
        cloud   = st.slider("Cloud Cover (%)", 0.0, 100.0, 40.0, 1.0)
        solar   = st.number_input("Solar MW", 0.0, 5000.0, 200.0, 10.0)
        wind_mw = st.number_input("Wind MW", 0.0, 10000.0, 1200.0, 50.0)
        hour    = st.slider("Hour (0-23)", 0, 23, 12)
        month   = st.slider("Month (1-12)", 1, 12, 6)

    is_night = 1 if (hour >= 22 or hour <= 5) else 0
    feat = np.array([[rp_1h, rp_3h, ci_in, wind, cloud, solar, wind_mw, hour, month, is_night]])
    pred_idx   = models["rp"].predict(feat)[0]
    pred_class = encoders["rp"].inverse_transform([pred_idx])[0]
    probs      = models["rp"].predict_proba(feat)[0]
    confidence = probs[pred_idx] * 100

    with col_out:
        color = RP_COLORS.get(pred_class, "#c8dde8")
        rec = ("🌞 Excellent renewable conditions. Perfect for green compute." if pred_class in ("Exceptional","Very High")
               else "✅ Good mix. Acceptable for training runs." if pred_class == "High"
               else "⚠️ Mixed grid. Check CI before scheduling." if pred_class == "Moderate"
               else "❌ Low renewables. Grid dominated by fossil fuels.")
        rec_cls = "rec-good" if pred_class in ("Exceptional","Very High","High") else "rec-warn" if pred_class=="Moderate" else "rec-bad"

        st.markdown(f"""
        <div class='pred-box' style='border-top-color:{color}'>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#4a6a7a;letter-spacing:2px;margin-bottom:8px'>RENEWABLE PREDICTION</div>
          <div class='pred-class' style='color:{color}'>{pred_class}</div>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.8rem;color:#4a6a7a;margin-top:6px'>Confidence: {confidence:.1f}% · Acc={stats['rp']['acc']:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(f"<div class='{rec_cls}' style='margin-top:12px'>{rec}</div>", unsafe_allow_html=True)

        # Prob bars
        order = ["Very Low","Low","Moderate","High","Very High","Exceptional"]
        cls_list = encoders["rp"].classes_
        prob_vals = []
        for c in order:
            if c in cls_list:
                idx = list(cls_list).index(c)
                prob_vals.append({"Class":c, "Probability": probs[idx]*100, "Color": RP_COLORS[c]})
        pdf = pd.DataFrame(prob_vals)
        fig_p = go.Figure(go.Bar(
            x=pdf["Probability"], y=pdf["Class"], orientation="h",
            marker_color=pdf["Color"], text=pdf["Probability"].apply(lambda x:f"{x:.1f}%"),
            textposition="outside"
        ))
        fig_p.update_layout(**PLOTLY_LAYOUT, title="Class Probabilities", height=280)
        fig_p.update_xaxes(range=[0,105])
        fig_p.update_yaxes(categoryorder="array", categoryarray=order)
        st.plotly_chart(fig_p, use_container_width=True)

    # Charts
    c1,c2 = st.columns(2)
    with c1:
        hourly_rp = grid.groupby("hour")["renewable_pct"].mean().reset_index()
        fig_rp = go.Figure(go.Bar(x=hourly_rp.hour, y=hourly_rp.renewable_pct,
            marker_color=["#00c8ff" if h==hour else "rgba(0,255,135,0.6)" for h in hourly_rp.hour]))
        fig_rp.update_layout(**PLOTLY_LAYOUT, title=f"Renewable % by Hour · {hour}:00 Highlighted", height=240)
        st.plotly_chart(fig_rp, use_container_width=True)

    with c2:
        hourly_gen = grid.groupby("hour").agg(solar=("solar_mw","mean"), wind=("wind_mw","mean")).reset_index()
        fig_gen = go.Figure()
        fig_gen.add_trace(go.Scatter(x=hourly_gen.hour, y=hourly_gen.solar, name="Solar MW",
            fill="tozeroy", line=dict(color="#ffd700",width=2), fillcolor="rgba(255,215,0,0.1)"))
        fig_gen.add_trace(go.Scatter(x=hourly_gen.hour, y=hourly_gen.wind, name="Wind MW",
            fill="tozeroy", line=dict(color="#00c8ff",width=2), fillcolor="rgba(0,200,255,0.08)"))
        fig_gen.update_layout(**PLOTLY_LAYOUT, title="Solar & Wind Generation by Hour", height=240,
            legend=dict(bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_gen, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: GREEN SCORE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Green Score" in page:
    st.markdown("## 🏅 Green Score Predictor")
    st.markdown("<div class='section-header'>GRID SNAPSHOT → SUSTAINABILITY GRADE</div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("#### Grid Conditions")
        ci_in   = st.number_input("Carbon Intensity (gCO₂/kWh)", 0.0, 900.0, 160.0, 1.0)
        rp_in   = st.slider("Renewable %", 0.0, 100.0, 55.0, 0.5)
        wind    = st.slider("Wind Speed (m/s)", 0.0, 30.0, 4.0, 0.1)
        temp    = st.slider("Temperature (°C)", -10.0, 50.0, 18.0, 0.5)
        hour    = st.slider("Hour (0-23)", 0, 23, 12)
        month   = st.slider("Month (1-12)", 1, 12, 6)
        gb      = st.number_input("Grid Balance (MW)", -2000.0, 2000.0, -50.0, 10.0)

    is_night = 1 if (hour>=22 or hour<=5) else 0
    is_peak  = 1 if (8<=hour<=20) else 0
    feat = np.array([[ci_in, rp_in, wind, temp, hour, month, is_night, is_peak, gb]])
    pred_idx   = models["gs"].predict(feat)[0]
    pred_class = encoders["gs"].inverse_transform([pred_idx])[0]
    probs      = models["gs"].predict_proba(feat)[0]
    confidence = probs[pred_idx] * 100
    grade_map  = {"Excellent":"A+","Very Good":"A","Good":"B","Fair":"C","Poor":"D"}
    grade      = grade_map.get(pred_class, "?")
    score_est  = max(0, min(100, 100-(ci_in/9)+(rp_in*0.5)+wind*1.5))

    with col_out:
        color = GS_COLORS.get(pred_class, "#c8dde8")
        gc1, gc2, gc3 = st.columns(3)
        gc1.metric("Grade", grade)
        gc2.metric("Class", pred_class)
        gc3.metric("Score Est.", f"{score_est:.0f}/100")

        # Gauge chart
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_est,
            delta={"reference":50,"valueformat":".0f"},
            title={"text":f"Green Score  ({pred_class})", "font":{"family":"Orbitron","color":"#c8dde8","size":13}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#4a6a7a","tickfont":{"color":"#4a6a7a","size":9}},
                "bar":{"color":color,"thickness":0.3},
                "bgcolor":"#0d1e2e",
                "bordercolor":"#1a3a50",
                "steps":[
                    {"range":[0,35],"color":"rgba(255,68,68,0.15)"},
                    {"range":[35,55],"color":"rgba(255,140,0,0.12)"},
                    {"range":[55,75],"color":"rgba(255,215,0,0.12)"},
                    {"range":[75,90],"color":"rgba(127,255,0,0.12)"},
                    {"range":[90,100],"color":"rgba(0,255,135,0.15)"},
                ],
                "threshold":{"line":{"color":color,"width":3},"thickness":0.8,"value":score_est}
            },
            number={"font":{"family":"Orbitron","color":color,"size":36}}
        ))
        fig_g.update_layout(paper_bgcolor="#050a0e", plot_bgcolor="#050a0e",
            height=280, margin=dict(l=20,r=20,t=40,b=20),
            font=dict(color="#c8dde8"))
        st.plotly_chart(fig_g, use_container_width=True)

        # Confidence
        cls_list = encoders["gs"].classes_
        order = ["Poor","Fair","Good","Very Good","Excellent"]
        prob_data = []
        for c in order:
            if c in cls_list:
                idx = list(cls_list).index(c)
                prob_data.append({"Class":c, "Probability":probs[idx]*100, "Color":GS_COLORS[c]})
        pdf2 = pd.DataFrame(prob_data)
        fig_p2 = go.Figure(go.Bar(
            x=pdf2["Class"], y=pdf2["Probability"],
            marker_color=pdf2["Color"], text=pdf2["Probability"].apply(lambda x:f"{x:.1f}%"),
            textposition="outside"
        ))
        fig_p2.update_layout(**PLOTLY_LAYOUT, title=f"Grade Probabilities · Confidence={confidence:.1f}%", height=220)
        fig_p2.update_yaxes(range=[0,110])
        st.plotly_chart(fig_p2, use_container_width=True)

    # Monthly score chart
    monthly_scores = []
    monthly_grid_agg = grid.groupby("month").agg(ci=("carbon_intensity","mean"),rp=("renewable_pct","mean")).reset_index()
    month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    for _, row in monthly_grid_agg.iterrows():
        adj_ci = ci_in * (row.ci / grid.carbon_intensity.mean())
        adj_rp = min(100, rp_in * (row.rp / grid.renewable_pct.mean()))
        sc = max(0, min(100, 100-(adj_ci/9)+(adj_rp*0.5)+wind*1.5))
        monthly_scores.append({"Month": month_names[row.month], "Score": sc, "month_n": row.month})
    ms_df = pd.DataFrame(monthly_scores)
    fig_ms = go.Figure(go.Bar(
        x=ms_df["Month"], y=ms_df["Score"],
        marker_color=["#00c8ff" if m==month else ("#00ff87" if s>=75 else "#ffd700" if s>=55 else "#ff4444")
                      for m,s in zip(ms_df.month_n, ms_df.Score)],
        text=ms_df["Score"].apply(lambda x:f"{x:.0f}"),
        textposition="outside"
    ))
    fig_ms.update_layout(**PLOTLY_LAYOUT, title=f"Estimated Green Score Across Months (Your Conditions) · Month {month} Highlighted",
        height=240)
    fig_ms.update_yaxes(range=[0,110])
    st.plotly_chart(fig_ms, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: CO2 PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "CO₂" in page:
    st.markdown("## 💨 CO₂ Emission Predictor")
    st.markdown("<div class='section-header'>GPU JOB PARAMETERS → KG CO₂ + IMPACT + COUNTRY COMPARISON</div>", unsafe_allow_html=True)

    GPU_PRESETS = {
        "NVIDIA A100 (400W)":400,"NVIDIA H100 (700W)":700,"NVIDIA V100 (300W)":300,
        "NVIDIA RTX 4090 (450W)":450,"NVIDIA T4 (70W)":70,"AMD MI300X (750W)":750,"Custom":0
    }

    col_in, col_out = st.columns([1, 2])

    with col_in:
        st.markdown("#### Training Job")
        gpu_name = st.selectbox("GPU Model", list(GPU_PRESETS.keys()))
        gpu_tdp  = st.number_input("GPU TDP (W)", 1, 2000, GPU_PRESETS.get(gpu_name,400) or 400)
        gpu_cnt  = st.number_input("Number of GPUs", 1, 1024, 4)
        dur      = st.number_input("Duration (hours)", 0.1, 10000.0, 24.0, 0.5)
        pue      = st.slider("Data Centre PUE", 1.0, 3.0, 1.3, 0.05)
        ci_grid  = st.number_input("Grid Carbon Intensity (gCO₂/kWh)", 0.0, 900.0, 160.0, 1.0)
        st.caption("Norway≈26 · France≈55 · Portugal≈160 · Germany≈378 · Poland≈720")

    power_kw  = (gpu_cnt * gpu_tdp * pue) / 1000
    energy    = power_kw * dur
    feat_co2  = np.array([[gpu_cnt, gpu_tdp, dur, pue, ci_grid]])
    co2_pred  = max(0, models["co2"].predict(feat_co2)[0])
    co2_form  = energy * (ci_grid / 1000)
    trees     = co2_pred / 0.0297
    car_km    = co2_pred / 0.21
    phones    = co2_pred / 0.005
    eu_cost   = co2_pred * 0.025

    with col_out:
        co2_color = "#00ff87" if co2_pred < 5 else "#ffd700" if co2_pred < 50 else "#ff4444"
        st.markdown(f"""
        <div class='pred-box' style='border-top-color:{co2_color}'>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.7rem;color:#4a6a7a;letter-spacing:2px;margin-bottom:8px'>ML PREDICTION · GRADIENT BOOSTING · MAE={stats['co2']['mae']:.1f}kg</div>
          <div class='pred-class' style='color:{co2_color}'>{co2_pred:.3f} kg CO₂</div>
          <div style='font-family:"Share Tech Mono",monospace;font-size:0.75rem;color:#4a6a7a;margin-top:6px'>Formula: {co2_form:.3f} kg · Power: {power_kw:.2f} kW · Energy: {energy:.2f} kWh</div>
        </div>
        """, unsafe_allow_html=True)

        ic1,ic2,ic3,ic4 = st.columns(4)
        ic1.markdown(f"<div class='impact-card'><div class='impact-val'>{trees:.1f}</div><div class='impact-lbl'>🌳 Tree-days</div></div>", unsafe_allow_html=True)
        ic2.markdown(f"<div class='impact-card'><div class='impact-val'>{car_km:.1f}</div><div class='impact-lbl'>🚗 Car km</div></div>", unsafe_allow_html=True)
        ic3.markdown(f"<div class='impact-card'><div class='impact-val'>{phones:.0f}</div><div class='impact-lbl'>📱 Phone charges</div></div>", unsafe_allow_html=True)
        ic4.markdown(f"<div class='impact-card'><div class='impact-val'>€{eu_cost:.2f}</div><div class='impact-lbl'>💶 EU Carbon cost</div></div>", unsafe_allow_html=True)

    # Country comparison
    COUNTRY_CI = {"Norway":25.9,"France":54.8,"Portugal":159.4,"Spain":199.8,
                  "UK":211.1,"Italy":285.4,"Germany":378.1,"Poland":719.9}
    country_results = []
    for country, c_ci in COUNTRY_CI.items():
        f = np.array([[gpu_cnt, gpu_tdp, dur, pue, c_ci]])
        co2_c = max(0, models["co2"].predict(f)[0])
        country_results.append({"Country":country,"CO2_kg":co2_c,"CI":c_ci})
    cr_df = pd.DataFrame(country_results).sort_values("CO2_kg")

    c1,c2 = st.columns(2)
    with c1:
        fig_cc = go.Figure(go.Bar(
            x=cr_df["CO2_kg"], y=cr_df["Country"], orientation="h",
            marker_color=["#00ff87" if v<5 else "#ffd700" if v<50 else "#ff4444" for v in cr_df["CO2_kg"]],
            text=cr_df["CO2_kg"].apply(lambda x:f"{x:.2f} kg"), textposition="outside"
        ))
        fig_cc.update_layout(**PLOTLY_LAYOUT, title="CO₂ — Same Job Across Countries", height=280)
        fig_cc.update_xaxes(range=[0, cr_df["CO2_kg"].max()*1.25])
        st.plotly_chart(fig_cc, use_container_width=True)

    with c2:
        # CI sensitivity: how CO2 changes with CI
        ci_range = np.linspace(20, 750, 40)
        co2_range = [max(0, models["co2"].predict(np.array([[gpu_cnt,gpu_tdp,dur,pue,ci]]))[0]) for ci in ci_range]
        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(x=ci_range, y=co2_range, mode="lines",
            line=dict(color="#00ff87",width=2), fill="tozeroy",
            fillcolor="rgba(0,255,135,0.08)", name="CO₂ kg"))
        fig_sens.add_vline(x=ci_grid, line_color="#00c8ff", line_dash="dash",
            annotation_text=f" Your CI={ci_grid:.0f}", annotation_font_color="#00c8ff")
        fig_sens.update_layout(**PLOTLY_LAYOUT, title="CO₂ Sensitivity to Carbon Intensity", height=280,
            xaxis_title="Grid CI (gCO₂/kWh)", yaxis_title="kg CO₂")
        st.plotly_chart(fig_sens, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: SMART SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Scheduler" in page:
    st.markdown("## ⏰ Smart Job Scheduler")
    st.markdown("<div class='section-header'>FIND THE GREENEST HOUR TO RUN YOUR TRAINING JOB</div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 3])

    with col_in:
        st.markdown("#### Job Parameters")
        dur    = st.number_input("Job Duration (hours)", 0.5, 168.0, 8.0, 0.5)
        month  = st.slider("Month (1-12)", 1, 12, 6)
        month_names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}

    # Compute hourly CI for that month
    subset = grid[grid["month"] == month]
    hourly_ci = subset.groupby("hour")["carbon_intensity"].mean()
    hourly_rp = subset.groupby("hour")["renewable_pct"].mean()

    best_h  = hourly_ci.idxmin()
    worst_h = hourly_ci.idxmax()
    savings = ((hourly_ci[worst_h] - hourly_ci[best_h]) / hourly_ci[worst_h] * 100)

    with col_out:
        sc1,sc2,sc3 = st.columns(3)
        sc1.metric("✅ Best Start Time",  f"{best_h:02d}:00", f"CI={hourly_ci[best_h]:.0f} gCO₂/kWh")
        sc2.metric("❌ Worst Start Time", f"{worst_h:02d}:00", f"CI={hourly_ci[worst_h]:.0f} gCO₂/kWh")
        sc3.metric("💚 CO₂ Savings",      f"{savings:.1f}%",   "Best vs Worst")

        top3 = hourly_ci.nsmallest(3)
        top3_str = "  ·  ".join([f"{h:02d}:00 (CI={v:.0f})" for h,v in top3.items()])
        st.markdown(f"<div class='rec-good' style='margin-bottom:12px'>🏆 Top 3 Green Windows: {top3_str}</div>", unsafe_allow_html=True)

        # 24h bar chart
        sched_df = pd.DataFrame({"hour":hourly_ci.index, "ci":hourly_ci.values, "rp":hourly_rp.values})
        fig_sc = go.Figure(go.Bar(
            x=sched_df.hour, y=sched_df.ci,
            marker_color=["#00ff87" if h==best_h else "#ff4444" if h==worst_h
                          else ("#00ff87" if v<70 else "#ffd700" if v<100 else "#ff8c00")
                          for h,v in zip(sched_df.hour, sched_df.ci)],
            text=sched_df.ci.apply(lambda x:f"{x:.0f}"), textposition="outside",
            customdata=sched_df.rp,
            hovertemplate="Hour %{x}:00<br>CI: %{y:.1f} gCO₂/kWh<br>Renewable: %{customdata:.1f}%<extra></extra>"
        ))
        fig_sc.add_annotation(x=best_h, y=hourly_ci[best_h], text="✅ BEST",
            showarrow=True, arrowhead=2, arrowcolor="#00ff87", font=dict(color="#00ff87"))
        fig_sc.add_annotation(x=worst_h, y=hourly_ci[worst_h], text="❌ WORST",
            showarrow=True, arrowhead=2, arrowcolor="#ff4444", font=dict(color="#ff4444"))
        fig_sc.update_layout(**PLOTLY_LAYOUT, title=f"Carbon Intensity by Hour · {month_names[month]}",
            height=300, xaxis_title="Hour of Day", yaxis_title="Avg CI (gCO₂/kWh)")
        st.plotly_chart(fig_sc, use_container_width=True)

    # Renewable % chart
    fig_rp_sc = go.Figure(go.Scatter(x=sched_df.hour, y=sched_df.rp,
        fill="tozeroy", line=dict(color="#00ff87",width=2),
        fillcolor="rgba(0,255,135,0.08)"))
    fig_rp_sc.update_layout(**PLOTLY_LAYOUT, title=f"Renewable % by Hour · {month_names[month]}",
        height=200, xaxis_title="Hour", yaxis_title="Renewable %")
    st.plotly_chart(fig_rp_sc, use_container_width=True)

    # Monthly best hour table
    st.markdown("#### Best Start Hour by Month")
    best_by_month = []
    for m in range(1,13):
        sub = grid[grid["month"]==m].groupby("hour")["carbon_intensity"].mean()
        if len(sub):
            best_by_month.append({
                "Month": month_names[m],
                "Best Hour": f"{sub.idxmin():02d}:00",
                "Min CI (gCO₂/kWh)": f"{sub.min():.1f}",
                "Max Renewable %": f"{grid[grid['month']==m].groupby('hour')['renewable_pct'].mean().max():.1f}%"
            })
    st.dataframe(pd.DataFrame(best_by_month), use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: ANOMALY DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "Anomaly" in page:
    st.markdown("## 🔍 Grid Anomaly Detector")
    st.markdown("<div class='section-header'>SCAN REAL GRID DATA FOR SPIKES, DROPS & INSTABILITIES</div>", unsafe_allow_html=True)

    col_in, col_out = st.columns([1, 3])

    with col_in:
        st.markdown("#### Detection Settings")
        window  = st.slider("Scan Window (hours)", 24, 8760, 168)
        thresh  = st.slider("Sigma Threshold", 1.5, 4.0, 2.0, 0.1)
        metric  = st.selectbox("Metric to Scan", ["carbon_intensity","renewable_pct","grid_balance_mw"])

    recent = grid.tail(window).copy().reset_index(drop=True)
    mu  = recent[metric].mean()
    sig = recent[metric].std()
    recent["z_score"] = (recent[metric] - mu) / sig
    recent["anomaly"] = recent["z_score"].abs() > thresh
    anomalies = recent[recent["anomaly"]].copy()
    anomalies["type"] = anomalies.apply(
        lambda r: "Spike" if r.z_score > 0 else "Drop", axis=1)

    with col_out:
        ac1,ac2,ac3 = st.columns(3)
        ac1.metric("Records Scanned",  f"{len(recent):,}")
        ac2.metric("Anomalies Found",  str(len(anomalies)),
            delta=f"{'⚠️ High' if len(anomalies)>20 else '✅ Normal'}")
        ac3.metric(f"Mean / σ",        f"{mu:.1f} / {sig:.1f}")

    # Time series with anomalies highlighted
    fig_an = go.Figure()
    fig_an.add_trace(go.Scatter(
        x=recent.index, y=recent[metric],
        mode="lines", name=metric,
        line=dict(color="#00ff87",width=1.5),
        fill="tozeroy", fillcolor="rgba(0,255,135,0.05)"
    ))
    if len(anomalies):
        fig_an.add_trace(go.Scatter(
            x=anomalies.index, y=anomalies[metric],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4444", size=8, symbol="x-thin-open", line=dict(width=2))
        ))
    # Threshold lines
    fig_an.add_hline(y=mu + thresh*sig, line_color="#ff4444", line_dash="dash",
        annotation_text=f"+{thresh}σ", annotation_font_color="#ff4444")
    fig_an.add_hline(y=mu - thresh*sig, line_color="#ff8c00", line_dash="dash",
        annotation_text=f"-{thresh}σ", annotation_font_color="#ff8c00")
    fig_an.add_hline(y=mu, line_color="#4a6a7a", line_dash="dot")
    fig_an.update_layout(**PLOTLY_LAYOUT,
        title=f"{metric} — Last {window}h · {len(anomalies)} Anomalies Detected (>{thresh}σ)",
        height=320, xaxis_title="Record Index", yaxis_title=metric)
    st.plotly_chart(fig_an, use_container_width=True)

    if len(anomalies):
        st.markdown("#### Anomaly Log")
        anom_display = anomalies[["timestamp","hour","month",metric,"z_score","type"]].head(20).copy()
        anom_display["z_score"] = anom_display["z_score"].round(2)
        anom_display[metric]    = anom_display[metric].round(2)
        st.dataframe(anom_display, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='rec-good'>✅ No anomalies detected in this window. Grid was stable.</div>", unsafe_allow_html=True)

    # Distribution
    c1,c2 = st.columns(2)
    with c1:
        fig_dist = go.Figure(go.Histogram(x=recent[metric], nbinsx=50,
            marker_color="rgba(0,255,135,0.5)", marker_line_color="#00ff87", marker_line_width=0.5))
        fig_dist.add_vline(x=mu+thresh*sig, line_color="#ff4444", line_dash="dash")
        fig_dist.add_vline(x=mu-thresh*sig, line_color="#ff8c00", line_dash="dash")
        fig_dist.update_layout(**PLOTLY_LAYOUT, title=f"Distribution of {metric}", height=240)
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        # Anomaly rate by hour
        grid_copy = recent.copy()
        hourly_anom = grid_copy.groupby("hour")["anomaly"].mean().reset_index()
        hourly_anom["rate_pct"] = hourly_anom["anomaly"] * 100
        fig_ha = go.Figure(go.Bar(x=hourly_anom.hour, y=hourly_anom.rate_pct,
            marker_color=["#ff4444" if v>5 else "#ffd700" if v>2 else "#00ff87" for v in hourly_anom.rate_pct]))
        fig_ha.update_layout(**PLOTLY_LAYOUT, title="Anomaly Rate by Hour (%)", height=240)
        st.plotly_chart(fig_ha, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: COUNTRY COMPARATOR
# ═══════════════════════════════════════════════════════════════════════════════
elif "Country" in page:
    st.markdown("## 🌍 Country CO₂ Comparator")
    st.markdown("<div class='section-header'>SAME JOB · 8 COUNTRIES · CO₂ SIDE BY SIDE</div>", unsafe_allow_html=True)

    GPU_PRESETS = {"NVIDIA A100 (400W)":400,"NVIDIA H100 (700W)":700,
                   "NVIDIA V100 (300W)":300,"NVIDIA RTX 4090 (450W)":450,
                   "NVIDIA T4 (70W)":70,"AMD MI300X (750W)":750,"Custom":0}

    col_in, col_out = st.columns([1, 2])
    with col_in:
        st.markdown("#### Job Definition")
        gpu_name = st.selectbox("GPU", list(GPU_PRESETS.keys()))
        gpu_tdp  = st.number_input("GPU TDP (W)", 1, 2000, GPU_PRESETS.get(gpu_name,400) or 400)
        gpu_cnt  = st.number_input("GPU Count", 1, 1024, 4)
        dur      = st.number_input("Duration (hours)", 0.1, 10000.0, 24.0, 1.0)
        pue      = st.slider("PUE", 1.0, 3.0, 1.3, 0.05)

    power_kw = (gpu_cnt * gpu_tdp * pue) / 1000

    countries_cc = cc.groupby("country").agg(
        ci=("carbon_intensity","mean"), rp=("renewable_pct","mean")
    ).reset_index()

    results = []
    for _, row in countries_cc.iterrows():
        feat = np.array([[gpu_cnt, gpu_tdp, dur, pue, row.ci]])
        co2 = max(0, models["co2"].predict(feat)[0])
        results.append({"Country":row["country"],"CO2_kg":co2,"CI":row.ci,"Renewable%":row.rp})
    res_df = pd.DataFrame(results).sort_values("CO2_kg")

    best_c  = res_df.iloc[0]
    worst_c = res_df.iloc[-1]
    savings = ((worst_c.CO2_kg - best_c.CO2_kg) / worst_c.CO2_kg * 100)

    with col_out:
        cc1,cc2,cc3 = st.columns(3)
        cc1.metric(f"🥇 Best: {best_c.Country}",  f"{best_c.CO2_kg:.2f} kg CO₂", f"CI={best_c.CI:.0f}")
        cc2.metric(f"🔴 Worst: {worst_c.Country}", f"{worst_c.CO2_kg:.2f} kg CO₂", f"CI={worst_c.CI:.0f}")
        cc3.metric("💚 Relocation Savings",        f"{savings:.0f}% CO₂",          f"{worst_c.CO2_kg-best_c.CO2_kg:.2f} kg saved")

    c1,c2 = st.columns(2)
    with c1:
        fig_c1 = go.Figure(go.Bar(
            x=res_df.CO2_kg, y=res_df.Country, orientation="h",
            marker_color=["#00ff87" if v<5 else "#ffd700" if v<50 else "#ff4444" for v in res_df.CO2_kg],
            text=res_df.CO2_kg.apply(lambda x:f"{x:.2f} kg"), textposition="outside"
        ))
        fig_c1.update_layout(**PLOTLY_LAYOUT, title="CO₂ by Country (ML Prediction)", height=320)
        fig_c1.update_xaxes(range=[0,res_df.CO2_kg.max()*1.3])
        st.plotly_chart(fig_c1, use_container_width=True)

    with c2:
        fig_c2 = go.Figure(go.Bar(
            x=res_df["Renewable%"], y=res_df.Country, orientation="h",
            marker_color="rgba(0,255,135,0.6)",
            text=res_df["Renewable%"].apply(lambda x:f"{x:.1f}%"), textposition="outside"
        ))
        fig_c2.update_layout(**PLOTLY_LAYOUT, title="Renewable % by Country", height=320)
        fig_c2.update_xaxes(range=[0,110])
        st.plotly_chart(fig_c2, use_container_width=True)

    # Full table
    res_df["Power (kW)"]   = power_kw
    res_df["Energy (kWh)"] = power_kw * dur
    res_df["CO2_kg"]       = res_df["CO2_kg"].round(3)
    res_df["CI"]           = res_df["CI"].round(1)
    res_df["Renewable%"]   = res_df["Renewable%"].round(1)
    st.markdown("#### Full Comparison Table")
    st.dataframe(res_df[["Country","CI","Renewable%","CO2_kg","Power (kW)","Energy (kWh)"]].reset_index(drop=True),
        use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Model" in page:
    st.markdown("## 📊 ML Model Performance")
    st.markdown("<div class='section-header'>ACCURACY · FEATURES · REAL DATA TRAINING DETAILS</div>", unsafe_allow_html=True)

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("CI Classifier",  f"{stats['ci']['acc']:.2f}%",  "RandomForest · 150 trees")
    m2.metric("RP Classifier",  f"{stats['rp']['acc']:.2f}%",  "RandomForest · 150 trees")
    m3.metric("GS Classifier",  f"{stats['gs']['acc']:.2f}%",  "RandomForest · 150 trees")
    m4.metric("CO₂ Regressor",  f"MAE={stats['co2']['mae']:.2f}kg", f"R²={stats['co2']['r2']:.3f}")

    c1,c2 = st.columns(2)
    with c1:
        acc_data = pd.DataFrame([
            {"Model":"CI Classifier","Accuracy":stats['ci']['acc']},
            {"Model":"RP Classifier","Accuracy":stats['rp']['acc']},
            {"Model":"GS Classifier","Accuracy":stats['gs']['acc']},
            {"Model":"CO₂ R²×100",  "Accuracy":stats['co2']['r2']*100},
        ])
        fig_acc = go.Figure(go.Bar(
            x=acc_data.Model, y=acc_data.Accuracy,
            marker_color=["#00ff87" if v>=95 else "#00c8ff" if v>=80 else "#ffd700" for v in acc_data.Accuracy],
            text=acc_data.Accuracy.apply(lambda x:f"{x:.2f}%"), textposition="outside"
        ))
        fig_acc.update_layout(**PLOTLY_LAYOUT, title="Model Accuracy / R² Scores", height=280)
        fig_acc.update_yaxes(range=[0,105])
        st.plotly_chart(fig_acc, use_container_width=True)

    with c2:
        ds_data = pd.DataFrame([
            {"Dataset":"Portugal Grid","Rows":17520,"Type":"Hourly Grid"},
            {"Dataset":"Training Runs","Rows":300,"Type":"GPU Runs"},
            {"Dataset":"Country Compare","Rows":5760,"Type":"Multi-country"},
        ])
        fig_ds = go.Figure(go.Bar(
            x=ds_data.Dataset, y=ds_data.Rows,
            marker_color=["#00ff87","#00c8ff","#ffd700"],
            text=ds_data.Rows.apply(lambda x:f"{x:,}"), textposition="outside"
        ))
        fig_ds.update_layout(**PLOTLY_LAYOUT, title="Dataset Sizes", height=280)
        fig_ds.update_yaxes(range=[0,20000])
        st.plotly_chart(fig_ds, use_container_width=True)

    for key, name in [("ci","Carbon Intensity Classifier"),("rp","Renewable % Classifier"),
                       ("gs","Green Score Classifier"),("co2","CO₂ Regressor")]:
        st.markdown(f"**{name}**")
        feats = stats[key]["features"]
        st.markdown(f"`{'` · `'.join(feats)}`")
        if key != "co2":
            st.markdown(f"Accuracy: `{stats[key]['acc']:.2f}%` · Classes: `{', '.join(stats[key]['classes'])}`")
        else:
            st.markdown(f"MAE: `{stats[key]['mae']:.3f} kg` · R²: `{stats[key]['r2']:.4f}`")
        st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE: DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif "Dataset" in page:
    st.markdown("## 📈 Dataset Explorer")
    st.markdown("<div class='section-header'>BROWSE 17,520 REAL PORTUGAL GRID RECORDS</div>", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["🗺 Grid Data","🤖 Training Runs","🌍 Country Comparison","💰 Carbon Budget"])

    with tab1:
        d1,d2,d3,d4 = st.columns(4)
        d1.metric("Records",      f"{len(grid):,}")
        d2.metric("Avg CI",       f"{grid.carbon_intensity.mean():.1f} gCO₂/kWh")
        d3.metric("Avg Renewable",f"{grid.renewable_pct.mean():.1f}%")
        d4.metric("Period",       "2023–2024")

        c1,c2 = st.columns(2)
        with c1:
            fig_ci_ts = px.line(grid.tail(720), x="timestamp", y="carbon_intensity",
                title="Carbon Intensity — Last 720 Hours")
            fig_ci_ts.update_traces(line_color="#ff4444", line_width=1)
            fig_ci_ts.update_layout(**PLOTLY_LAYOUT, height=240)
            st.plotly_chart(fig_ci_ts, use_container_width=True)
        with c2:
            fig_rp_ts = px.line(grid.tail(720), x="timestamp", y="renewable_pct",
                title="Renewable % — Last 720 Hours")
            fig_rp_ts.update_traces(line_color="#00ff87", line_width=1)
            fig_rp_ts.update_layout(**PLOTLY_LAYOUT, height=240)
            st.plotly_chart(fig_rp_ts, use_container_width=True)

        st.markdown("#### Sample Data")
        st.dataframe(grid[["timestamp","carbon_intensity","renewable_pct","solar_mw",
                            "wind_mw","temperature_c","wind_speed_ms","green_score"]].tail(20),
                     use_container_width=True, hide_index=True)

    with tab2:
        r1,r2,r3,r4 = st.columns(4)
        r1.metric("Total Runs",  f"{len(runs)}")
        r2.metric("Total CO₂",  f"{runs.co2_kg.sum():.1f} kg")
        r3.metric("Avg CO₂",    f"{runs.co2_kg.mean():.2f} kg")
        r4.metric("Avg Duration",f"{runs.duration_h.mean():.1f}h")

        c1,c2 = st.columns(2)
        with c1:
            team_co2 = runs.groupby("team")["co2_kg"].sum().reset_index().sort_values("co2_kg",ascending=False)
            fig_t = go.Figure(go.Bar(x=team_co2.team, y=team_co2.co2_kg,
                marker_color=["#ff4444" if v>200 else "#ffd700" if v>50 else "#00ff87" for v in team_co2.co2_kg],
                text=team_co2.co2_kg.apply(lambda x:f"{x:.1f}"), textposition="outside"))
            fig_t.update_layout(**PLOTLY_LAYOUT, title="CO₂ by Team", height=260)
            st.plotly_chart(fig_t, use_container_width=True)
        with c2:
            gpu_cnt_df = runs["gpu_type"].value_counts().reset_index()
            gpu_cnt_df.columns = ["GPU","Count"]
            fig_gpu = go.Figure(go.Bar(x=gpu_cnt_df.GPU, y=gpu_cnt_df.Count,
                marker_color="rgba(0,200,255,0.7)"))
            fig_gpu.update_layout(**PLOTLY_LAYOUT, title="Most Used GPUs", height=260,
                xaxis_tickangle=-30)
            st.plotly_chart(fig_gpu, use_container_width=True)

        st.dataframe(runs[["run_id","team","model_type","gpu_type","gpu_count",
                            "duration_h","pue","carbon_intensity","co2_kg","green_score"]].head(20),
                     use_container_width=True, hide_index=True)

    with tab3:
        avg_cc = cc.groupby("country").agg(ci=("carbon_intensity","mean"),rp=("renewable_pct","mean")).reset_index()
        c1,c2 = st.columns(2)
        with c1:
            fig_ci_cc = go.Figure(go.Bar(x=avg_cc.country, y=avg_cc.ci,
                marker_color=["#00ff87" if v<100 else "#ffd700" if v<300 else "#ff4444" for v in avg_cc.ci],
                text=avg_cc.ci.apply(lambda x:f"{x:.0f}"), textposition="outside"))
            fig_ci_cc.update_layout(**PLOTLY_LAYOUT, title="Avg Carbon Intensity by Country", height=280)
            st.plotly_chart(fig_ci_cc, use_container_width=True)
        with c2:
            fig_rp_cc = go.Figure(go.Bar(x=avg_cc.country, y=avg_cc.rp,
                marker_color="rgba(0,255,135,0.6)",
                text=avg_cc.rp.apply(lambda x:f"{x:.1f}%"), textposition="outside"))
            fig_rp_cc.update_layout(**PLOTLY_LAYOUT, title="Avg Renewable % by Country", height=280)
            st.plotly_chart(fig_rp_cc, use_container_width=True)

    with tab4:
        b1,b2 = st.columns(2)
        with b1:
            fig_b1 = go.Figure()
            fig_b1.add_trace(go.Bar(x=budg.month_name, y=budg.budget_kg, name="Budget", marker_color="rgba(0,255,135,0.4)"))
            fig_b1.add_trace(go.Bar(x=budg.month_name, y=budg.used_kg,   name="Used",   marker_color="rgba(255,68,68,0.7)"))
            fig_b1.update_layout(**PLOTLY_LAYOUT, title="Carbon Budget: Used vs Allocated", height=260, barmode="group",
                legend=dict(bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_b1, use_container_width=True)
        with b2:
            fig_b2 = go.Figure(go.Bar(x=budg.month_name, y=budg.pct_used,
                marker_color=["#ff4444" if v>100 else "#ffd700" if v>75 else "#00ff87" for v in budg.pct_used],
                text=budg.pct_used.apply(lambda x:f"{x:.0f}%"), textposition="outside"))
            fig_b2.add_hline(y=100, line_color="#ff4444", line_dash="dash", annotation_text="100% limit")
            fig_b2.update_layout(**PLOTLY_LAYOUT, title="Budget Utilisation %", height=260)
            st.plotly_chart(fig_b2, use_container_width=True)
        st.dataframe(budg, use_container_width=True, hide_index=True)
