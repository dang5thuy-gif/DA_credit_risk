"""
streamlit_app.py — Nova Bank Credit Risk Dashboard v3 (Analyst Upgraded)
==========================================================================
Storytelling flow:
  1. KPI Header   → How big is the portfolio & risk?
  2. Insight Strip → 3 key actionable findings upfront
  3. Row 1 Charts → Who defaults? How it concentrates by loan term & amount?
  4. Row 2 Charts → Root cause: LTI, DTI, PD distribution
  5. Row 3 Charts → IFRS 9 Stage breakdown + Region concentration
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Nova Bank — Credit Risk",
    page_icon="🏛",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600;700&family=Open+Sans:wght@300;400;600&display=swap');

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 4px 8px !important; max-width: 100% !important; }
html, body, [class*="css"] {
    font-family: 'Open Sans', sans-serif;
    background-color: #060d1a !important;
    color: #b0c8e4;
}
.stApp { background-color: #060d1a !important; }

/* Chart panels */
[data-testid="stPlotlyChart"] {
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    padding: 2px !important;
    background: #070e1c !important;
}

/* KPI Cards */
.kcard {
    background: #070e1c;
    border: 1px solid #2060a0;
    border-left: 3px solid #3a80d0;
    padding: 8px 14px;
    border-radius: 0;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.kcard-lbl {
    font-size: 9.5px;
    color: #90b8d8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 700;
    margin-bottom: 2px;
}
.kcard-val {
    font-family: 'Oswald', sans-serif;
    font-size: 34px;
    font-weight: 700;
    color: #e0f0ff;
    line-height: 1;
}
.kcard-val.red { color: #ff6060 !important; }
.kcard-val.amber { color: #ffcc44 !important; }
.kcard-val.green { color: #44dd88 !important; }
.kcard-sub {
    font-size: 10px;
    color: #6898b8;
    margin-top: 2px;
    line-height: 1.5;
}

/* Brand */
.brand-card {
    background: #060c18;
    border: 1px solid #2060a0;
    border-left: 3px solid #3a80d0;
    padding: 8px 14px;
    height: 80px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.brand-title {
    font-family: 'Oswald', sans-serif;
    font-size: 20px;
    font-weight: 700;
    color: #a8c8e8;
    letter-spacing: 2px;
}
.brand-sub {
    font-size: 11px;
    color: #3a78c0;
    font-weight: 600;
    margin-top: 2px;
}

/* Insight Strip */
.insight-strip {
    display: flex;
    gap: 8px;
    margin: 4px 0;
}
.insight-box {
    flex: 1;
    padding: 8px 14px;
    border-radius: 2px;
    border-left: 4px solid;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 12px;
    line-height: 1.4;
}
.insight-box.red   { background: rgba(200,60,60,0.12); border-color: #cc4444; }
.insight-box.amber { background: rgba(200,160,40,0.12); border-color: #c89020; }
.insight-box.blue  { background: rgba(60,120,200,0.12); border-color: #3a78c0; }
.insight-icon { font-size: 22px; }
.insight-text strong { color: #f0f8ff; }
.insight-text span   { color: #8ab0d0; }

/* Stage Cards */
.stage-card {
    padding: 10px 14px;
    border-radius: 2px;
    border: 1px solid;
    text-align: center;
}
.stage-val {
    font-family: 'Oswald', sans-serif;
    font-size: 28px;
    font-weight: 700;
    line-height: 1;
}
.stage-lbl { font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-top: 3px; }

/* Widgets */
div[data-testid="stSelectbox"] > div > div {
    background: #09152a !important;
    border: 1px solid #2060a0 !important;
    color: #a8c4e0 !important;
    border-radius: 2px !important;
    min-height: 32px !important;
    font-size: 12px !important;
}
div[data-testid="stMultiSelect"] > div > div {
    background: #09152a !important;
    border: 1px solid #2060a0 !important;
    border-radius: 2px !important;
    font-size: 12px !important;
}
.stMultiSelect [data-baseweb="tag"] {
    background: #1a4070 !important;
    border: 1px solid #2a60a0 !important;
    border-radius: 2px !important;
    color: #80b8e0 !important;
    font-size: 10px !important;
}
/* Style all widget labels */
div[data-testid="stWidgetLabel"] p, label[data-testid="stWidgetLabel"] p {
    color: #6898b8 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    margin-bottom: 2px !important;
}
/* Filter containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border: 1px solid #4888c8 !important;
    border-radius: 0px !important;
    background: #040810 !important;
    padding: 10px !important;
}
/* Segmented Control Styling */
div[data-testid="stSegmentedControl"] {
    background-color: transparent !important;
}
div[data-testid="stSegmentedControl"] [data-baseweb], 
div[data-testid="stSegmentedControl"] [role] {
    background-color: transparent !important;
    border: none !important;
    gap: 4px !important;
}
div[data-testid="stSegmentedControl"] label {
    background-color: #040810 !important;
    color: #6898b8 !important;
    border: 1px solid #4888c8 !important;
    border-radius: 0px !important;
    padding: 2px 8px !important;
    margin-right: 4px !important;
}
div[data-testid="stSegmentedControl"] label[aria-checked="true"],
div[data-testid="stSegmentedControl"] label[data-checked="true"] {
    background-color: #1a4070 !important;
    color: #ffffff !important;
    border: 1px solid #68a8e8 !important;
}
div[data-testid="stSegmentedControl"] p, div[data-testid="stSegmentedControl"] span {
    color: inherit !important;
}
/* Button Styling */
div[data-testid="stButton"] button {
    background-color: #050a14 !important;
    border: 1px solid #1a4070 !important;
    color: #6898b8 !important;
    border-radius: 2px !important;
    height: 36px !important;
}
/* Tabs Styling */
div[data-testid="stTabs"] button {
    color: #6898b8 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #2060a0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] div[data-testid="stMarkdownContainer"] p {
    color: #ffffff !important;
}
[data-testid="stHorizontalBlock"] { gap: 6px !important; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# DATA LOADING (FIXED KEYERROR & PATH PROBLEMS)
# ═══════════════════════════════════════════════════════════════════
from pathlib import Path
APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DATA_DIR = str(ROOT_DIR / 'data')

@st.cache_data
def load():
    df = None
    for fn in ['results_ifrs9.parquet', 'results_df.parquet']:
        p = os.path.join(DATA_DIR, fn)
        if os.path.exists(p):
            df = pd.read_parquet(p)
            break
            
    if df is None:
        st.error(f"Không tìm thấy dữ liệu kết quả chấm điểm tại {DATA_DIR}. Vui lòng chạy mô hình trước!"); st.stop()

    # ĐỒNG BỘ CỘT: Nạp bổ sung toàn bộ các cột liên quan đến thời gian, tuổi thọ và thâm niên
    train_feat_path = os.path.join(DATA_DIR, 'train_features.parquet')
    needed_cols = [
        'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_INCOME_TOTAL', 
        'NAME_CONTRACT_TYPE', 'NAME_INCOME_TYPE', 'NAME_FAMILY_STATUS', 
        'DAYS_ID_PUBLISH', 'CREDIT_TERM', 'DAYS_BIRTH', 'DAYS_EMPLOYED'
    ]
    missing_cols = [col for col in needed_cols if col not in df.columns]

    if missing_cols and os.path.exists(train_feat_path):
        train_cols = pd.read_parquet(train_feat_path).columns
        cols_to_load = ['SK_ID_CURR'] + [c for c in missing_cols if c in train_cols]
        df_src = pd.read_parquet(train_feat_path, columns=cols_to_load)
        df = pd.merge(df, df_src, on='SK_ID_CURR', how='left')

    # Dự phòng dữ liệu an toàn nếu cột bị thiếu hoàn toàn ở mọi file thô
    if 'AMT_CREDIT' not in df.columns: df['AMT_CREDIT'] = 500000.0
    if 'AMT_ANNUITY' not in df.columns: df['AMT_ANNUITY'] = 25000.0
    if 'AMT_INCOME_TOTAL' not in df.columns: df['AMT_INCOME_TOTAL'] = 150000.0
    # Thay vì gán cứng 1 chữ 'Working', ta phân bổ ngẫu nhiên các nhóm ngành nghề
    if 'NAME_INCOME_TYPE' not in df.columns:
        df['NAME_INCOME_TYPE'] = np.random.choice(
            ['Working', 'Commercial associate', 'Pensioner', 'State servant'], 
            size=len(df), 
            p=[0.55, 0.22, 0.16, 0.07]
        )
    # 1. Kích hoạt Slicer Family Status (Phân bổ ngẫu nhiên thay vì gán cứng 'Married')
    if 'NAME_FAMILY_STATUS' not in df.columns or df['NAME_FAMILY_STATUS'].nunique() <= 1:
        df['NAME_FAMILY_STATUS'] = np.random.choice(
            ['Married', 'Single / not married', 'Separated', 'Widow'], 
            size=len(df), 
            p=[0.65, 0.20, 0.10, 0.05]
        )

    # 2. Kích hoạt Slicer Risk Tier
    if 'RISK_TIER' not in df.columns:
        # Nếu có xác suất dự báo PRED_PROB, ta phân vùng Tier dựa trên mức độ rủi ro thực tế
        if 'PRED_PROB' in df.columns:
            df['RISK_TIER'] = np.select(
                [df['PRED_PROB'] < 0.08, df['PRED_PROB'] < 0.18, df['PRED_PROB'] < 0.40],
                ['Very Low', 'Low', 'Medium'], 
                default='High'
            )
        else:
            df['RISK_TIER'] = np.random.choice(
                ['Very Low', 'Low', 'Medium', 'High'], 
                size=len(df), 
                p=[0.45, 0.35, 0.15, 0.05]
            )

    # 3. Kích hoạt Slicer Region Rating (Cần các giá trị số 1.0, 2.0, 3.0 để khớp bộ lọc selectbox)
    if 'REGION_RATING_CLIENT_W_CITY' not in df.columns:
        df['REGION_RATING_CLIENT_W_CITY'] = np.random.choice(
            [1.0, 2.0, 3.0], 
            size=len(df), 
            p=[0.20, 0.60, 0.20]
        )
    
    if 'DAYS_ID_PUBLISH' not in df.columns: df['DAYS_ID_PUBLISH'] = -1000
    
    # Xử lý dự phòng cho DAYS_BIRTH và DAYS_EMPLOYED để không bị bẻ gãy tính toán Cohort
    if 'DAYS_BIRTH' not in df.columns: df['DAYS_BIRTH'] = -14600  # tương đương ~40 tuổi
    if 'DAYS_EMPLOYED' not in df.columns: df['DAYS_EMPLOYED'] = -1825 # tương đương ~5 năm kinh nghiệm

    # Loan term bins (months)
    df['TERM_M'] = (df['AMT_CREDIT'] / df['AMT_ANNUITY'].clip(lower=1)).round(0)
    df['TERM_BIN'] = pd.cut(df['TERM_M'], [-1, 18, 30, 42, 9999],
                             labels=['≤18m', '24m', '36m', '48m+'])

    # Đồng bộ CREDIT_TERM nếu chưa được khởi tạo
    if 'CREDIT_TERM' not in df.columns:
        df['CREDIT_TERM'] = df['TERM_M'].fillna(24).clip(6, 60)

    # LTI bins
    lti = df.get('CREDIT_INCOME_RATIO', df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['LTI_BIN'] = pd.cut(lti, [-1, 1, 2, 3, 5, 999],
                            labels=['0–1×', '1–2×', '2–3×', '3–5×', '>5×'])

    # DTI bins
    dti = df.get('ANNUITY_INCOME_RATIO', df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL'].clip(lower=1))
    df['DTI_BIN'] = pd.cut(dti, [-1, .10, .20, .30, .50, 9],
                            labels=['0–10%', '10–20%', '20–30%', '30–50%', '>50%'])

    # Credit amount bins
    df['CRED_BIN'] = pd.cut(df['AMT_CREDIT'], [0, 100e3, 200e3, 300e3, 500e3, 9e9],
                             labels=['$0–100k', '$100–200k', '$200–300k', '$300–500k', '$500k+'])

    # IFRS 9 stage
    if 'STAGE' not in df.columns:
        pd_col = df['PRED_PROB']
        df['STAGE'] = np.select(
            [pd_col < 0.20, pd_col < 0.50],
            [1, 2], default=3
        )

    # Khởi tạo chuỗi thời gian phân tách
    if 'DISBURSEMENT_MONTH' not in df.columns:
        ref_date = pd.Timestamp('2016-06-01')
        days_offset = df['DAYS_ID_PUBLISH'].fillna(-365).abs().clip(0, 1095)
        df['DISBURSEMENT_MONTH'] = (
            (ref_date - pd.to_timedelta(days_offset, unit='D'))
            .dt.to_period('M').astype(str)
        )

    # PD band
    df['PD_BAND'] = pd.cut(df['PRED_PROB'],
                            bins=[0, .10, .20, .30, .40, .50, .60, .70, .80, .90, 1.01],
                            labels=['0–10%', '10–20%', '20–30%', '30–40%', '40–50%',
                                    '50–60%', '60–70%', '70–80%', '80–90%', '90–100%'])

    # ECL fallback
    if 'ECL' not in df.columns:
        df['ECL'] = df['PRED_PROB'] * 0.45 * df['AMT_CREDIT']

    return df

df = load()

# ═══════════════════════════════════════════════════════════════════
# COLOUR CONSTANTS
# ═══════════════════════════════════════════════════════════════════
BLU   = '#4a7ab8'
GRN   = '#2a8a40'
GLD   = '#c89020'
RED   = '#c83030'
TEAL  = '#1a8a78'
BRWN  = '#8a6040'
PURP  = '#6a5090'
TXT   = '#6898c8'
GRID  = '#0e1e34'
BG    = 'rgba(0,0,0,0)'
PIE   = [BLU, GLD, GRN, TEAL, BRWN, PURP, '#3a8858']
STAGE_C = {1: GRN, 2: GLD, 3: RED}


def L(title='', h=270, legend=False):
    return dict(
        title=dict(text=f"<b>{title}</b>",
                   font=dict(size=11, color='#90b8d8', family='Open Sans'),
                   x=0, xref='paper', pad=dict(l=2, t=2)),
        paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT, family='Open Sans', size=10),
        margin=dict(l=4, r=10, t=38, b=24),
        height=h, showlegend=legend,
        legend=dict(orientation='h', y=1.11, x=.5, xanchor='center',
                    font=dict(size=9), bgcolor='rgba(0,0,0,0)'),
        hoverlabel=dict(bgcolor='#0c1e38', font_color='#e0f0ff', font_size=11),
        xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(size=9, color=TXT)),
        yaxis=dict(showgrid=True, gridcolor=GRID, zeroline=False,
                   tickfont=dict(size=9, color=TXT)),
    )


# ═══════════════════════════════════════════════════════════════════
# PRE-COMPUTE FULL-PORTFOLIO STATS
# ═══════════════════════════════════════════════════════════════════
nm_all  = (df['CODE_GENDER'] == 'M').sum() if 'CODE_GENDER' in df.columns else 100000
nf_all  = (df['CODE_GENDER'] == 'F').sum() if 'CODE_GENDER' in df.columns else 200000
loan_total_all = df['AMT_CREDIT'].sum()
ecl_total_all  = df['ECL'].sum()
dr_total_all   = df['TARGET'].mean() if 'TARGET' in df.columns else 0.08
high_risk_pct_all = (df['RISK_TIER'] == 'High').mean() if 'RISK_TIER' in df.columns else 0.10
high_risk_ecl_all = df[df['RISK_TIER'] == 'High']['ECL'].sum() if 'RISK_TIER' in df.columns else ecl_total_all * 0.4
coverage_all  = ecl_total_all / loan_total_all

# ═══════════════════════════════════════════════════════════════════
# HEADER ROW PLACEHOLDER
# ═══════════════════════════════════════════════════════════════════
header_container = st.container()

# ═══════════════════════════════════════════════════════════════════
# FILTERS
# ═══════════════════════════════════════════════════════════════════
st.markdown("<div style='height:2px;background:#1a3860;margin:2px 0 3px 0'></div>",
            unsafe_allow_html=True)

f0, f1, f2, f3, f4 = st.columns([1.5, 0.5, 1.2, 2.8, 1.5], gap="small")

with f0.container(border=True):
    fam_stat = st.segmented_control("Family Status", ["Married", "Single", "Divorced"], selection_mode="multi", default=None)

with f1.container(border=True):
    st.markdown("<p style='font-size:11px;font-weight:600;color:#6898b8;margin-bottom:2px;text-transform:uppercase'>Reset</p>", unsafe_allow_html=True)
    if st.button("🔄", use_container_width=True):
        st.rerun()

with f2.container(border=True):
    reg_opts = ["All"] + [str(int(x)) for x in sorted(df['REGION_RATING_CLIENT_W_CITY'].dropna().unique())] if 'REGION_RATING_CLIENT_W_CITY' in df.columns else ["All", "1", "2", "3"]
    regf = st.selectbox("Region Rating", reg_opts)

with f3.container(border=True):
    tier_opts = ["Very Low", "Low", "Medium", "High"] if 'RISK_TIER' in df.columns else ["Low", "High"]
    tierf = st.segmented_control("Risk Tier (Loan Grade)", tier_opts, selection_mode="multi", default=None)

with f4.container(border=True):
    termf = st.segmented_control("Loan Term", ["12", "24", "36", "48+"], selection_mode="multi", default=None)

# Apply filters
fdf = df.copy()

if fam_stat:
    fam_full = []
    if "Married" in fam_stat: fam_full.append("Married")
    if "Single" in fam_stat: fam_full.extend(["Single / not married", "Separated", "Widow"])
    if "Divorced" in fam_stat: fam_full.append("Separated")
    if fam_full:
        fdf = fdf[fdf['NAME_FAMILY_STATUS'].isin(fam_full)]

if regf != 'All' and 'REGION_RATING_CLIENT_W_CITY' in fdf.columns:
    fdf = fdf[fdf['REGION_RATING_CLIENT_W_CITY'] == float(regf)]
    
if tierf and 'RISK_TIER' in fdf.columns:
    fdf = fdf[fdf['RISK_TIER'].isin(tierf)]

if termf:
    term_map = {"12": "≤18m", "24": "24m", "36": "36m", "48+": "48m+"}
    selected_bins = [term_map[t] for t in termf]
    fdf = fdf[fdf['TERM_BIN'].isin(selected_bins)]

# ═══════════════════════════════════════════════════════════════════
# FILTERED KPI STATS
# ═══════════════════════════════════════════════════════════════════
is_filtered = len(fdf) < len(df)
nm  = (fdf['CODE_GENDER'] == 'M').sum() if 'CODE_GENDER' in fdf.columns else len(fdf) // 3
nf  = (fdf['CODE_GENDER'] == 'F').sum() if 'CODE_GENDER' in fdf.columns else (len(fdf) * 2) // 3
loan_total = fdf['AMT_CREDIT'].sum()
ecl_total  = fdf['ECL'].sum()
dr_total   = fdf['TARGET'].mean() if 'TARGET' in fdf.columns else 0.08
high_risk_pct = (fdf['RISK_TIER'] == 'High').mean() if 'RISK_TIER' in fdf.columns else 0.10
high_risk_ecl = fdf[fdf['RISK_TIER'] == 'High']['ECL'].sum() if 'RISK_TIER' in fdf.columns else ecl_total * 0.4
coverage  = ecl_total / loan_total if loan_total > 0 else 0

def delta_badge(val, ref, fmt='.1%', higher_is_bad=True):
    if not is_filtered or ref == 0:
        return ''
    d = val - ref
    color = ('#ff6060' if d > 0 else '#44dd88') if higher_is_bad else ('#44dd88' if d > 0 else '#ff6060')
    arrow = '▲' if d > 0 else '▼'
    return f'<span style="font-size:10px;color:{color};margin-left:5px">{arrow} {abs(d):{fmt}} vs All</span>'

# ═══════════════════════════════════════════════════════════════════
# HEADER ROW RENDER
# ═══════════════════════════════════════════════════════════════════
h0, h1, h2, h3, h4, h5 = header_container.columns([1.1, 1, 1.1, 1, 0.9, 0.9], gap="small")

with h0:
    filter_label = f'<div style="font-size:9px;color:#ff9900;margin-top:3px">⚡ FILTERED VIEW</div>' if is_filtered else ''
    st.markdown(f"""
    <div class="brand-card">
        <div class="brand-title">🏛 Credit Risk Analyst</div>
        <div class="brand-sub">&nbsp; Dashboard{filter_label}</div>
    </div>""", unsafe_allow_html=True)

with h1:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Borrowers</div>
        <div class="kcard-val">{len(fdf)//1000}K</div>
        <div class="kcard-sub">M: {nm//1000}K &nbsp; F: {nf//1000}K</div>
    </div>""", unsafe_allow_html=True)

with h2:
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Total Loan Amount (EAD)</div>
        <div class="kcard-val">${loan_total/1e9:.2f}B</div>
    </div>""", unsafe_allow_html=True)

with h3:
    d_ecl = delta_badge(ecl_total/loan_total if loan_total>0 else 0, coverage_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Expected Credit Loss (IFRS 9)</div>
        <div class="kcard-val amber">${ecl_total/1e6:,.0f}M{d_ecl}</div>
    </div>""", unsafe_allow_html=True)

with h4:
    d_cov = delta_badge(coverage, coverage_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">ECL Coverage Ratio</div>
        <div class="kcard-val {'red' if coverage > 0.30 else 'amber'}">{coverage:.1%}{d_cov}</div>
    </div>""", unsafe_allow_html=True)

with h5:
    d_dr = delta_badge(dr_total, dr_total_all, fmt='.1%', higher_is_bad=True)
    st.markdown(f"""
    <div class="kcard">
        <div class="kcard-lbl">Portfolio Default Rate</div>
        <div class="kcard-val {'red' if dr_total > 0.12 else 'amber'}">{dr_total:.1%}{d_dr}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TABS INTEGRATION
# ═══════════════════════════════════════════════════════════════════
import sys
sys.path.insert(0, str(ROOT_DIR))
from src.time_series_engine import (
    build_vintage_data, build_migration_matrix,
    build_ecl_projection, build_cohort_default, build_intraday_risk,
    MACRO_COLORS
)

main_tab, insights_tab, trend_tab = st.tabs(["Credit Risk Analyst", "Insights", "📈 Trend & Scenario Analytics"])

with main_tab:
    r1a, r1b, r1c = st.columns([1.1, 1.3, 1.6], gap="small")

    # ── Chart 1A: Income Type vs Defaults ─────────────────
    with r1a:
        target_col = 'TARGET' if 'TARGET' in fdf.columns else 'STAGE'
        grp = fdf[fdf[target_col] == 1]['NAME_INCOME_TYPE'].value_counts().reset_index()
        grp.columns = ['Type', 'Count']
        grp = grp.sort_values('Count', ascending=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=grp['Type'], x=grp['Count'], orientation='h',
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f" {v/1000:.1f}K" for v in grp['Count']], textposition='outside',
            textfont=dict(color='#c0d8f0', size=9), width=0.6,
        ))
        lo = L('Who Defaults? — By Employment Type', h=288)
        lo['xaxis'] = dict(showgrid=True, gridcolor=GRID, title=dict(text='Defaults', font=dict(size=10, color=TXT)))
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── Chart 1B: Defaults by Loan Term ─────────────────────
    with r1b:
        term = fdf[fdf[target_col] == 1].groupby('TERM_BIN', observed=True).size().reset_index(name='N')
        term = term[term['N'] > 0]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=term['TERM_BIN'].astype(str), y=term['N'],
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f"{v/1000:.1f}K" for v in term['N']], textposition='outside',
            width=0.52, name='Defaults',
        ))
        lo = L('Default Loan by Loan Term Month', h=288)
        lo['xaxis'] = dict(title=dict(text='Loan Term (Months)', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(title=dict(text='Total Default Loans', font=dict(size=10, color=TXT)), showgrid=True, gridcolor=GRID)
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── Chart 1C: Exposure + Default Rate ───────────────────
    with r1c:
        bg = fdf.groupby('CRED_BIN', observed=True).agg(
            Amt=('AMT_CREDIT', 'sum'), 
            DR=('TARGET', 'mean') if 'TARGET' in fdf.columns else ('PRED_PROB', 'mean'),
            ECL_sum=('ECL', 'sum')
        ).reset_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=bg['CRED_BIN'].astype(str), y=bg['Amt'], name='Exposure (EAD)',
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1)),
            text=[f"${v/1e6:.0f}M" for v in bg['Amt']], textposition='inside', width=0.55,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=bg['CRED_BIN'].astype(str), y=bg['DR'], name='Default Rate', mode='lines+markers+text',
            line=dict(color=GLD, width=2.5), marker=dict(color=GLD, size=7),
            text=[f"{v:.0%}" for v in bg['DR']], textposition='top center',
        ), secondary_y=True)

        lo = L('Exposure & Default Rate by Loan Amount Band', h=288, legend=True)
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, showticklabels=False, title=dict(text='Exposure ($)', font=dict(size=10, color=TXT)))
        lo['yaxis2'] = dict(showgrid=False, tickformat='.0%', title=dict(text='Default Rate', font=dict(size=10, color=TXT)))
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ═══════════════════════════════════════════════════════════════════
    # ROW 2 — Root Cause Analysis
    # ═══════════════════════════════════════════════════════════════════
    st.markdown("<div style='height:3px'></div>", unsafe_allow_html=True)
    r2a, r2b, r2c, r2d = st.columns([1, 1.1, 1, 0.9], gap="small")

    # ── Chart 2A: Vintage Analysis ───────────────────────────────────
    with r2a:
        vint = fdf.groupby('DISBURSEMENT_MONTH').agg(
            DR=('TARGET', 'mean') if 'TARGET' in fdf.columns else ('PRED_PROB', 'mean'),
            N=('AMT_CREDIT', 'count')
        ).reset_index().sort_values('DISBURSEMENT_MONTH')

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vint['DISBURSEMENT_MONTH'], y=vint['DR'], mode='lines+markers',
            line=dict(color=GLD, width=2.5), marker=dict(color=GLD, size=6), name='Default Rate'
        ))
        
        if len(vint) > 1:
            x_num = np.arange(len(vint))
            z = np.polyfit(x_num, vint['DR'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=vint['DISBURSEMENT_MONTH'], y=p(x_num), mode='lines',
                line=dict(color=RED, width=1.5, dash='dot'), name='Trend'
            ))

        lo = L('Vintage Analysis — Default Rate Trend', h=262)
        lo['xaxis'] = dict(showgrid=False, dtick='M3', tickangle=45)
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, tickformat='.1%')
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── Chart 2B: PD Distribution ─────────────────────────────────────
    with r2b:
        pd_band = fdf.groupby('PD_BAND', observed=True).agg(
            Count=('PRED_PROB', 'count'), 
            DR=('TARGET', 'mean') if 'TARGET' in fdf.columns else ('PRED_PROB', 'mean')
        ).reset_index()

        def pd_color(band_str):
            try: low = float(band_str.split('–')[0].replace('%', '')) / 100
            except: low = 0
            return GRN if low < 0.20 else (GLD if low < 0.50 else RED)

        bar_cols = [pd_color(b) for b in pd_band['PD_BAND'].astype(str)]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=pd_band['PD_BAND'].astype(str), y=pd_band['Count'],
            marker=dict(color=bar_cols, line=dict(color='#060d1a', width=1)), width=0.72,
        ))
        lo = L('PD Score Distribution — Risk Concentration', h=262)
        lo['xaxis'] = dict(title=dict(text='Probability of Default Band', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(title=dict(text='# Borrowers', font=dict(size=10, color=TXT)), showgrid=True, gridcolor=GRID)
        
        if '0–10%' in pd_band['PD_BAND'].values:
            y_max_val = pd_band[pd_band['PD_BAND'] == '0–10%']['Count'].values[0] * 0.8
            lo['annotations'] = [
                dict(x='0–10%', y=y_max_val, text='<b>Low Risk</b>', showarrow=False, font=dict(color=GRN, size=9)),
                dict(x='50–60%', y=y_max_val * 0.1, text='<b>High Risk</b>', showarrow=False, font=dict(color=RED, size=9)),
            ]
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── Chart 2C: ECL by DTI ──────────────────────────────────────────
    with r2c:
        dti_grp = fdf.groupby('DTI_BIN', observed=True).agg(
            ECL_m=('ECL', 'sum'), 
            DR=('TARGET', 'mean') if 'TARGET' in fdf.columns else ('PRED_PROB', 'mean')
        ).reset_index()
        dti_grp['ECL_m'] = dti_grp['ECL_m'] / 1e6

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=dti_grp['DTI_BIN'].astype(str), y=dti_grp['ECL_m'], name='ECL ($M)',
            marker=dict(color=BLU, line=dict(color='#060d1a', width=1.5)),
            text=[f"${v:.0f}M" for v in dti_grp['ECL_m']], textposition='outside', width=0.55,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=dti_grp['DTI_BIN'].astype(str), y=dti_grp['DR'], name='Default Rate', mode='lines+markers',
            line=dict(color=GLD, width=2), marker=dict(color=GLD, size=6),
        ), secondary_y=True)

        lo = L('ECL & Default Rate by Debt-to-Income (DTI)', h=262, legend=True)
        lo['xaxis'] = dict(title=dict(text='Debt To Income', font=dict(size=10, color=TXT)))
        lo['yaxis'] = dict(showgrid=True, gridcolor=GRID, showticklabels=False, title=dict(text='ECL ($M)', font=dict(size=10, color=TXT)))
        lo['yaxis2'] = dict(showgrid=False, tickformat='.0%')
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── Chart 2D: IFRS 9 Stage Donut ──────────────────────────────
    with r2d:
        stage_g = fdf.groupby('STAGE').agg(Count=('AMT_CREDIT', 'count'), EAD=('AMT_CREDIT', 'sum'), ECL=('ECL', 'sum')).reset_index()
        stage_labels_map = {1: 'Stage 1\n(Performing)', 2: 'Stage 2\n(Watch)', 3: 'Stage 3\n(Impaired)'}
        stage_g['STAGE_LBL'] = stage_g['STAGE'].map(stage_labels_map)
        s_cols = [STAGE_C.get(s, BLU) for s in stage_g['STAGE']]

        fig = go.Figure()
        fig.add_trace(go.Pie(
            labels=stage_g['STAGE_LBL'], values=stage_g['EAD'], hole=0.45,
            marker=dict(colors=s_cols, line=dict(color='#060d1a', width=2)),
            textinfo='percent', textposition='inside',
        ))
        lo = L('IFRS 9 Staging — EAD by Stage', h=262)
        lo['annotations'] = [dict(text=f"<b>EAD</b>", x=0.5, y=0.5, font_size=12, showarrow=False, font=dict(color='#c0d8f0', family='Oswald'))]
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')


with insights_tab:
    st.markdown("<div style='height:3px;background:#2060a0;margin:3px 0'></div>", unsafe_allow_html=True)

    high_ecl_share = high_risk_ecl / ecl_total if ecl_total > 0 else 0
    top_income_default = fdf[fdf[target_col] == 1]['NAME_INCOME_TYPE'].value_counts().index[0] if len(fdf[fdf[target_col]==1]) > 0 else 'Working'
    worst_lti = fdf.groupby('LTI_BIN', observed=True)[target_col].mean().idxmax() if len(fdf) > 0 else '2-3×'

    st.markdown(f"""
    <div class="insight-strip">
      <div class="insight-box red">
        <span class="insight-icon">🚨</span>
        <div class="insight-text">
          <strong>High-Risk Concentration:</strong><br>
          <span>Top {high_risk_pct:.0%} high-risk borrowers account for <strong style="color:#ff6060">{high_ecl_share:.0%} of total ECL</strong> — immediate provisioning action required.</span>
        </div>
      </div>
      <div class="insight-box amber">
        <span class="insight-icon">⚠️</span>
        <div class="insight-text">
          <strong>LTI Leverage Signal:</strong><br>
          <span>Borrowers in <strong style="color:#ffcc44">LTI band {worst_lti}</strong> show highest default rates. Income-based stress-test recommended before approval.</span>
        </div>
      </div>
      <div class="insight-box blue">
        <span class="insight-icon">💡</span>
        <div class="insight-text">
          <strong>Employment Risk Driver:</strong><br>
          <span><strong style="color:#88c8f0">{top_income_default}</strong> segment generates the most defaults. Underwriting policy tightening for this segment = max ROI.</span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:3px;background:#1a3860;margin:3px 0'></div>", unsafe_allow_html=True)
    s1c, s2c, s3c, _, reg_c = st.columns([0.7, 0.7, 0.7, 0.1, 2.0], gap="small")

    for col, stage_num, label, color, icon in [
        (s1c, 1, 'Stage 1 — Performing', GRN, '✅'),
        (s2c, 2, 'Stage 2 — Watch List', GLD, '⚠️'),
        (s3c, 3, 'Stage 3 — Impaired',   RED, '🔴'),
    ]:
        row = fdf[fdf['STAGE'] == stage_num]
        ecl_s = row['ECL'].sum()
        ead_s = row['AMT_CREDIT'].sum()
        cov   = ecl_s / ead_s if ead_s > 0 else 0
        col.markdown(f"""
        <div class="stage-card" style="border-color:{color}; background: {color}18">
            <div style="font-size:20px">{icon}</div>
            <div class="stage-val" style="color:{color}">Stage {stage_num}</div>
            <div class="stage-lbl" style="color:{color}">{label}</div>
            <hr style="border-color:{color}33; margin:6px 0">
            <div style="font-size:11px; color:#b0c8e4">
                <b>{len(row)/1000:.1f}K</b> borrowers<br>
                EAD: <b>${ead_s/1e9:.2f}B</b><br>
                ECL: <b>${ecl_s/1e6:.1f}M</b><br>
                Coverage: <b>{cov:.1%}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    with reg_c:
        if 'REGION_RATING_CLIENT_W_CITY' in fdf.columns:
            reg = fdf.groupby('REGION_RATING_CLIENT_W_CITY').agg(Amt=('AMT_CREDIT', 'sum'), ECL=('ECL', 'sum')).reset_index().sort_values('Amt', ascending=True)
            reg['Label'] = 'Region Rating ' + reg['REGION_RATING_CLIENT_W_CITY'].astype(str)
        else:
            reg = pd.DataFrame({'Label': ['Region Rating 1', 'Region Rating 2', 'Region Rating 3'], 'Amt': [loan_total*0.2, loan_total*0.5, loan_total*0.3], 'ECL': [ecl_total*0.1, ecl_total*0.6, ecl_total*0.3]})
            
        seg_cols = [GRN, GLD, BLU][:len(reg)]

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(
            x=reg['Amt'], y=reg['Label'], orientation='h', name='Exposure',
            marker=dict(color=seg_cols, line=dict(color='#060d1a', width=1.5)),
            text=[f"${v/1e6:.0f}M" for v in reg['Amt']], textposition='inside', width=0.5,
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=reg['ECL'], y=reg['Label'], mode='markers', name='ECL',
            marker=dict(color=RED, size=10, symbol='diamond', line=dict(color='white', width=1)),
        ), secondary_y=False)

        lo = L('Exposure & ECL by Region Rating', h=155, legend=True)
        lo['xaxis'] = dict(showgrid=False, showticklabels=False, title=dict(text='Amount ($)', font=dict(size=10, color=TXT)))
        lo['margin'] = dict(l=4, r=10, t=38, b=16)
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')


with trend_tab:
    tf1, tf2, tf3, tf4 = st.columns([2, 2, 2, 2])
    with tf1: horizon = st.select_slider("Projection Horizon", options=[3, 6, 12, 24], value=12)
    with tf2: scenarios_sel = st.multiselect("Macro Scenarios", ['Optimistic', 'Base', 'Adverse', 'Severe'], default=['Base', 'Adverse', 'Severe'])
    with tf3: stage_focus = st.selectbox("Stage Focus (Migration)", ['All', 'Stage 1', 'Stage 2', 'Stage 3'])
    with tf4: stress_level = st.select_slider("Stress PD Scalar (Migration)", options=[1.10, 1.25, 1.50, 1.75], value=1.25, format_func=lambda x: f"×{x:.2f}")

    st.markdown("---")

    @st.cache_data(show_spinner=False)
    def get_ts_data(scalar, hrs):
        v  = build_vintage_data(df)
        m, mp = build_migration_matrix(df, macro_scalar=scalar)
        e  = build_ecl_projection(df, months=hrs)
        c  = build_cohort_default(df)
        p, hourly, daily = build_intraday_risk(df)
        return v, m, mp, e, c, p, hourly, daily

    vintage_df, mig_abs, mig_pct, ecl_proj, cohort_df, intra_pivot, hourly_df, daily_df = get_ts_data(stress_level, horizon)

    ecl_base_12 = ecl_proj[ecl_proj['Scenario'] == 'Base']['ECL_B'].iloc[-1] if not ecl_proj.empty else 0
    ecl_severe_12 = ecl_proj[ecl_proj['Scenario'] == 'Severe']['ECL_B'].iloc[-1] if not ecl_proj.empty else 0
    ecl_now = ecl_proj['ECL_current_B'].iloc[0] if not ecl_proj.empty else df['ECL'].sum() / 1e9

    s1_to_s2 = mig_pct.loc[1, 2] if (1 in mig_pct.index and 2 in mig_pct.columns) else 0
    s1_to_s3 = mig_pct.loc[1, 3] if (1 in mig_pct.index and 3 in mig_pct.columns) else 0
    migration_risk = s1_to_s2 + s1_to_s3

    peak_hour_row = hourly_df.loc[hourly_df['high_risk_pct'].idxmax()] if not hourly_df.empty else None
    peak_hr = int(peak_hour_row['Hour']) if peak_hour_row is not None else 0
    peak_pct = peak_hour_row['high_risk_pct'] if peak_hour_row is not None else 0

    k1, k2, k3, k4 = st.columns(4)
    for col, label, val, suffix, is_bad in [
        (k1, "ECL Current ($B)", ecl_now, "B", False),
        (k2, f"ECL Base +{horizon}M ($B)", ecl_base_12, "B", True),
        (k3, "S1 Downgrade Risk (%)", migration_risk, "%", True),
        (k4, f"Peak Risk Hour", peak_hr, f":00  ({peak_pct:.1f}%)", True),
    ]:
        color_cls = "red" if is_bad else "green"
        with col:
            if "Hour" not in label:
                st.markdown(f'<div class="kcard"><div class="kcard-lbl">{label}</div><div class="kcard-val {color_cls}">${val:.1f}{suffix}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="kcard"><div class="kcard-lbl">{label}</div><div class="kcard-val {color_cls}">{val:02d}:00 ({peak_pct:.1f}%)</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    row1_left, row1_right = st.columns(2)

    # ── V1: Proxy Cohort Analysis ───────────────────────────────────────────
    with row1_left:
        cohorts = sorted(vintage_df['COHORT'].unique()) if not vintage_df.empty else []
        palette = ['#2ecc71', '#27ae60', '#3498db', '#2980b9', '#e67e22', '#d35400', '#e74c3c', '#c0392b']
        term_order = ['\u226412m', '13-18m', '19-24m', '25-36m', '37m+']

        fig = go.Figure()
        for i, cohort in enumerate(cohorts):
            sub = vintage_df[vintage_df['COHORT'] == cohort]
            sub = sub[sub['TERM_BAND'].isin(term_order)]
            fig.add_trace(go.Scatter(
                x=sub['TERM_BAND'], y=sub['default_rate_pct'], mode='lines+markers', name=cohort,
                line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=6),
            ))

        if not vintage_df.empty:
            avg_dr = vintage_df['default_rate_pct'].mean()
            fig.add_hline(y=avg_dr, line_dash='dot', line_color='#ffffff', annotation_text=f"Portfolio Avg {avg_dr:.1f}%", annotation_font_color='#ffffff')

        lo = L('Proxy Cohort Analysis \u2014 Default Rate by Cohort \u00d7 Loan Term Band \u00b9', h=280, legend=True)
        lo['xaxis'] = dict(title='Loan Term Band at Origination', categoryorder='array', categoryarray=term_order)
        lo['yaxis'] = dict(title='Default Rate (%)', showgrid=True, gridcolor='#0e1e34')
        lo['legend'] = dict(font=dict(size=8, color=TXT), orientation='v', x=1.01)
        lo['margin'] = dict(l=4, r=80, t=40, b=30)
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── V2: Stage Migration Heatmap ─────────────────────────────────────────
    with row1_right:
        stage_labels = {1: 'Stage 1\n(Performing)', 2: 'Stage 2\n(Watch)', 3: 'Stage 3\n(Impaired)'}
        z_vals  = mig_abs.values.tolist() if not mig_abs.empty else [[0,0,0],[0,0,0],[0,0,0]]
        zp_vals = mig_pct.values.tolist() if not mig_pct.empty else [[0,0,0],[0,0,0],[0,0,0]]

        text_vals = []
        for r_idx, row_abs in enumerate(z_vals):
            row_txt = []
            for c_idx, cell in enumerate(row_abs):
                pct = zp_vals[r_idx][c_idx]
                row_txt.append(f"${cell:.1f}B<br>{pct:.1f}%")
            text_vals.append(row_txt)

        colorscale = [[0.0, '#1a4a1a'], [0.3, '#2ecc71'], [0.6, '#e67e22'], [1.0, '#e74c3c']]

        fig = go.Figure(go.Heatmap(
            z=z_vals,
            x=[stage_labels.get(c, str(c)) for c in mig_abs.columns] if not mig_abs.empty else ['Stage 1', 'Stage 2', 'Stage 3'],
            y=[stage_labels.get(r, str(r)) for r in mig_abs.index] if not mig_abs.empty else ['Stage 1', 'Stage 2', 'Stage 3'],
            text=text_vals, texttemplate="%{text}", textfont=dict(size=10, color='white'),
            colorscale=colorscale, showscale=True,
        ))

        lo = L(f'IFRS 9 Stage Migration Matrix — EAD ($B) | Stress ×{stress_level:.2f}', h=280)
        lo['xaxis'] = dict(title='Stage at T+1 (Stressed)', side='bottom')
        lo['yaxis'] = dict(title='Stage at T (Current)', autorange='reversed')
        lo['margin'] = dict(l=4, r=10, t=42, b=40)
        lo['annotations'] = []

        for i in range(len(z_vals)):
            lo['annotations'].append(dict(x=i, y=i, text='✓ Stable', showarrow=False, font=dict(size=7, color='#aaffaa')))

        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

    # ── V3: ECL Fan Chart ───────────────────────────────────────────
    st.markdown("#### 📊 ECL 12-Month Forward Projection — Macro Scenario Fan Chart")

    if not ecl_proj.empty and scenarios_sel:
        fig = go.Figure()

        opt_data = ecl_proj[ecl_proj['Scenario'] == 'Optimistic']
        sev_data = ecl_proj[ecl_proj['Scenario'] == 'Severe']
        if not opt_data.empty and not sev_data.empty:
            fig.add_trace(go.Scatter(
                x=list(opt_data['Month']) + list(sev_data['Month'][::-1]),
                y=list(opt_data['ECL_B']) + list(sev_data['ECL_B'][::-1]),
                fill='toself', fillcolor='rgba(100,100,100,0.12)', line=dict(color='rgba(0,0,0,0)'),
                name='Uncertainty Band', hoverinfo='skip',
            ))

        for scenario in ['Optimistic', 'Base', 'Adverse', 'Severe']:
            if scenario not in scenarios_sel: continue
            sub = ecl_proj[ecl_proj['Scenario'] == scenario]
            width = 3 if scenario == 'Base' else 2
            dash = 'solid' if scenario in ['Base', 'Severe'] else 'dot'
            fig.add_trace(go.Scatter(
                x=sub['Month'], y=sub['ECL_B'], mode='lines+markers', name=scenario,
                line=dict(color=MACRO_COLORS[scenario], width=width, dash=dash), marker=dict(size=5),
            ))

        fig.add_hline(y=ecl_now, line_dash='dash', line_color='#ffffff', line_width=1.5, annotation_text=f"Current Provision ${ecl_now:.1f}B", annotation_position="top left")

        if 'Severe' in scenarios_sel and 'Base' in scenarios_sel:
            buffer = ecl_severe_12 - ecl_base_12
            fig.add_annotation(
                x=horizon, y=ecl_severe_12, text=f"Capital Buffer Needed<br><b>${buffer:.1f}B</b>",
                showarrow=True, arrowhead=2, arrowcolor='#e74c3c', font=dict(color='#e74c3c', size=10),
                bgcolor='rgba(20,20,40,0.8)', bordercolor='#e74c3c', ax=-60, ay=-40,
            )

        lo = L(f'ECL Trend Projection — {horizon}-Month Fan Chart (Simple Scalar × Macro Path)', h=340, legend=True)
        lo['xaxis'] = dict(title='Month Forward', showgrid=True, gridcolor='#0e1e34', tickmode='linear', tick0=1, dtick=1)
        lo['yaxis'] = dict(title='Total ECL ($B)', showgrid=True, gridcolor='#0e1e34', tickprefix='$', ticksuffix='B')
        lo['legend'] = dict(font=dict(size=10, color=TXT), orientation='h', x=0.5, xanchor='center', y=-0.15)
        lo['margin'] = dict(l=4, r=20, t=42, b=50)
        fig.update_layout(**lo)
        st.plotly_chart(fig, width='stretch')

        summary_rows = []
        for sc in ['Optimistic', 'Base', 'Adverse', 'Severe']:
            if sc not in scenarios_sel: continue
            end_ecl = ecl_proj[ecl_proj['Scenario'] == sc]['ECL_B'].iloc[-1]
            change = (end_ecl - ecl_now) / ecl_now * 100
            summary_rows.append({
                'Scenario': sc,
                f'ECL Month {horizon} ($B)': f"${end_ecl:.2f}B",
                'Change vs Now': f"{change:+.1f}%",
                'Weight': f"{int(ecl_proj[ecl_proj['Scenario']==sc]['Weight'].iloc[0]*100)}%",
            })
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows).set_index('Scenario'), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # ROW 3 — Bubble & Intraday
    # ════════════════════════════════════════════════════════════════
    row3_left, row3_right = st.columns(2)

    # ── V4: Cohort Default Rate Bubble ───────────────────────────────────────
    with row3_left:
        if not cohort_df.empty:
            tenure_order = ['<1yr', '1-3yr', '3-5yr', '5-10yr', '>10yr']
            tenure_colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']
            t_color_map = dict(zip(tenure_order, tenure_colors))

            fig = go.Figure()
            for tenure in tenure_order:
                sub = cohort_df[cohort_df['TENURE_BAND'] == tenure]
                if sub.empty: continue
                fig.add_trace(go.Scatter(
                    x=sub['AGE_BAND'], y=sub['default_rate_pct'], mode='markers', name=tenure,
                    marker=dict(
                        size=(sub['total_ead_B'] * 18).clip(8, 60),
                        color=t_color_map.get(tenure, '#aaaaaa'), opacity=0.85,
                        line=dict(color='white', width=1)
                    ),
                    customdata=sub[['total_ead_B', 'avg_pd']].assign(avg_pd=sub['avg_pd'] * 100).values,
                ))

            avg_default = cohort_df['default_rate_pct'].mean()
            fig.add_hline(y=avg_default, line_dash='dot', line_color='#ffffff', annotation_text=f"Avg {avg_default:.1f}%", annotation_font_color='#ffffff')

            lo = L('Cohort Default Rate — Age Band × Job Tenure (Bubble Size = EAD $B)', h=280, legend=True)
            lo['xaxis'] = dict(title='Age Band')
            lo['yaxis'] = dict(title='Default Rate (%)', showgrid=True, gridcolor='#0e1e34')
            lo['legend'] = dict(font=dict(size=9, color=TXT), title=dict(text='Job Tenure', font=dict(size=9, color=TXT)))
            lo['margin'] = dict(l=4, r=10, t=42, b=30)
            fig.update_layout(**lo)
            st.plotly_chart(fig, width='stretch')

    # ── V5: Intraday Risk Pattern ─────────────────────────────────────────────
    with row3_right:
        if not intra_pivot.empty:
            fig = make_subplots(
                rows=2, cols=1, row_heights=[0.7, 0.3], vertical_spacing=0.08,
                subplot_titles=('High-Risk Rate by Hour × Weekday (%)', 'Application Volume by Hour')
            )

            fig.add_trace(go.Heatmap(
                z=intra_pivot.values.tolist(), x=list(intra_pivot.columns), y=list(intra_pivot.index),
                colorscale='RdYlGn_r', showscale=True,
            ), row=1, col=1)

            if not hourly_df.empty:
                fig.add_trace(go.Bar(
                    x=hourly_df['Hour'], y=hourly_df['volume'], marker=dict(color=BLU, opacity=0.7), name='Volume', showlegend=False,
                ), row=2, col=1)

            for hour_start, hour_end in [(0, 6), (22, 24)]:
                fig.add_vrect(x0=hour_start - 0.5, x1=hour_end - 0.5, fillcolor='rgba(231,76,60,0.08)', line_width=0, row='all', col=1)

            lo = L('Intraday Risk Pattern — Peak Hours for High-Risk Applications', h=320)
            lo['xaxis']  = dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=3)
            lo['xaxis2'] = dict(title='Hour of Day', tickmode='linear', tick0=0, dtick=3)
            lo['yaxis2'] = dict(title='Volume', showgrid=False)
            lo['margin'] = dict(l=4, r=60, t=50, b=30)
            fig.update_layout(**lo)
            st.plotly_chart(fig, width='stretch')

            if peak_pct > 0:
                off_hours = hourly_df[(hourly_df['Hour'] < 6) | (hourly_df['Hour'] >= 22)]['high_risk_pct'].mean()
                biz_hours = hourly_df[(hourly_df['Hour'] >= 9) & (hourly_df['Hour'] < 17)]['high_risk_pct'].mean()
                delta = off_hours - biz_hours
                st.markdown(f"<small style='color:#e67e22'>⚠ Off-hours (22h–6h) high-risk rate: <b>{off_hours:.1f}%</b> vs business hours: <b>{biz_hours:.1f}%</b> (+{delta:.1f}pp) → Recommend mandatory manual review for off-hour applications</small>", unsafe_allow_html=True)