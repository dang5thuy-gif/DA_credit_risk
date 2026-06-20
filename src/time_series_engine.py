"""
src/time_series_engine.py
=========================
Time Series & Trend Analysis Engine for Credit Risk Portfolio.

Provides 5 analytical datasets:
  1. build_vintage_data()     - Portfolio cohort vintage analysis
  2. build_migration_matrix() - IFRS 9 Stage migration (EAD-based)
  3. build_ecl_projection()   - 12-month ECL fan chart (simple scalar)
  4. build_cohort_default()   - Default rate by Age x Job Tenure
  5. build_intraday_risk()    - High-risk rate by Hour x Weekday
"""
import pandas as pd
import numpy as np
from pathlib import Path

# ─── Macro scenario paths (12-month PD scalars) ──────────────────────────────
MACRO_PATHS = {
    'Optimistic': [0.98, 0.97, 0.96, 0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87],
    'Base':       [1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
    'Adverse':    [1.05, 1.08, 1.10, 1.12, 1.13, 1.15, 1.16, 1.17, 1.18, 1.19, 1.20, 1.20],
    'Severe':     [1.10, 1.18, 1.25, 1.32, 1.38, 1.45, 1.50, 1.54, 1.57, 1.60, 1.62, 1.65],
}

MACRO_WEIGHTS  = {'Optimistic': 0.20, 'Base': 0.50, 'Adverse': 0.20, 'Severe': 0.10}
MACRO_COLORS   = {'Optimistic': '#2ecc71', 'Base': '#3498db', 'Adverse': '#e67e22', 'Severe': '#e74c3c'}

REF_DATE       = pd.Timestamp('2016-06-01')   # Approximate dataset cut-off date
HIGH_RISK_THRESHOLD = 0.48


# ─────────────────────────────────────────────────────────────────────────────
# 1. Portfolio Vintage Analysis
# ─────────────────────────────────────────────────────────────────────────────
def build_vintage_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build vintage cohort data using DAYS_ID_PUBLISH as a proxy for
    application date. Returns default rate per cohort x loan-term bucket.

    PROXY NOTE: DAYS_ID_PUBLISH = days before REF_DATE when the ID was
    last published. It approximates when the customer was onboarded, but
    is NOT the actual disbursement/application date. Results should be
    labelled 'Proxy Vintage' and not presented as exact vintage curves.
    X-axis = Loan Term Band (CREDIT_TERM buckets), NOT Months on Book.
    """
    dfc = df.copy()

    # FIX: DAYS_ID_PUBLISH is negative (X days BEFORE REF_DATE)
    # so APP_DATE = REF_DATE - abs(DAYS_ID_PUBLISH) = a date in the past
    dfc['APP_DATE'] = REF_DATE - pd.to_timedelta(
        dfc['DAYS_ID_PUBLISH'].abs().clip(0, 1460), unit='D'
    )
    dfc['COHORT'] = dfc['APP_DATE'].dt.to_period('Q').astype(str)

    # Loan Term Band (NOT Months on Book — honest labelling)
    # CREDIT_TERM = total loan duration in months at origination
    dfc['CREDIT_TERM'] = dfc['CREDIT_TERM'].clip(6, 60).fillna(24)
    dfc['TERM_BAND'] = pd.cut(
        dfc['CREDIT_TERM'],
        bins=[0, 12, 18, 24, 36, 60],
        labels=['≤12m', '13-18m', '19-24m', '25-36m', '37m+']
    ).astype(str)

    target_col = 'TARGET' if 'TARGET' in dfc.columns else None
    if target_col:
        vintage = (
            dfc.groupby(['COHORT', 'TERM_BAND'])
            .agg(
                default_rate=(target_col, 'mean'),
                count=('SK_ID_CURR', 'count'),
                avg_pd=('PRED_PROB', 'mean'),
                total_ead=('EAD', 'sum'),
            )
            .reset_index()
        )
        vintage['default_rate_pct'] = vintage['default_rate'] * 100
        vintage['avg_pd_pct'] = vintage['avg_pd'] * 100
    else:
        vintage = (
            dfc.groupby(['COHORT', 'TERM_BAND'])
            .agg(
                avg_pd=('PRED_PROB', 'mean'),
                count=('SK_ID_CURR', 'count'),
                total_ead=('EAD', 'sum'),
            )
            .reset_index()
        )
        vintage['default_rate_pct'] = vintage['avg_pd'] * 100
        vintage['avg_pd_pct'] = vintage['avg_pd'] * 100

    vintage['total_ead_B'] = vintage['total_ead'] / 1e9
    return vintage


# ─────────────────────────────────────────────────────────────────────────────
# 2. Stage Migration Matrix (EAD-based, simple stress simulation)
# ─────────────────────────────────────────────────────────────────────────────
def _classify_stage(pd_series: pd.Series,
                    bureau_bad: pd.Series = None,
                    late_ratio: pd.Series = None) -> pd.Series:
    """Classify Stage purely from PD + behavioral signals (no TARGET leakage)."""
    stage = pd.Series(1, index=pd_series.index, dtype=np.int8)

    s3 = (pd_series > 0.70)
    stage[s3] = 3

    s2 = (
        (pd_series > 0.20) |
        (bureau_bad.fillna(0) == 1 if bureau_bad is not None else False) |
        (late_ratio.fillna(0) > 0.30 if late_ratio is not None else False)
    ) & (stage < 3)
    stage[s2] = 2

    return stage


def build_migration_matrix(df: pd.DataFrame, macro_scalar: float = 1.25) -> pd.DataFrame:
    """
    Build 3x3 Stage Migration matrix (EAD in $B).
    T   = current stage from PRED_PROB
    T+1 = stressed stage: PD * macro_scalar
    Returns pivot table of EAD flows.
    """
    dfc = df.copy()

    bureau_bad  = dfc.get('bureau_bad_debt_flag')
    late_ratio  = dfc.get('inst_late_ratio')

    dfc['Stage_T']  = _classify_stage(dfc['PRED_PROB'], bureau_bad, late_ratio)
    dfc['Stage_T1'] = _classify_stage(
        (dfc['PRED_PROB'] * macro_scalar).clip(0, 0.999),
        bureau_bad, late_ratio
    )

    matrix = (
        dfc.groupby(['Stage_T', 'Stage_T1'])['EAD']
        .sum()
        .reset_index()
    )
    matrix['EAD_B'] = matrix['EAD'] / 1e9

    pivot = matrix.pivot(index='Stage_T', columns='Stage_T1', values='EAD_B').fillna(0)
    # Ensure all stages present
    for s in [1, 2, 3]:
        if s not in pivot.index:
            pivot.loc[s] = 0
        if s not in pivot.columns:
            pivot[s] = 0

    pivot = pivot.sort_index().sort_index(axis=1)

    # Also compute % of row total (% of each Stage's EAD that migrates)
    row_totals = pivot.sum(axis=1)
    pivot_pct = pivot.div(row_totals, axis=0) * 100

    return pivot, pivot_pct


# ─────────────────────────────────────────────────────────────────────────────
# 3. ECL 12-Month Fan Chart (Simple Scalar x Macro Path)
# ─────────────────────────────────────────────────────────────────────────────
def build_ecl_projection(df: pd.DataFrame, months: int = 12) -> pd.DataFrame:
    """
    Project ECL over next N months using simple macro scalar paths.
    ECL(t) = ECL_current x macro_scalar(t) x portfolio_decay(t)

    portfolio_decay: exponential decay as loans mature/repay
    Assume ~2% monthly amortization = (0.98)^t decay factor
    """
    ecl_current = df['ECL'].sum() / 1e9   # $B
    ead_current = df['EAD'].sum() / 1e9   # $B

    month_idx = list(range(1, months + 1))
    decay = [(0.98 ** t) for t in month_idx]   # portfolio natural run-off

    records = []
    for scenario, path in MACRO_PATHS.items():
        scalars = path[:months]
        ecl_path = [ecl_current * s * d for s, d in zip(scalars, decay)]
        for m, ecl_val in zip(month_idx, ecl_path):
            records.append({
                'Month': m,
                'Scenario': scenario,
                'ECL_B': round(ecl_val, 3),
                'Color': MACRO_COLORS[scenario],
                'Weight': MACRO_WEIGHTS[scenario],
            })

    proj = pd.DataFrame(records)
    proj['ECL_current_B'] = ecl_current
    proj['EAD_current_B'] = ead_current
    return proj


# ─────────────────────────────────────────────────────────────────────────────
# 4. Cohort Default Rate (Age Band x Job Tenure)
# ─────────────────────────────────────────────────────────────────────────────
def build_cohort_default(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute default rate by Age Band x Job Tenure Band.
    Uses actual TARGET labels where available, else falls back to PRED_PROB.
    """
    dfc = df.copy()

    dfc['AGE_YEARS'] = (-dfc['DAYS_BIRTH'] / 365).round(0).clip(18, 70)
    dfc['AGE_BAND'] = pd.cut(
        dfc['AGE_YEARS'],
        bins=[18, 25, 35, 45, 55, 70],
        labels=['18-25', '26-35', '36-45', '46-55', '56+']
    ).astype(str)

    dfc['JOB_TENURE_YRS'] = (-dfc['DAYS_EMPLOYED'] / 365).clip(0, 20)
    dfc['TENURE_BAND'] = pd.cut(
        dfc['JOB_TENURE_YRS'],
        bins=[-0.01, 1, 3, 5, 10, 20],
        labels=['<1yr', '1-3yr', '3-5yr', '5-10yr', '>10yr']
    ).astype(str)

    target_col = 'TARGET' if 'TARGET' in dfc.columns else 'PRED_PROB'

    cohort = (
        dfc.groupby(['AGE_BAND', 'TENURE_BAND'], observed=True)
        .agg(
            default_rate=(target_col, 'mean'),
            count=('SK_ID_CURR', 'count'),
            total_ead=('EAD', 'sum'),
            avg_ecl=('ECL', 'mean'),
            avg_pd=('PRED_PROB', 'mean'),
        )
        .reset_index()
    )

    cohort['default_rate_pct'] = cohort['default_rate'] * 100
    cohort['total_ead_B'] = cohort['total_ead'] / 1e9
    cohort['avg_ecl_k'] = cohort['avg_ecl'] / 1e3   # in $K

    age_order = ['18-25', '26-35', '36-45', '46-55', '56+']
    cohort['AGE_BAND'] = pd.Categorical(cohort['AGE_BAND'], categories=age_order, ordered=True)
    cohort = cohort.sort_values('AGE_BAND')

    return cohort


# ─────────────────────────────────────────────────────────────────────────────
# 5. Intraday Risk Pattern (Hour x Weekday)
# ─────────────────────────────────────────────────────────────────────────────
def build_intraday_risk(df: pd.DataFrame) -> tuple:
    """
    Compute high-risk rate (%) for each Hour x Weekday combination.
    Returns: (pivot_heatmap, hourly_summary, daily_summary)
    """
    dfc = df.copy()
    dfc['IS_HIGH_RISK'] = (dfc['PRED_PROB'] > HIGH_RISK_THRESHOLD).astype(int)

    # Decode one-hot WEEKDAY columns
    weekday_cols = [c for c in dfc.columns if 'WEEKDAY_APPR_PROCESS_START' in c]
    day_order = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    if weekday_cols:
        dfc['WEEKDAY_RAW'] = (
            dfc[weekday_cols]
            .idxmax(axis=1)
            .str.replace('WEEKDAY_APPR_PROCESS_START_', '', regex=False)
        )
        day_map = {d: l for d, l in zip(day_order, day_labels)}
        dfc['WEEKDAY'] = dfc['WEEKDAY_RAW'].map(day_map).fillna('Unknown')
    else:
        dfc['WEEKDAY'] = 'Unknown'

    hour_col = 'HOUR_APPR_PROCESS_START'
    if hour_col not in dfc.columns:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # Pivot heatmap: Day x Hour
    pivot = dfc.pivot_table(
        values='IS_HIGH_RISK',
        index='WEEKDAY',
        columns=hour_col,
        aggfunc='mean',
        fill_value=0
    ) * 100  # -> %

    # Reorder weekdays
    existing_days = [l for l in day_labels if l in pivot.index]
    pivot = pivot.reindex(existing_days)

    # Hourly summary
    hourly = (
        dfc.groupby(hour_col)
        .agg(
            high_risk_rate=('IS_HIGH_RISK', 'mean'),
            volume=('SK_ID_CURR', 'count'),
        )
        .reset_index()
    )
    hourly['high_risk_pct'] = hourly['high_risk_rate'] * 100
    hourly.columns = ['Hour', 'high_risk_rate', 'volume', 'high_risk_pct']

    # Daily summary
    daily = (
        dfc.groupby('WEEKDAY')
        .agg(
            high_risk_rate=('IS_HIGH_RISK', 'mean'),
            volume=('SK_ID_CURR', 'count'),
        )
        .reset_index()
    )
    daily['high_risk_pct'] = daily['high_risk_rate'] * 100
    daily = daily[daily['WEEKDAY'].isin(day_labels)]
    daily['WEEKDAY'] = pd.Categorical(daily['WEEKDAY'], categories=day_labels, ordered=True)
    daily = daily.sort_values('WEEKDAY')

    return pivot, hourly, daily
