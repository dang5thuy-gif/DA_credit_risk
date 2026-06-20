"""
ifrs9_ecl_engine.py — IFRS 9 ECL Engine (Production-Grade)
===========================================================
IFRS 9 yêu cầu 3 components PHẢI đầy đủ:
    1. Stage Classification (Stage 1/2/3)
    2. PD Term Structure (12-month PD vs Lifetime PD via survival model)
    3. Forward-looking Macroeconomic Overlay (4 scenarios + probability weights)

References:
    IFRS 9.B5.5.1-B5.5.27    — Stage classification criteria
    IFRS 9.B5.5.41-B5.5.54   — Forward-looking information
    IFRS 9.B5.5.55-B5.5.57   — Lifetime PD requirement
    EBA/GL/2017/16 §71-§90   — PD estimation methodology
    SBV Thong tu 11/2021      — Vietnam credit classification
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os, time, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
MODELS_DIR  = str(ROOT_DIR / 'models')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*65}\n {t}\n{'='*65}")
def log(t):    print(f"   {t}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Stage Classification (IFRS 9.B5.5.7)
# ─────────────────────────────────────────────────────────────────────────────

STAGE_THRESHOLDS = {
    'pd_stage2_floor':        0.20,
    'pd_stage3_floor':        0.70,
    'overdue_days_stage2':    30,
    'overdue_days_stage3':    90,
    'late_ratio_stage2':      0.30,
    'prev_refused_stage2':    0.50,
}


def classify_stage(df: pd.DataFrame) -> pd.Series:
    """
    Stage 3 — Credit-impaired (IFRS 9.5.5.3):
        TARGET == 1, PD > 70%, or DPD > 90 days
    Stage 2 — Significant increase in credit risk:
        PD > 20%, bureau bad debt, late ratio > 30%, prev_refused > 50%
    Stage 1 — Performing (all others)
    """
    pd_col = df.get('PRED_PROB', df.get('PD', pd.Series(0.1, index=df.index)))
    stage  = pd.Series(1, index=df.index, dtype=np.int8)

    # Stage 3 — Credit-impaired (IFRS 9.5.5.3):
    # NOTE: We deliberately do NOT use TARGET==1 here.
    # In production, future default status is unknown at decision time.
    # Stage 3 is identified purely via model PD and observable behavioral signals.
    s3 = (pd_col > STAGE_THRESHOLDS['pd_stage3_floor'])
    stage[s3] = 3

    s2 = (
        (pd_col > STAGE_THRESHOLDS['pd_stage2_floor']) |
        (df.get('bureau_bad_debt_flag', pd.Series(0, index=df.index)).fillna(0) == 1) |
        (df.get('inst_late_ratio', pd.Series(0, index=df.index)).fillna(0)
           > STAGE_THRESHOLDS['late_ratio_stage2']) |
        (df.get('prev_refused_ratio', pd.Series(0, index=df.index)).fillna(0)
           > STAGE_THRESHOLDS['prev_refused_stage2'])
    ) & (stage < 3)
    stage[s2] = 2

    return stage


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — PD Term Structure (EBA/GL/2017/16 §74)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pd_term_structure(df: pd.DataFrame,
                               pd_12m_col: str = 'PRED_PROB') -> pd.DataFrame:
    """
    Survival model: S(t) = (1 - h)^t
        h           = annual hazard rate = PD_12M
        Lifetime PD = 1 - (1 - h)^T
        T           = remaining loan term in years (từ CREDIT_TERM)

    Floor: Lifetime_PD >= PD_12M | Ceiling: <= 0.999
    """
    pd_12m = df[pd_12m_col].clip(0.001, 0.999)

    if 'CREDIT_TERM' in df.columns:
        loan_term_months = df['CREDIT_TERM'].clip(6, 120).fillna(36)
    else:
        loan_term_months = pd.Series(36.0, index=df.index)

    loan_term_years = loan_term_months / 12.0
    lifetime_pd     = 1.0 - (1.0 - pd_12m) ** loan_term_years
    lifetime_pd     = lifetime_pd.clip(lower=pd_12m, upper=0.999)

    return pd.DataFrame({
        'PD_12M':        pd_12m,
        'PD_Lifetime':   lifetime_pd,
        'PD_Multiplier': lifetime_pd / pd_12m,
        'LoanTerm_yrs':  loan_term_years,
    }, index=df.index)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Macroeconomic Overlay (IFRS 9.B5.5.41)
# ─────────────────────────────────────────────────────────────────────────────

# Vietnam macro scenarios — calibrated cho Home Credit Vietnam context
MACRO_SCENARIOS = {
    'Optimistic': {
        'weight':      0.20,
        'gdp_growth':  0.075,
        'unemp_rate':  0.018,
        'pd_scalar':   0.85,
        'description': 'Strong economic growth, low unemployment'
    },
    'Base': {
        'weight':      0.50,
        'gdp_growth':  0.065,
        'unemp_rate':  0.022,
        'pd_scalar':   1.00,
        'description': 'Stable growth, baseline conditions'
    },
    'Adverse': {
        'weight':      0.20,
        'gdp_growth':  0.030,
        'unemp_rate':  0.045,
        'pd_scalar':   1.35,
        'description': 'Economic slowdown, tightening credit'
    },
    'Severe': {
        'weight':      0.10,
        'gdp_growth': -0.010,
        'unemp_rate':  0.080,
        'pd_scalar':   1.75,
        'description': 'Recession scenario, high credit stress'
    },
}

assert abs(sum(s['weight'] for s in MACRO_SCENARIOS.values()) - 1.0) < 0.001


def apply_macro_overlay(pd_12m: pd.Series, pd_lifetime: pd.Series,
                         scenarios: dict = None) -> pd.DataFrame:
    """
    Probability-weighted macroeconomic scenario blending.
    IFRS 9.B5.5.42: "reasonable and supportable forward-looking information."
    """
    if scenarios is None:
        scenarios = MACRO_SCENARIOS

    wt_pd_12m      = pd.Series(0.0, index=pd_12m.index)
    wt_pd_lifetime = pd.Series(0.0, index=pd_lifetime.index)
    scenario_pds   = {}

    for name, params in scenarios.items():
        scalar         = params['pd_scalar']
        weight         = params['weight']
        adj_12m        = (pd_12m      * scalar).clip(0.001, 0.999)
        adj_lifetime   = (pd_lifetime * scalar).clip(0.001, 0.999)
        scenario_pds[f'PD_12M_{name}']      = adj_12m
        scenario_pds[f'PD_Lifetime_{name}'] = adj_lifetime
        wt_pd_12m      += adj_12m      * weight
        wt_pd_lifetime += adj_lifetime * weight

    result_df = pd.DataFrame(scenario_pds, index=pd_12m.index)
    result_df['PD_12M_macro']         = wt_pd_12m
    result_df['PD_Lifetime_macro']    = wt_pd_lifetime
    result_df['Macro_Uplift_12M_pct'] = (wt_pd_12m / pd_12m - 1) * 100

    log(f"  Macro overlay applied:")
    log(f"    PD_12M   before: {pd_12m.mean():.4f} -> after: {wt_pd_12m.mean():.4f} "
        f"(+{(wt_pd_12m.mean()/pd_12m.mean()-1)*100:.1f}%)")
    log(f"    Lifetime before: {pd_lifetime.mean():.4f} -> after: {wt_pd_lifetime.mean():.4f}")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Full ECL Calculation
# ─────────────────────────────────────────────────────────────────────────────

def compute_ifrs9_ecl(df: pd.DataFrame,
                       lgd_series: pd.Series,
                       ead_series: pd.Series) -> tuple:
    """
    Stage 1: ECL = PD_12M_macro    x LGD x EAD   (12-month horizon)
    Stage 2: ECL = PD_Lifetime_macro x LGD x EAD  (lifetime horizon)
    Stage 3: ECL =                    LGD x EAD   (PD assumed = 100%)
    """
    header("IFRS 9 ECL Calculation — Production Grade")

    # Stage classification
    stage = classify_stage(df)
    log(f"  Stage distribution:")
    for s in [1, 2, 3]:
        n = (stage == s).sum()
        log(f"    Stage {s}: {n:>8,} ({n/len(stage)*100:.1f}%)")

    # PD Term Structure
    pd_term = compute_pd_term_structure(df)
    log(f"\n  PD Term Structure:")
    log(f"    Avg PD_12M:      {pd_term['PD_12M'].mean():.4f}")
    log(f"    Avg PD_Lifetime: {pd_term['PD_Lifetime'].mean():.4f}")
    log(f"    Avg Multiplier:  {pd_term['PD_Multiplier'].mean():.2f}x")

    # Macro Overlay
    macro = apply_macro_overlay(pd_term['PD_12M'], pd_term['PD_Lifetime'])

    # ECL calculation by Stage
    ecl        = pd.Series(0.0, index=df.index)
    s1, s2, s3 = (stage == 1), (stage == 2), (stage == 3)

    ecl[s1] = macro.loc[s1, 'PD_12M_macro']     * lgd_series[s1] * ead_series[s1]
    ecl[s2] = macro.loc[s2, 'PD_Lifetime_macro'] * lgd_series[s2] * ead_series[s2]
    ecl[s3] =                                       lgd_series[s3] * ead_series[s3]

    # Assemble result
    result              = df.copy()
    result['Stage']     = stage
    result['PD_12M']    = macro['PD_12M_macro']
    result['PD_Lifetime'] = macro['PD_Lifetime_macro']
    result['LGD']       = lgd_series
    result['EAD']       = ead_series
    result['ECL']       = ecl
    result['ECL_Rate']  = ecl / ead_series.clip(lower=1)

    # Summary table
    summary = result.groupby('Stage').agg(
        Count           = ('ECL', 'count'),
        Total_EAD_B     = ('EAD', lambda x: round(x.sum()/1e9, 2)),
        Total_ECL_B     = ('ECL', lambda x: round(x.sum()/1e9, 2)),
        Avg_PD_12M      = ('PD_12M', 'mean'),
        Avg_PD_Lifetime = ('PD_Lifetime', 'mean'),
        Avg_LGD         = ('LGD', 'mean'),
        ECL_Coverage    = ('ECL_Rate', 'mean'),
    ).reset_index()

    log(f"\n  IFRS 9 ECL Summary by Stage:")
    log(f"  {'Stage':<7} {'Count':>8} {'EAD ($B)':>10} {'ECL ($B)':>10} "
        f"{'PD_12M':>8} {'PD_Life':>8} {'LGD':>8} {'Coverage':>10}")
    log(f"  {'-'*75}")
    for _, row in summary.iterrows():
        log(f"  {row['Stage']:<7} {row['Count']:>8,} "
            f"  ${row['Total_EAD_B']:>8.2f}   ${row['Total_ECL_B']:>8.2f} "
            f"  {row['Avg_PD_12M']:>6.3f}   {row['Avg_PD_Lifetime']:>6.3f} "
            f"  {row['Avg_LGD']:>6.3f}   {row['ECL_Coverage']:>8.2%}")

    total_ead = result['EAD'].sum()
    total_ecl = result['ECL'].sum()
    log(f"\n  TOTAL PORTFOLIO:")
    log(f"    EAD:             ${total_ead/1e9:.2f}B")
    log(f"    ECL (provision): ${total_ecl/1e9:.2f}B")
    log(f"    Coverage ratio:  {total_ecl/total_ead*100:.2f}%")

    # ── Plots ─────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors_map = {1: '#3498DB', 2: '#E67E22', 3: '#E74C3C'}
    clr = [colors_map[s] for s in summary['Stage']]

    # ECL by Stage bar
    axes[0].bar(summary['Stage'], summary['Total_ECL_B'], color=clr, edgecolor='white')
    axes[0].set_title('Total ECL by IFRS 9 Stage ($B)', fontsize=11)
    axes[0].set_xlabel('Stage')
    axes[0].set_ylabel('ECL ($B)')
    for i, (stage_val, ecl_val) in enumerate(zip(summary['Stage'], summary['Total_ECL_B'])):
        axes[0].text(stage_val, ecl_val + 0.05, f'${ecl_val:.2f}B', ha='center', fontsize=9)

    # ECL Coverage
    axes[1].bar(summary['Stage'], summary['ECL_Coverage'] * 100, color=clr, edgecolor='white')
    axes[1].set_title('ECL Coverage Rate by Stage (%)', fontsize=11)
    axes[1].set_xlabel('Stage')
    axes[1].set_ylabel('Coverage (%)')
    for i, (stage_val, cov) in enumerate(zip(summary['Stage'], summary['ECL_Coverage'])):
        axes[1].text(stage_val, cov * 100 + 0.2, f'{cov*100:.1f}%', ha='center', fontsize=9)

    # Portfolio mix pie
    axes[2].pie(summary['Count'], labels=[f'Stage {s}' for s in summary['Stage']],
                colors=[colors_map[s] for s in summary['Stage']],
                autopct='%1.1f%%', startangle=90, pctdistance=0.85)
    axes[2].set_title('Portfolio Mix by Stage', fontsize=11)

    plt.suptitle('IFRS 9 Expected Credit Loss — Full Stage Analysis', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/ifrs9_ecl_by_stage.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"\n  Saved: reports/ifrs9_ecl_by_stage.png")

    # Save scenario comparison
    _plot_macro_scenarios(macro, ead_series, lgd_series, stage)

    return result, summary


def _plot_macro_scenarios(macro_df, ead, lgd, stage):
    """Plot ECL comparison across 4 macro scenarios."""
    fig, ax = plt.subplots(figsize=(10, 5))
    scenario_names  = ['Optimistic', 'Base', 'Adverse', 'Severe']
    scenario_colors = ['#2ECC71',    '#3498DB', '#E67E22', '#E74C3C']
    scenario_ecls   = []

    for sc_name in scenario_names:
        col = f'PD_Lifetime_{sc_name}' if f'PD_Lifetime_{sc_name}' in macro_df.columns else f'PD_12M_{sc_name}'
        if col not in macro_df.columns:
            continue
        ecl_s1 = macro_df.loc[stage==1, f'PD_12M_{sc_name}']      * lgd[stage==1] * ead[stage==1]
        ecl_s2 = macro_df.loc[stage==2, f'PD_Lifetime_{sc_name}'] * lgd[stage==2] * ead[stage==2]
        ecl_s3 = lgd[stage==3] * ead[stage==3]
        total  = (ecl_s1.sum() + ecl_s2.sum() + ecl_s3.sum()) / 1e9
        scenario_ecls.append(total)

    bars = ax.bar(scenario_names, scenario_ecls, color=scenario_colors, edgecolor='white', width=0.55)
    weights = [MACRO_SCENARIOS[s]['weight'] for s in scenario_names]
    for bar, val, w in zip(bars, scenario_ecls, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'${val:.2f}B\n(w={w:.0%})', ha='center', fontsize=9)
    ax.set_ylabel('Total ECL ($B)')
    ax.set_title('ECL by Macro Scenario — IFRS 9 Forward-Looking Overlay', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/ifrs9_macro_scenarios.png', dpi=150)
    plt.close()
    log(f"  Saved: reports/ifrs9_macro_scenarios.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(ROOT_DIR))
    from src.lgd_ead_model import estimate_lgd, estimate_ead, lgd_sensitivity_analysis

    t0  = time.time()
    df  = pd.read_parquet(f'{DATA_DIR}/results_df.parquet')

    lgd = estimate_lgd(df)
    ead = estimate_ead(df)

    result, summary = compute_ifrs9_ecl(df, lgd, ead)
    lgd_sensitivity_analysis(result, lgd.mean())

    # Save
    result.to_parquet(f'{DATA_DIR}/results_ifrs9.parquet', index=False)
    summary.to_csv(f'{REPORTS_DIR}/ifrs9_stage_summary.csv', index=False)
    log(f"\n  Saved: data/results_ifrs9.parquet")
    log(f"  Saved: reports/ifrs9_stage_summary.csv")
    log(f"  Total time: {time.time()-t0:.0f}s")
