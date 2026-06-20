"""
woe_iv_scorecard.py — Phase 2b: Basel II Scorecard via WoE / IV
===============================================================
Weight of Evidence (WoE) và Information Value (IV) là tiêu chuẩn
bắt buộc trong Basel II Internal Ratings-Based (IRB) approach.

IV Guidelines (Siddiqi 2006 — chuẩn ngành):
    IV < 0.02  → Useless     → drop
    0.02-0.10  → Weak        → keep with caution
    0.10-0.30  → Medium      → good predictor
    0.30-0.50  → Strong      → excellent predictor
    > 0.50     → Suspicious  → check for data leakage

Output:
    reports/iv_ranking.csv       — IV của tất cả features
    reports/iv_ranking.png       — Bar chart IV
    data/train_woe.parquet       — WoE-transformed features
    data/woe_maps.json           — WoE lookup tables (production use)
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
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*65}\n {t}\n{'='*65}")
def log(t):    print(f"   {t}")


# ─────────────────────────────────────────────────────────────────────────────
# Core WoE / IV computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_woe_iv(series: pd.Series, target: pd.Series, n_bins: int = 10) -> dict:
    """
    Tính WoE và IV cho một feature.
    Trả về dict: iv, flag, woe_map, bin_table
    """
    total_events    = int(target.sum())
    total_nonevents = int((target == 0).sum())

    if total_events == 0 or total_nonevents == 0:
        return {'iv': 0.0, 'flag': 'Useless', 'woe_map': {}, 'bin_table': None}

    df_tmp = pd.DataFrame({'x': series, 'y': target}).dropna()

    # Bin the feature
    if df_tmp['x'].dtype in [object, bool] or df_tmp['x'].nunique() <= 10:
        df_tmp['bin'] = df_tmp['x'].astype(str)
    else:
        try:
            df_tmp['bin'] = pd.qcut(df_tmp['x'], q=n_bins, duplicates='drop').astype(str)
        except Exception:
            df_tmp['bin'] = pd.cut(df_tmp['x'], bins=n_bins, duplicates='drop').astype(str)

    grp = df_tmp.groupby('bin')['y'].agg(['sum', 'count']).reset_index()
    grp.columns = ['bin', 'events', 'total']
    grp['nonevents'] = grp['total'] - grp['events']

    # Laplace smoothing to avoid log(0)
    grp['pct_ev']  = (grp['events']    + 0.5) / (total_events    + 0.5)
    grp['pct_nev'] = (grp['nonevents'] + 0.5) / (total_nonevents + 0.5)

    grp['woe'] = np.log(grp['pct_ev'] / grp['pct_nev'])
    grp['iv']  = (grp['pct_ev'] - grp['pct_nev']) * grp['woe']

    iv_total = grp['iv'].sum()

    if   iv_total > 0.50: flag = 'Suspicious'
    elif iv_total > 0.30: flag = 'Strong'
    elif iv_total > 0.10: flag = 'Medium'
    elif iv_total > 0.02: flag = 'Weak'
    else:                 flag = 'Useless'

    woe_map = dict(zip(grp['bin'], grp['woe']))
    return {'iv': round(float(iv_total), 6), 'flag': flag,
            'woe_map': woe_map, 'bin_table': grp}


def run_iv_screening(train_df: pd.DataFrame, features: list,
                     target_col: str = 'TARGET') -> pd.DataFrame:
    """Chạy IV screening cho toàn bộ feature list."""
    header("WoE / IV Screening — Basel II Feature Selection")
    target  = train_df[target_col]
    records = []

    for i, feat in enumerate(features):
        result = compute_woe_iv(train_df[feat], target)
        records.append({'rank': i, 'feature': feat,
                        'iv': result['iv'], 'flag': result['flag'],
                        'keep': result['flag'] not in ('Useless', 'Suspicious')})
        if (i + 1) % 20 == 0:
            log(f"Processed {i+1}/{len(features)} features...")

    iv_df = (pd.DataFrame(records)
               .sort_values('iv', ascending=False)
               .reset_index(drop=True))
    iv_df['rank'] = range(1, len(iv_df) + 1)

    log(f"\n IV Screening complete — {len(iv_df)} features analysed")
    log(f" Strong    (IV > 0.30) : {(iv_df['iv'] > 0.30).sum():>4} features")
    log(f" Medium    (0.10-0.30) : {((iv_df['iv'] > 0.10) & (iv_df['iv'] <= 0.30)).sum():>4} features")
    log(f" Weak      (0.02-0.10) : {((iv_df['iv'] > 0.02) & (iv_df['iv'] <= 0.10)).sum():>4} features")
    log(f" Useless   (< 0.02)   : {(iv_df['iv'] < 0.02).sum():>4} features  <- will drop")
    log(f" Suspicious (> 0.50)  : {(iv_df['iv'] > 0.50).sum():>4} features  <- review for leakage")
    log(f"\n Top 25 by IV:")
    for _, row in iv_df.head(25).iterrows():
        bar = chr(9608) * int(row['iv'] * 50)
        log(f"  {int(row['rank']):>3}. {row['feature']:<50} {row['iv']:.4f} {bar} [{row['flag']}]")

    # Save CSV
    iv_df.to_csv(f'{REPORTS_DIR}/iv_ranking.csv', index=False)
    log(f"\n  Saved: reports/iv_ranking.csv")

    # Plot IV bar chart
    top40     = iv_df.head(40).copy()
    color_map = {'Strong': '#E74C3C', 'Medium': '#E67E22', 'Weak': '#3498DB',
                 'Useless': '#BDC3C7', 'Suspicious': '#8E44AD'}
    colors    = [color_map.get(f, '#BDC3C7') for f in top40['flag']]

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.barh(range(len(top40)), top40['iv'], color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(top40)))
    ax.set_yticklabels(top40['feature'], fontsize=8)
    ax.invert_yaxis()
    for x_val in [0.02, 0.10, 0.30]:
        ax.axvline(x_val, color='gray', linestyle='--', alpha=0.6, linewidth=1)
    ax.text(0.02, -1, 'Weak',   fontsize=7, color='gray')
    ax.text(0.10, -1, 'Medium', fontsize=7, color='gray')
    ax.text(0.30, -1, 'Strong', fontsize=7, color='gray')
    ax.set_xlabel('Information Value (IV)', fontsize=11)
    ax.set_title('Feature IV Ranking — Basel II Scorecard Selection\n'
                 'Red=Strong | Orange=Medium | Blue=Weak | Gray=Useless', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/iv_ranking.png', dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  Saved: reports/iv_ranking.png")
    return iv_df


def build_woe_transforms(train_df: pd.DataFrame, iv_df: pd.DataFrame,
                          features_to_keep: list,
                          target_col: str = 'TARGET') -> dict:
    """Fit WoE maps trên training data. IMPORTANT: fit chỉ trên train (no leakage)."""
    target   = train_df[target_col]
    woe_maps = {}
    for feat in features_to_keep:
        result            = compute_woe_iv(train_df[feat], target)
        woe_maps[feat]    = result['woe_map']

    woe_maps_serialisable = {
        feat: {str(k): float(v) for k, v in mp.items()}
        for feat, mp in woe_maps.items()
    }
    with open(f'{DATA_DIR}/woe_maps.json', 'w') as f:
        json.dump(woe_maps_serialisable, f, indent=2)
    log(f"  Saved WoE maps: data/woe_maps.json ({len(woe_maps)} features)")
    return woe_maps


def apply_woe_transforms(df: pd.DataFrame, woe_maps: dict,
                          suffix: str = '_WOE') -> pd.DataFrame:
    """Apply WoE transformation. Unseen bins imputed với 0 (neutral WoE)."""
    df_out = df.copy()
    for feat, woe_map in woe_maps.items():
        if feat not in df_out.columns:
            continue
        col = df_out[feat]
        if col.dtype in [object, bool] or col.nunique() <= 10:
            bin_series = col.astype(str)
        else:
            try:
                bin_series = pd.qcut(col, q=10, duplicates='drop').astype(str)
            except Exception:
                bin_series = pd.cut(col, bins=10, duplicates='drop').astype(str)
        df_out[f'{feat}{suffix}'] = bin_series.map(woe_map).fillna(0)
    log(f"  Applied WoE transforms: {len(woe_maps)} new '{suffix}' columns added")
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    t0 = time.time()
    header("Loading train_features.parquet")
    train_df = pd.read_parquet(f'{DATA_DIR}/train_features.parquet')
    test_df  = pd.read_parquet(f'{DATA_DIR}/test_features.parquet')
    FEATURES = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    log(f"Train: {train_df.shape} | Features: {len(FEATURES)}")

    # Step 1: IV screening
    iv_df = run_iv_screening(train_df, FEATURES)

    # Step 2: Select features
    keep_features = iv_df[iv_df['keep'] == True]['feature'].tolist()
    drop_features = iv_df[iv_df['keep'] == False]['feature'].tolist()
    log(f"\n  Features kept after IV screening : {len(keep_features)}")
    log(f"  Features dropped (IV<0.02 or >0.5): {len(drop_features)}")

    # Step 3: Build WoE maps (fit on train only — no leakage)
    woe_maps = build_woe_transforms(train_df, iv_df, keep_features)

    # Step 4: Apply WoE transforms
    keep_in_test = [f for f in keep_features if f in test_df.columns]
    train_woe = apply_woe_transforms(
        train_df[keep_features + ['TARGET', 'SK_ID_CURR']], woe_maps)
    test_woe  = apply_woe_transforms(
        test_df[keep_in_test + ['SK_ID_CURR']], woe_maps)

    # Save
    train_woe.to_parquet(f'{DATA_DIR}/train_woe.parquet', index=False)
    test_woe.to_parquet( f'{DATA_DIR}/test_woe.parquet',  index=False)
    log(f"\n  Saved: data/train_woe.parquet  {train_woe.shape}")
    log(f"  Saved: data/test_woe.parquet   {test_woe.shape}")
    log(f"\n  Total time: {time.time()-t0:.0f}s")
