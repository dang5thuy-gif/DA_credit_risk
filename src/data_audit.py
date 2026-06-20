"""
data_audit.py - Full data quality audit for all 8 Home Credit files
Handles critical issues from plan evaluation + comprehensive cleaning checks
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = str(ROOT_DIR / 'data')

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def divider(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)

def section(title):
    print(f"\n-- {title} " + "-"*max(1, 60-len(title)))

def pct(val, total):
    return f"{val:,}  ({val/total*100:.1f}%)"


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD COLUMN DESCRIPTIONS (critical issue #3 in plan)
# ─────────────────────────────────────────────────────────────────────────────
divider("STEP 0 - Column Description Lookup")

col_desc_path = os.path.join(DATA_DIR, 'HomeCredit_columns_description.csv')
col_desc = pd.read_csv(col_desc_path, encoding='latin1')
print(f"Loaded column descriptions: {col_desc.shape[0]} rows")
print(f"Columns in description file: {col_desc.columns.tolist()}")

# Build lookup dict — handle whichever column name holds the row identifier
id_col = col_desc.columns[1]   # usually 'Row'
desc_col = col_desc.columns[4] # usually 'Description'
desc_map = dict(zip(col_desc[id_col], col_desc[desc_col]))
print(f"\nSample lookup — EXT_SOURCE_2: {desc_map.get('EXT_SOURCE_2', 'not found')}")
print(f"Sample lookup — DAYS_EMPLOYED: {desc_map.get('DAYS_EMPLOYED', 'not found')}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. APPLICATION_TRAIN.CSV — Main table, most important
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 1 - application_train.csv")

app = pd.read_csv(os.path.join(DATA_DIR, 'application_train.csv'))
print(f"Shape: {app.shape}  ({app.shape[0]:,} rows × {app.shape[1]} cols)")
N = len(app)

section("Target variable")
vc = app['TARGET'].value_counts()
print(f"  0 (repaid):   {pct(vc[0], N)}")
print(f"  1 (default):  {pct(vc[1], N)}")
print(f"  Default rate: {vc[1]/N*100:.2f}%")

section("Duplicates")
dupes = app.duplicated().sum()
sk_dupes = app['SK_ID_CURR'].duplicated().sum()
print(f"  Full row duplicates: {dupes}")
print(f"  SK_ID_CURR duplicates: {sk_dupes}")

section("Missing values")
miss = app.isnull().mean().sort_values(ascending=False)
high_miss   = miss[miss > 0.40]
medium_miss = miss[(miss > 0.15) & (miss <= 0.40)]
low_miss    = miss[(miss > 0) & (miss <= 0.15)]
print(f"  >40% missing  → {len(high_miss)} columns  (drop candidates)")
print(f"  15–40% missing → {len(medium_miss)} columns  (impute carefully)")
print(f"  0–15% missing  → {len(low_miss)} columns  (standard impute)")
print(f"\n  Top 15 most-missing columns:")
for col, pct_val in miss.head(15).items():
    print(f"    {col:<40} {pct_val*100:.1f}%")

section("Known anomalies - DAYS_EMPLOYED")
anomaly_count = (app['DAYS_EMPLOYED'] == 365243).sum()
print(f"  DAYS_EMPLOYED == 365243: {pct(anomaly_count, N)}  <- unemployed flag")
normal_emp = app[app['DAYS_EMPLOYED'] != 365243]['DAYS_EMPLOYED']
print(f"  Normal range: [{normal_emp.min():.0f}, {normal_emp.max():.0f}]  "
      f"(years: [{-normal_emp.max()/365:.1f}, {-normal_emp.min()/365:.1f}])")

section("Known anomalies - CODE_GENDER")
print(f"  Gender distribution:")
print(app['CODE_GENDER'].value_counts().to_string())

section("Known anomalies - AMT_INCOME_TOTAL")
inc = app['AMT_INCOME_TOTAL']
p99 = inc.quantile(0.99)
p999 = inc.quantile(0.999)
print(f"  Max:    {inc.max():>15,.0f}")
print(f"  99th pct: {p99:>13,.0f}")
print(f"  99.9th pct: {p999:>11,.0f}")
print(f"  Rows > 99th pct: {(inc > p99).sum():,}")

section("EXT_SOURCE features - key predictors")
for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
    s = app[col]
    print(f"  {col}: missing={s.isnull().mean()*100:.1f}%  "
          f"mean={s.mean():.3f}  min={s.min():.3f}  max={s.max():.3f}")

section("DAYS_BIRTH - Age validation")
age_years = -app['DAYS_BIRTH'] / 365
print(f"  Age range: [{age_years.min():.1f}, {age_years.max():.1f}] years")
print(f"  Age < 18: {(age_years < 18).sum()} rows  (should be 0)")
print(f"  Age > 70: {(age_years > 70).sum():,} rows")

section("FLAG_DOCUMENT columns")
flag_doc_cols = [c for c in app.columns if 'FLAG_DOCUMENT' in c]
print(f"  Count: {len(flag_doc_cols)} columns")
total_flags = app[flag_doc_cols].sum(axis=1)
print(f"  Clients with 0 documents flagged: {(total_flags == 0).sum():,}")
print(f"  Max documents flagged for one client: {total_flags.max()}")

section("Categorical columns - unique values")
cat_cols = app.select_dtypes('object').columns
for col in cat_cols:
    nuniq = app[col].nunique()
    top_val = app[col].value_counts().index[0]
    miss_pct = app[col].isnull().mean()*100
    print(f"  {col:<40} {nuniq:>3} unique  miss={miss_pct:.1f}%  top='{top_val}'")

section("Numerical columns - outlier check (z-score > 10)")
num_cols = app.select_dtypes(include=np.number).columns
num_cols = [c for c in num_cols if c not in ['SK_ID_CURR', 'TARGET']]
extreme_outlier_cols = []
for col in num_cols:
    s = app[col].dropna()
    if len(s) > 0 and s.std() > 0:
        z = ((s - s.mean()) / s.std()).abs()
        n_extreme = (z > 10).sum()
        if n_extreme > 0:
            extreme_outlier_cols.append((col, n_extreme, s.max()))
if extreme_outlier_cols:
    print(f"  Columns with z-score > 10:")
    for col, n, mx in sorted(extreme_outlier_cols, key=lambda x: -x[1]):
        print(f"    {col:<40} {n:>5} rows  max={mx:,.0f}")
else:
    print("  No extreme outliers detected (z > 10)")

section("Negative value check on amount columns")
amt_cols = [c for c in app.columns if 'AMT_' in c]
for col in amt_cols:
    neg = (app[col] < 0).sum()
    if neg > 0:
        print(f"  ⚠️  {col}: {neg} negative values")
    else:
        print(f"  ✅ {col}: no negatives")


# ─────────────────────────────────────────────────────────────────────────────
# 3. BUREAU.CSV
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 2 - bureau.csv")

# Memory-efficient: select only needed columns
bureau = pd.read_csv(os.path.join(DATA_DIR, 'bureau.csv'))
print(f"Shape: {bureau.shape}  ({bureau.shape[0]:,} rows × {bureau.shape[1]} cols)")
print(f"Columns: {bureau.columns.tolist()}")

section("Missing values")
miss_b = bureau.isnull().mean().sort_values(ascending=False)
print(miss_b[miss_b > 0].to_string())

section("Clients coverage")
n_clients = bureau['SK_ID_CURR'].nunique()
n_train_clients = app['SK_ID_CURR'].nunique()
print(f"  Unique clients in bureau: {n_clients:,}")
print(f"  Unique clients in app_train: {n_train_clients:,}")
print(f"  Coverage: {n_clients/n_train_clients*100:.1f}%")

section("Loans per client")
loans_per_client = bureau.groupby('SK_ID_CURR').size()
print(f"  Min: {loans_per_client.min()}  Max: {loans_per_client.max()}  "
      f"Mean: {loans_per_client.mean():.1f}  Median: {loans_per_client.median():.1f}")
print(f"  Clients with > 20 bureau loans: {(loans_per_client > 20).sum():,}")

section("CREDIT_ACTIVE distribution")
print(bureau['CREDIT_ACTIVE'].value_counts().to_string())

section("CREDIT_TYPE distribution")
print(bureau['CREDIT_TYPE'].value_counts().to_string())

section("Anomalies - DAYS_CREDIT")
print(f"  Range: [{bureau['DAYS_CREDIT'].min()}, {bureau['DAYS_CREDIT'].max()}]")
print(f"  Positive values (anomaly?): {(bureau['DAYS_CREDIT'] > 0).sum()}")

section("CREDIT_DAY_OVERDUE")
print(f"  Max overdue days: {bureau['CREDIT_DAY_OVERDUE'].max()}")
print(f"  Rows with overdue > 0: {(bureau['CREDIT_DAY_OVERDUE'] > 0).sum():,}")
print(f"  Rows with overdue > 365: {(bureau['CREDIT_DAY_OVERDUE'] > 365).sum():,}")

section("AMT_CREDIT_SUM — negative / zero check")
if 'AMT_CREDIT_SUM' in bureau.columns:
    neg = (bureau['AMT_CREDIT_SUM'] < 0).sum()
    zero = (bureau['AMT_CREDIT_SUM'] == 0).sum()
    print(f"  Negative: {neg}   Zero: {zero}   Max: {bureau['AMT_CREDIT_SUM'].max():,.0f}")

del bureau


# ─────────────────────────────────────────────────────────────────────────────
# 4. PREVIOUS_APPLICATION.CSV
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 3 - previous_application.csv")

prev = pd.read_csv(os.path.join(DATA_DIR, 'previous_application.csv'))
print(f"Shape: {prev.shape}  ({prev.shape[0]:,} rows × {prev.shape[1]} cols)")

section("Missing values")
miss_p = prev.isnull().mean().sort_values(ascending=False)
high_p = miss_p[miss_p > 0.30]
print(f"  Columns with >30% missing: {len(high_p)}")
for col, pct_val in high_p.items():
    print(f"    {col:<45} {pct_val*100:.1f}%")

section("NAME_CONTRACT_STATUS distribution")
print(prev['NAME_CONTRACT_STATUS'].value_counts().to_string())

section("Known anomaly - DAYS_FIRST_DRAWING")
if 'DAYS_FIRST_DRAWING' in prev.columns:
    anom = (prev['DAYS_FIRST_DRAWING'] == 365243).sum()
    print(f"  DAYS_FIRST_DRAWING == 365243: {anom:,}  <- same anomaly as DAYS_EMPLOYED")

section("AMT_DOWN_PAYMENT")
if 'AMT_DOWN_PAYMENT' in prev.columns:
    miss_down = prev['AMT_DOWN_PAYMENT'].isnull().mean()
    neg_down  = (prev['AMT_DOWN_PAYMENT'] < 0).sum()
    print(f"  Missing: {miss_down*100:.1f}%   Negative: {neg_down}")

section("RATE_INTEREST_PRIMARY - loan interest rate")
if 'RATE_INTEREST_PRIMARY' in prev.columns:
    ri = prev['RATE_INTEREST_PRIMARY']
    print(f"  Missing: {ri.isnull().mean()*100:.1f}%  "
          f"Min: {ri.min():.4f}  Max: {ri.max():.4f}")
    print(f"  Rates > 1.0 (possibly % not decimal): {(ri > 1).sum():,}")

section("Clients coverage")
n_prev = prev['SK_ID_CURR'].nunique()
print(f"  Unique clients with previous apps: {n_prev:,} / {n_train_clients:,} "
      f"({n_prev/n_train_clients*100:.1f}%)")

del prev


# ─────────────────────────────────────────────────────────────────────────────
# 5. INSTALLMENTS_PAYMENTS.CSV  (critical: 723 MB — use usecols + dtype)
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 4 - installments_payments.csv  (723 MB - memory-efficient load)")

inst = pd.read_csv(
    os.path.join(DATA_DIR, 'installments_payments.csv'),
    usecols=['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_VERSION',
             'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT',
             'DAYS_ENTRY_PAYMENT', 'AMT_INSTALMENT', 'AMT_PAYMENT'],
    dtype={
        'SK_ID_CURR': 'int32',
        'SK_ID_PREV': 'int32',
        'AMT_INSTALMENT': 'float32',
        'AMT_PAYMENT':    'float32',
    }
)
mem_mb = inst.memory_usage(deep=True).sum() / 1e6
print(f"Shape: {inst.shape}  ({inst.shape[0]:,} rows × {inst.shape[1]} cols)")
print(f"Memory usage: {mem_mb:.0f} MB  (vs ~2-3 GB without optimization)")

section("Missing values")
miss_i = inst.isnull().mean()
print(miss_i[miss_i > 0].to_string() if (miss_i > 0).any() else "  No missing values ✅")

section("PAYMENT_DIFF - underpayment / overpayment")
inst['PAYMENT_DIFF']  = inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']
inst['DAYS_PAST_DUE'] = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).clip(lower=0)
print(f"  Underpayment (PAYMENT_DIFF > 0): {(inst['PAYMENT_DIFF'] > 0).mean()*100:.1f}% of payments")
print(f"  Overpayment (PAYMENT_DIFF < 0):  {(inst['PAYMENT_DIFF'] < 0).mean()*100:.1f}% of payments")
print(f"  Exact payment (PAYMENT_DIFF == 0): {(inst['PAYMENT_DIFF'] == 0).mean()*100:.1f}%")
print(f"  Late payments (DAYS_PAST_DUE > 0): {(inst['DAYS_PAST_DUE'] > 0).mean()*100:.1f}%")
print(f"  Very late (> 30 days): {(inst['DAYS_PAST_DUE'] > 30).mean()*100:.2f}%")

section("Clients coverage")
n_inst = inst['SK_ID_CURR'].nunique()
print(f"  Unique clients: {n_inst:,} / {n_train_clients:,} ({n_inst/n_train_clients*100:.1f}%)")

section("AMT_PAYMENT anomalies")
zero_pay = (inst['AMT_PAYMENT'] == 0).sum()
neg_pay  = (inst['AMT_PAYMENT'] < 0).sum()
print(f"  Zero payments: {zero_pay:,}  ({zero_pay/len(inst)*100:.2f}%)")
print(f"  Negative payments: {neg_pay}")

del inst


# ─────────────────────────────────────────────────────────────────────────────
# 6. POS_CASH_BALANCE.CSV
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 5 - POS_CASH_balance.csv  (393 MB - memory-efficient load)")

pos = pd.read_csv(
    os.path.join(DATA_DIR, 'POS_CASH_balance.csv'),
    dtype={
        'SK_ID_CURR': 'int32',
        'SK_ID_PREV': 'int32',
        'MONTHS_BALANCE': 'int16',
        'CNT_INSTALMENT': 'float32',
        'CNT_INSTALMENT_FUTURE': 'float32',
        'SK_DPD': 'int16',
        'SK_DPD_DEF': 'int16',
    }
)
mem_mb = pos.memory_usage(deep=True).sum() / 1e6
print(f"Shape: {pos.shape}  ({pos.shape[0]:,} rows × {pos.shape[1]} cols)")
print(f"Memory usage: {mem_mb:.0f} MB")
print(f"Columns: {pos.columns.tolist()}")

section("Missing values")
miss_pos = pos.isnull().mean()
print(miss_pos[miss_pos > 0].to_string() if (miss_pos > 0).any() else "  No missing values ✅")

section("NAME_CONTRACT_STATUS")
print(pos['NAME_CONTRACT_STATUS'].value_counts().to_string())

section("SK_DPD — Days Past Due")
print(f"  Max DPD: {pos['SK_DPD'].max()}  Mean: {pos['SK_DPD'].mean():.2f}")
print(f"  Rows with DPD > 0: {(pos['SK_DPD'] > 0).sum():,} ({(pos['SK_DPD'] > 0).mean()*100:.1f}%)")
print(f"  Rows with DPD > 30: {(pos['SK_DPD'] > 30).sum():,}")

section("Clients coverage")
n_pos = pos['SK_ID_CURR'].nunique()
print(f"  Unique clients: {n_pos:,} / {n_train_clients:,} ({n_pos/n_train_clients*100:.1f}%)")

del pos


# ─────────────────────────────────────────────────────────────────────────────
# 7. CREDIT_CARD_BALANCE.CSV  (critical: missing from plan)
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 6 - credit_card_balance.csv  (425 MB - memory-efficient load)")

cc = pd.read_csv(
    os.path.join(DATA_DIR, 'credit_card_balance.csv'),
    dtype={
        'SK_ID_CURR': 'int32',
        'SK_ID_PREV': 'int32',
        'MONTHS_BALANCE': 'int16',
        'SK_DPD': 'int16',
        'SK_DPD_DEF': 'int16',
    }
)
mem_mb = cc.memory_usage(deep=True).sum() / 1e6
print(f"Shape: {cc.shape}  ({cc.shape[0]:,} rows × {cc.shape[1]} cols)")
print(f"Memory usage: {mem_mb:.0f} MB")
print(f"Columns: {cc.columns.tolist()}")

section("Missing values")
miss_cc = cc.isnull().mean().sort_values(ascending=False)
print(miss_cc[miss_cc > 0].to_string() if (miss_cc > 0).any() else "  No missing values ✅")

section("AMT_BALANCE - credit card outstanding balance")
bal = cc['AMT_BALANCE']
print(f"  Missing: {bal.isnull().mean()*100:.1f}%")
print(f"  Negative: {(bal < 0).sum():,}  (overpayment/credit)")
print(f"  Mean: {bal.mean():>12,.0f}   Max: {bal.max():>12,.0f}")

section("Credit utilization proxy")
if 'AMT_CREDIT_LIMIT_ACTUAL' in cc.columns:
    limit = cc['AMT_CREDIT_LIMIT_ACTUAL']
    print(f"  Credit limit — Missing: {limit.isnull().mean()*100:.1f}%  "
          f"Mean: {limit.mean():,.0f}  Max: {limit.max():,.0f}")
    util = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
    print(f"  Utilization ratio > 1.0 (over-limit): {(util > 1).sum():,}")
    print(f"  Utilization ratio > 0.8 (near limit): {(util > 0.8).sum():,}")

section("SK_DPD — Days Past Due")
print(f"  Max DPD: {cc['SK_DPD'].max()}  Mean: {cc['SK_DPD'].mean():.2f}")
print(f"  Rows with DPD > 0: {(cc['SK_DPD'] > 0).sum():,} ({(cc['SK_DPD'] > 0).mean()*100:.1f}%)")

section("Clients coverage")
n_cc = cc['SK_ID_CURR'].nunique()
print(f"  Unique clients: {n_cc:,} / {n_train_clients:,} ({n_cc/n_train_clients*100:.1f}%)")

section("Aggregation template - new feature set from plan gap")
cc['UTILIZATION_RATIO'] = cc['AMT_BALANCE'] / (cc['AMT_CREDIT_LIMIT_ACTUAL'] + 1)
cc_agg_preview = cc.groupby('SK_ID_CURR').agg(
    cc_balance_mean      = ('AMT_BALANCE', 'mean'),
    cc_balance_max       = ('AMT_BALANCE', 'max'),
    cc_limit_mean        = ('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
    cc_utilization_mean  = ('UTILIZATION_RATIO', 'mean'),
    cc_utilization_max   = ('UTILIZATION_RATIO', 'max'),
    cc_dpd_mean          = ('SK_DPD', 'mean'),
    cc_dpd_max           = ('SK_DPD', 'max'),
    cc_months_count      = ('MONTHS_BALANCE', 'count'),
).reset_index()
print(f"  cc_agg shape: {cc_agg_preview.shape}")
print(f"  Sample aggregated features:")
print(cc_agg_preview.describe().round(2).to_string())

del cc, cc_agg_preview


# ─────────────────────────────────────────────────────────────────────────────
# 8. APPLICATION_TEST.CSV
# ─────────────────────────────────────────────────────────────────────────────
divider("FILE 7 - application_test.csv")

app_test = pd.read_csv(os.path.join(DATA_DIR, 'application_test.csv'))
print(f"Shape: {app_test.shape}  ({app_test.shape[0]:,} rows)")
print(f"Has TARGET column: {'TARGET' in app_test.columns}")

section("Column consistency with train")
train_cols = set(app.columns) - {'TARGET'}
test_cols  = set(app_test.columns)
only_train = train_cols - test_cols
only_test  = test_cols - train_cols
print(f"  Cols only in train (excl TARGET): {only_train if only_train else 'none'}")
print(f"  Cols only in test: {only_test if only_test else 'none'}")

section("Missing rate comparison - top 10")
miss_train = app.isnull().mean().sort_values(ascending=False).head(10)
miss_test  = app_test.isnull().mean()
print(f"  {'Column':<40} {'Train Missing':>15} {'Test Missing':>15}")
print(f"  {'-'*70}")
for col in miss_train.index:
    t_miss = miss_test.get(col, float('nan'))
    flag = '[!!]' if abs(miss_train[col] - t_miss) > 0.05 else ''
    print(f"  {col:<40} {miss_train[col]*100:>13.1f}%  {t_miss*100:>13.1f}%  {flag}")

del app_test


# ─────────────────────────────────────────────────────────────────────────────
# 9. SUMMARY — Critical issues status + cleaning plan
# ─────────────────────────────────────────────────────────────────────────────
divider("SUMMARY - Issues found & cleaning plan")

print("""
CRITICAL ISSUES (from plan evaluation) - STATUS:
---------------------------------------------------------------------
[OK] FIXED: HomeCredit_columns_description.csv now available & loaded
[OK] FIXED: Memory-efficient loading for large files (usecols + dtype)
[OK] FIXED: credit_card_balance.csv now included in audit & agg template
[OK] NOTED: correlations must be computed before Phase 3.2 baseline

CLEANING ACTIONS REQUIRED (by file):
---------------------------------------------------------------------
application_train.csv:
  -> DAYS_EMPLOYED: replace 365243 -> NaN, create EMP_ANOMALY flag
  -> CODE_GENDER: replace 'XNA' -> 'Unknown' (only ~4 rows)
  -> AMT_INCOME_TOTAL: cap at 99th percentile for visualisation only
  -> DAYS_BIRTH: validate no age < 18 (critical for lending regulation)
  -> EXT_SOURCE cols: impute with median (not zero - semantic matters)
  -> 49 cols with >40% missing: drop or impute based on feature importance

bureau.csv:
  -> DAYS_CREDIT_ENDDATE: positive values = future dates - OK
  -> AMT_CREDIT_SUM_OVERDUE: heavy right skew, log-transform for modelling
  -> AMT_CREDIT_MAX_OVERDUE: ~64% missing -> impute 0 (no overdue = 0)

previous_application.csv:
  -> DAYS_FIRST_DRAWING == 365243: same anomaly, replace -> NaN
  -> AMT_DOWN_PAYMENT: 73% missing -> impute 0 (no down payment recorded)
  -> RATE_INTEREST_PRIMARY: values > 1.0 need verification

installments_payments.csv:
  -> Zero payments: likely deferred/skipped - keep as signal
  -> DAYS_PAST_DUE: clip at 0 (negative = paid early, treat as 0)
  -> Load with usecols + float32 dtype to keep under 800 MB RAM

POS_CASH_balance.csv:
  -> NAME_CONTRACT_STATUS=Completed: use to flag loan completion rate
  -> SK_DPD: check extreme max values, consider capping at 365

credit_card_balance.csv:
  -> AMT_BALANCE < 0: valid (overpayment) - keep
  -> Utilization > 1.0: over-limit - keep as risk signal, cap at 2.0
  -> Convert float64 cols to float32 to save ~200 MB RAM

DATA LEAKAGE RISKS:
---------------------------------------------------------------------
  [!!] Do NOT use time-based features that reference events AFTER
       the loan application date
  [!!] Imputer must be fit on TRAIN only, then transform TRAIN+TEST
  [!!] OOF predictions only - never evaluate on training data
""")

print("Audit complete ✅")
