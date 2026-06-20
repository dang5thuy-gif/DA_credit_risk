"""
data_cleaning.py - Professional Data Cleaning Pipeline
Home Credit Default Risk Dataset
Author  : Data Analyst
Purpose : Clean all 8 source files -> save cleaned Parquet files
          ready for modeling (Phase 2) and SQL Server import
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = str(ROOT_DIR / 'data')
OUTPUT_DIR = str(ROOT_DIR / 'data' / 'cleaned')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Logging helper ────────────────────────────────────────────────────────────
def log(msg): print(f"  {msg}")
def header(title):
    print(f"\n{'='*65}")
    print(f"  {title}")
    print('='*65)
def step(msg): print(f"\n  >> {msg}")

# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — Summarise cleaning result
# ─────────────────────────────────────────────────────────────────────────────
def cleaning_summary(df_before, df_after, label):
    rows_before = df_before.shape[0]
    cols_before = df_before.shape[1]
    rows_after  = df_after.shape[0]
    cols_after  = df_after.shape[1]
    miss_before = df_before.isnull().mean().mean() * 100
    miss_after  = df_after.isnull().mean().mean() * 100
    log(f"  Rows : {rows_before:,} -> {rows_after:,}")
    log(f"  Cols : {cols_before} -> {cols_after}  "
        f"({'dropped ' + str(cols_before - cols_after) if cols_before > cols_after else 'unchanged'})")
    log(f"  Avg missing rate: {miss_before:.1f}% -> {miss_after:.1f}%")

def save(df, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.parquet")
    df.to_parquet(path, index=False)
    size_mb = os.path.getsize(path) / 1e6
    log(f"  Saved: {path}  ({size_mb:.1f} MB)")


# =============================================================================
# 1. APPLICATION_TRAIN & TEST
#    - Source of truth, 307,511 applicants, 122 columns
# =============================================================================
header("1. application_train.csv  +  application_test.csv")
t0 = time.time()

for split in ['train', 'test']:
    step(f"Loading application_{split}.csv ...")
    df = pd.read_csv(os.path.join(DATA_DIR, f'application_{split}.csv'))
    df_orig = df.copy()
    log(f"Shape: {df.shape}")

    # ── 1a. Anomaly: DAYS_EMPLOYED = 365243 (unemployed encoding) ─────────────
    step("Fix DAYS_EMPLOYED anomaly")
    n_anom = (df['DAYS_EMPLOYED'] == 365243).sum()
    df['EMP_ANOMALY'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    log(f"EMP_ANOMALY flagged: {n_anom:,} rows ({n_anom/len(df)*100:.1f}%)")

    # ── 1b. Anomaly: CODE_GENDER = 'XNA' (only 4 rows) ───────────────────────
    step("Fix CODE_GENDER XNA")
    n_xna = (df['CODE_GENDER'] == 'XNA').sum()
    df['CODE_GENDER'] = df['CODE_GENDER'].replace('XNA', 'Unknown')
    log(f"Replaced CODE_GENDER 'XNA': {n_xna} rows")

    # ── 1c. Drop columns with >40% missing ────────────────────────────────────
    step("Drop high-missing columns (>40%)")
    miss_rate = df.isnull().mean()
    drop_cols = miss_rate[miss_rate > 0.40].index.tolist()
    # Exception: Keep EXT_SOURCE even if missing - they are strongest predictors
    keep_always = ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']
    drop_cols   = [c for c in drop_cols if c not in keep_always]
    df.drop(columns=drop_cols, inplace=True)
    log(f"Dropped {len(drop_cols)} columns: {drop_cols[:5]}... (and {len(drop_cols)-5} more)")

    # ── 1d. Impute EXT_SOURCE with median (NEVER with 0) ──────────────────────
    step("Impute EXT_SOURCE_1/2/3 with median")
    for col in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
        if col in df.columns:
            med = df[col].median()
            n_filled = df[col].isnull().sum()
            df[col] = df[col].fillna(med)
            log(f"  {col}: filled {n_filled:,} NaN with median={med:.4f}")

    # ── 1e. Explicitly impute DAYS_EMPLOYED (was set to NaN in step 1a) ────────
    step("Impute DAYS_EMPLOYED NaN (unemployed clients) with median")
    med_emp = df['DAYS_EMPLOYED'].median()
    n_emp_nan = df['DAYS_EMPLOYED'].isnull().sum()
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].fillna(med_emp)
    log(f"  Filled {n_emp_nan:,} NaN with median={med_emp:.0f} days ({-med_emp/365:.1f} yrs)")

    # ── 1f. Cap outliers ───────────────────────────────────────────────────────
    step("Cap outliers")
    # AMT_INCOME_TOTAL: cap at 99th percentile
    p99_income = df['AMT_INCOME_TOTAL'].quantile(0.99)
    n_capped   = (df['AMT_INCOME_TOTAL'] > p99_income).sum()
    df['AMT_INCOME_TOTAL'] = df['AMT_INCOME_TOTAL'].clip(upper=p99_income)
    log(f"  AMT_INCOME_TOTAL: capped {n_capped:,} rows at {p99_income:,.0f}")

    # AMT_REQ_CREDIT_BUREAU_QRT: max sensible = 12 (once a month), clip at 12
    if 'AMT_REQ_CREDIT_BUREAU_QRT' in df.columns:
        n_c = (df['AMT_REQ_CREDIT_BUREAU_QRT'] > 12).sum()
        df['AMT_REQ_CREDIT_BUREAU_QRT'] = df['AMT_REQ_CREDIT_BUREAU_QRT'].clip(upper=12)
        log(f"  AMT_REQ_CREDIT_BUREAU_QRT: capped {n_c:,} rows at 12")

    # Social circle: DEF_30/60 cap at 20, OBS cap at 100
    for col, cap_val in [('DEF_30_CNT_SOCIAL_CIRCLE', 20),
                         ('DEF_60_CNT_SOCIAL_CIRCLE', 20),
                         ('OBS_30_CNT_SOCIAL_CIRCLE', 100),
                         ('OBS_60_CNT_SOCIAL_CIRCLE', 100)]:
        if col in df.columns:
            n_c = (df[col] > cap_val).sum()
            df[col] = df[col].clip(upper=cap_val)
            if n_c: log(f"  {col}: capped {n_c:,} rows at {cap_val}")

    # ── 1g. Impute remaining categoricals with mode ────────────────────────────
    step("Impute remaining categoricals (mode)")
    cat_cols_with_missing = [c for c in df.select_dtypes('object').columns
                              if df[c].isnull().any()]
    for col in cat_cols_with_missing:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        log(f"  {col}: filled NaN with mode='{mode_val}'")

    # ── 1h. Impute remaining numerical with median ─────────────────────────────
    step("Impute remaining numericals (median)")
    num_cols_missing = [c for c in df.select_dtypes(include=np.number).columns
                        if df[c].isnull().any()]
    for col in num_cols_missing:
        med = df[col].median()
        df[col].fillna(med, inplace=True)
    log(f"  Filled {len(num_cols_missing)} numeric columns with median")

    # ── 1i. Validate no missing left ──────────────────────────────────────────
    remaining_miss = df.isnull().sum().sum()
    log(f"  Remaining NaN after all steps: {remaining_miss}")
    if remaining_miss > 0:
        still_missing = df.isnull().sum()
        still_missing = still_missing[still_missing > 0]
        log(f"  Columns still missing: {still_missing.to_dict()}")
        # Force-fill any stragglers with median/mode
        for col in still_missing.index:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode_vals = df[col].mode()
                df[col] = df[col].fillna(mode_vals[0] if len(mode_vals) > 0 else 'Unknown')
        remaining_miss = df.isnull().sum().sum()
        log(f"  Remaining NaN after force-fill: {remaining_miss}")
    assert remaining_miss == 0, f"Still has {remaining_miss} missing values after all steps!"

    # ── Summary & Save ─────────────────────────────────────────────────────────
    step(f"Summary for application_{split}")
    cleaning_summary(df_orig, df, split)
    save(df, f'application_{split}_clean')

log(f"\nDone in {time.time()-t0:.1f}s")


# =============================================================================
# 2. BUREAU.CSV
#    - Credit history from other institutions (1.7M rows, 17 cols)
# =============================================================================
header("2. bureau.csv")
t0 = time.time()

step("Loading...")
bureau = pd.read_csv(os.path.join(DATA_DIR, 'bureau.csv'))
bureau_orig = bureau.copy()
log(f"Shape: {bureau.shape}")

# ── 2a. Drop AMT_ANNUITY (71.5% missing, not useful for aggregation) ─────────
step("Drop AMT_ANNUITY (71.5% missing in bureau)")
bureau.drop(columns=['AMT_ANNUITY'], inplace=True)

# ── 2b. Impute AMT_CREDIT_MAX_OVERDUE with 0 ──────────────────────────────────
#    Missing = client never had any overdue, not "data unknown"
step("Impute AMT_CREDIT_MAX_OVERDUE -> 0 (no overdue history = 0)")
n_fill = bureau['AMT_CREDIT_MAX_OVERDUE'].isnull().sum()
bureau['AMT_CREDIT_MAX_OVERDUE'] = bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(0)
log(f"  Filled {n_fill:,} NaN with 0")

# ── 2c. Cap CREDIT_DAY_OVERDUE at 365 ─────────────────────────────────────────
#    Max was 2792 days (7.6 years). Cap at 365 for stability.
step("Cap CREDIT_DAY_OVERDUE at 365 days")
n_capped = (bureau['CREDIT_DAY_OVERDUE'] > 365).sum()
bureau['CREDIT_DAY_OVERDUE'] = bureau['CREDIT_DAY_OVERDUE'].clip(upper=365)
log(f"  Capped {n_capped:,} rows")

# ── 2d. AMT_CREDIT_SUM = 0: replace with NaN then impute median ───────────────
step("Fix AMT_CREDIT_SUM == 0 (66,582 rows likely data entry error)")
n_zero = (bureau['AMT_CREDIT_SUM'] == 0).sum()
bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].replace(0, np.nan)
med_credit = bureau['AMT_CREDIT_SUM'].median()
bureau['AMT_CREDIT_SUM'] = bureau['AMT_CREDIT_SUM'].fillna(med_credit)
log(f"  Replaced {n_zero:,} zeros with median={med_credit:,.0f}")

# ── 2e. Impute remaining missing with median/0 ─────────────────────────────────
step("Impute remaining missing")
# DAYS_CREDIT_ENDDATE: future dates OK, missing = no end date
# DAYS_ENDDATE_FACT: only for closed loans, missing for active = expected -> 0
bureau['DAYS_ENDDATE_FACT'] = bureau['DAYS_ENDDATE_FACT'].fillna(0)
# AMT_CREDIT_SUM_DEBT / LIMIT / OVERDUE: missing = 0 (no debt recorded)
for col in ['AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE']:
    bureau[col] = bureau[col].fillna(0)
# DAYS_CREDIT_ENDDATE: impute median
bureau['DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_ENDDATE'].fillna(
    bureau['DAYS_CREDIT_ENDDATE'].median())

remaining_miss = bureau.isnull().sum().sum()
log(f"  Remaining NaN: {remaining_miss}")

step("Summary")
cleaning_summary(bureau_orig, bureau, 'bureau')
save(bureau, 'bureau_clean')
log(f"Done in {time.time()-t0:.1f}s")
del bureau, bureau_orig


# =============================================================================
# 3. PREVIOUS_APPLICATION.CSV
#    - Loan application history (1.67M rows, 37 cols)
# =============================================================================
header("3. previous_application.csv")
t0 = time.time()

step("Loading...")
prev = pd.read_csv(os.path.join(DATA_DIR, 'previous_application.csv'))
prev_orig = prev.copy()
log(f"Shape: {prev.shape}")

# ── 3a. Drop near-useless columns (>99% missing) ──────────────────────────────
step("Drop RATE_INTEREST_PRIMARY + RATE_INTEREST_PRIVILEGED (99.6% missing)")
prev.drop(columns=['RATE_INTEREST_PRIMARY', 'RATE_INTEREST_PRIVILEGED'], inplace=True)

# ── 3b. Anomaly: DAYS_FIRST_DRAWING = 365243 ──────────────────────────────────
step("Fix DAYS_FIRST_DRAWING anomaly (same as DAYS_EMPLOYED)")
days_cols_with_anomaly = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE',
                           'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE',
                           'DAYS_TERMINATION']
for col in days_cols_with_anomaly:
    if col in prev.columns:
        n_anom = (prev[col] == 365243).sum()
        prev[col] = prev[col].replace(365243, np.nan)
        if n_anom: log(f"  {col}: replaced {n_anom:,} anomaly rows with NaN")

# ── 3c. Fix AMT_DOWN_PAYMENT: negative -> 0 ───────────────────────────────────
step("Fix AMT_DOWN_PAYMENT negatives -> 0")
if 'AMT_DOWN_PAYMENT' in prev.columns:
    n_neg = (prev['AMT_DOWN_PAYMENT'] < 0).sum()
    prev['AMT_DOWN_PAYMENT'] = prev['AMT_DOWN_PAYMENT'].clip(lower=0)
    prev['AMT_DOWN_PAYMENT'] = prev['AMT_DOWN_PAYMENT'].fillna(0)
    log(f"  Clipped {n_neg} negative, filled NaN with 0")

# ── 3d. Impute remaining DAYS_ columns with median ────────────────────────────
step("Impute remaining DAYS_ columns with median")
days_cols = [c for c in prev.columns if 'DAYS_' in c and prev[c].isnull().any()]
for col in days_cols:
    med = prev[col].median()
    n_fill = prev[col].isnull().sum()
    prev[col] = prev[col].fillna(med)
    log(f"  {col}: filled {n_fill:,} NaN with median={med:.1f}")

# ── 3e. Impute numericals ──────────────────────────────────────────────────────
step("Impute remaining numericals (median), categoricals (mode)")
for col in prev.select_dtypes(include=np.number).columns:
    if prev[col].isnull().any():
        prev[col] = prev[col].fillna(prev[col].median())

for col in prev.select_dtypes('object').columns:
    if prev[col].isnull().any():
        prev[col] = prev[col].fillna(prev[col].mode()[0])

# ── 3f. RATE_DOWN_PAYMENT: fill 0 then clip ──────────────────────────────────
if 'RATE_DOWN_PAYMENT' in prev.columns:
    prev['RATE_DOWN_PAYMENT'] = prev['RATE_DOWN_PAYMENT'].fillna(0).clip(0, 1)

remaining_miss = prev.isnull().sum().sum()
log(f"  Remaining NaN: {remaining_miss}")

step("Summary")
cleaning_summary(prev_orig, prev, 'previous_application')
save(prev, 'previous_application_clean')
log(f"Done in {time.time()-t0:.1f}s")
del prev, prev_orig


# =============================================================================
# 4. INSTALLMENTS_PAYMENTS.CSV
#    - Payment records (13.6M rows, 8 cols) — memory-efficient load
# =============================================================================
header("4. installments_payments.csv  (13.6M rows)")
t0 = time.time()

step("Loading with usecols + dtype optimization...")
inst = pd.read_csv(
    os.path.join(DATA_DIR, 'installments_payments.csv'),
    dtype={
        'SK_ID_CURR': 'int32', 'SK_ID_PREV': 'int32',
        'AMT_INSTALMENT': 'float32', 'AMT_PAYMENT': 'float32',
    }
)
inst_orig = inst.copy()
log(f"Shape: {inst.shape}  |  RAM: {inst.memory_usage(deep=True).sum()/1e6:.0f} MB")

# ── 4a. Impute the tiny ~0.02% missing in DAYS_ENTRY_PAYMENT + AMT_PAYMENT ───
step("Impute 0.02% missing (DAYS_ENTRY_PAYMENT, AMT_PAYMENT)")
inst['DAYS_ENTRY_PAYMENT'] = inst['DAYS_ENTRY_PAYMENT'].fillna(
    inst['DAYS_ENTRY_PAYMENT'].median())
inst['AMT_PAYMENT'] = inst['AMT_PAYMENT'].fillna(inst['AMT_PAYMENT'].median())

# ── 4b. Engineer PAYMENT_DIFF + DAYS_PAST_DUE (core features) ────────────────
step("Engineer PAYMENT_DIFF and DAYS_PAST_DUE")
inst['PAYMENT_DIFF'] = (inst['AMT_INSTALMENT'] - inst['AMT_PAYMENT']).astype('float32')
# Negative DAYS_PAST_DUE = paid early -> treat as 0 (no lateness)
raw_dpd = (inst['DAYS_ENTRY_PAYMENT'] - inst['DAYS_INSTALMENT']).astype('float32')
inst['DAYS_PAST_DUE'] = raw_dpd.clip(lower=0)
log(f"  Late payments: {(inst['DAYS_PAST_DUE'] > 0).sum():,} "
    f"({(inst['DAYS_PAST_DUE'] > 0).mean()*100:.1f}%)")

remaining_miss = inst.isnull().sum().sum()
log(f"  Remaining NaN: {remaining_miss}")

step("Summary")
cleaning_summary(inst_orig, inst, 'installments_payments')
save(inst, 'installments_payments_clean')
log(f"Done in {time.time()-t0:.1f}s")
del inst, inst_orig


# =============================================================================
# 5. POS_CASH_BALANCE.CSV
#    - POS and cash loan monthly snapshots (10M rows, 8 cols)
# =============================================================================
header("5. POS_CASH_balance.csv  (10M rows)")
t0 = time.time()

step("Loading with dtype optimization...")
pos = pd.read_csv(
    os.path.join(DATA_DIR, 'POS_CASH_balance.csv'),
    dtype={
        'SK_ID_CURR': 'int32', 'SK_ID_PREV': 'int32',
        'MONTHS_BALANCE': 'int16',
        'CNT_INSTALMENT': 'float32', 'CNT_INSTALMENT_FUTURE': 'float32',
        'SK_DPD': 'int16', 'SK_DPD_DEF': 'int16',
    }
)
pos_orig = pos.copy()
log(f"Shape: {pos.shape}  |  RAM: {pos.memory_usage(deep=True).sum()/1e6:.0f} MB")

# ── 5a. Cap SK_DPD at 365 (max was 4,231 — extreme outlier) ──────────────────
step("Cap SK_DPD at 365 days (max was 4,231)")
n_cap = (pos['SK_DPD'] > 365).sum()
pos['SK_DPD'] = pos['SK_DPD'].clip(upper=365).astype('int16')
n_cap2 = (pos['SK_DPD_DEF'] > 365).sum()
pos['SK_DPD_DEF'] = pos['SK_DPD_DEF'].clip(upper=365).astype('int16')
log(f"  SK_DPD: capped {n_cap:,} rows  |  SK_DPD_DEF: capped {n_cap2:,} rows")

# ── 5b. Impute CNT_INSTALMENT (0.26% missing) ─────────────────────────────────
step("Impute CNT_INSTALMENT (0.26% missing) with median")
for col in ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']:
    med = pos[col].median()
    pos[col] = pos[col].fillna(med)

# ── 5c. Map NAME_CONTRACT_STATUS 'XNA' (2 rows) to 'Unknown' ─────────────────
step("Fix NAME_CONTRACT_STATUS 'XNA' -> 'Unknown'")
n_xna = (pos['NAME_CONTRACT_STATUS'] == 'XNA').sum()
pos['NAME_CONTRACT_STATUS'] = pos['NAME_CONTRACT_STATUS'].replace('XNA', 'Unknown')
log(f"  Replaced {n_xna} XNA values")

remaining_miss = pos.isnull().sum().sum()
log(f"  Remaining NaN: {remaining_miss}")

step("Summary")
cleaning_summary(pos_orig, pos, 'POS_CASH_balance')
save(pos, 'POS_CASH_balance_clean')
log(f"Done in {time.time()-t0:.1f}s")
del pos, pos_orig


# =============================================================================
# 6. CREDIT_CARD_BALANCE.CSV
#    - Credit card monthly snapshots (3.84M rows, 23 cols)
# =============================================================================
header("6. credit_card_balance.csv  (3.84M rows)")
t0 = time.time()

step("Loading with dtype optimization...")
cc = pd.read_csv(
    os.path.join(DATA_DIR, 'credit_card_balance.csv'),
    dtype={
        'SK_ID_CURR': 'int32', 'SK_ID_PREV': 'int32',
        'MONTHS_BALANCE': 'int16',
        'SK_DPD': 'int16', 'SK_DPD_DEF': 'int16',
    }
)
cc_orig = cc.copy()
log(f"Shape: {cc.shape}  |  RAM: {cc.memory_usage(deep=True).sum()/1e6:.0f} MB")

# ── 6a. Impute missing drawing/payment columns with 0 ─────────────────────────
#    Missing ~19-20%: no drawings made in that month = 0 activity
step("Impute drawing/payment columns with 0 (no activity = 0)")
fill_zero_cols = [
    'AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT',
    'AMT_DRAWINGS_OTHER_CURRENT', 'AMT_DRAWINGS_POS_CURRENT',
    'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_OTHER_CURRENT',
    'CNT_DRAWINGS_POS_CURRENT', 'AMT_PAYMENT_CURRENT',
    'AMT_INST_MIN_REGULARITY', 'CNT_INSTALMENT_MATURE_CUM',
]
for col in fill_zero_cols:
    if col in cc.columns:
        cc[col] = cc[col].fillna(0)
log(f"  Filled {len(fill_zero_cols)} columns with 0")

# ── 6b. Cap SK_DPD at 365 ─────────────────────────────────────────────────────
step("Cap SK_DPD at 365 (max was 3,260)")
n_cap = (cc['SK_DPD'] > 365).sum()
cc['SK_DPD'] = cc['SK_DPD'].clip(upper=365).astype('int16')
n_cap2 = (cc['SK_DPD_DEF'] > 365).sum()
cc['SK_DPD_DEF'] = cc['SK_DPD_DEF'].clip(upper=365).astype('int16')
log(f"  SK_DPD: capped {n_cap:,} rows  |  SK_DPD_DEF: capped {n_cap2:,} rows")

# ── 6c. AMT_BALANCE < 0: valid (credit balance), clip at 0 for agg ────────────
#    Keep raw for SQL (valid business data), but note for feature engineering
step("Note AMT_BALANCE negatives (valid — overpayment/credit)")
n_neg = (cc['AMT_BALANCE'] < 0).sum()
log(f"  Negative AMT_BALANCE: {n_neg:,} rows — kept as-is (valid)")

# ── 6d. Compute safe utilization ratio (filter limit = 0 first) ───────────────
step("Compute UTILIZATION_RATIO safely (filter AMT_CREDIT_LIMIT_ACTUAL=0)")
n_zero_limit = (cc['AMT_CREDIT_LIMIT_ACTUAL'] == 0).sum()
log(f"  Rows with credit limit = 0: {n_zero_limit:,} -> utilization set to NaN")
cc['UTILIZATION_RATIO'] = np.where(
    cc['AMT_CREDIT_LIMIT_ACTUAL'] > 0,
    cc['AMT_BALANCE'] / cc['AMT_CREDIT_LIMIT_ACTUAL'],
    np.nan
)
# Cap at 2.0 (200% over-limit is extreme enough)
n_over = (cc['UTILIZATION_RATIO'] > 2.0).sum()
cc['UTILIZATION_RATIO'] = cc['UTILIZATION_RATIO'].clip(upper=2.0)
cc['UTILIZATION_RATIO'] = cc['UTILIZATION_RATIO'].fillna(0)
log(f"  Capped utilization > 2.0: {n_over:,} rows")
log(f"  Final utilization — mean={cc['UTILIZATION_RATIO'].mean():.3f}  "
    f"max={cc['UTILIZATION_RATIO'].max():.3f}")

remaining_miss = cc.isnull().sum().sum()
log(f"  Remaining NaN: {remaining_miss}")

step("Summary")
cleaning_summary(cc_orig, cc, 'credit_card_balance')
save(cc, 'credit_card_balance_clean')
log(f"Done in {time.time()-t0:.1f}s")
del cc, cc_orig


# =============================================================================
# FINAL REPORT
# =============================================================================
header("CLEANING COMPLETE — Output Files")
for f in os.listdir(OUTPUT_DIR):
    path = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(path) / 1e6
    print(f"  {f:<45} {size:.1f} MB")

print(f"""
All cleaned files saved to: {OUTPUT_DIR}/
- Format: Parquet (columnar, fast read, type-safe)
- Ready for: Phase 2 Feature Engineering + SQL Server import
""")
