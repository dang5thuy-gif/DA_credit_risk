"""
feature_engineering.py — Phase 2: Feature Engineering
Home Credit Default Risk — theo PROJECT_PLAN.md Phase 2

Steps:
  2.1  Features from application (ratios, time, flags)
  2.2  Aggregation from bureau
  2.3  Aggregation from previous_application
  2.4  Aggregation from installments_payments
  2.5  Aggregation from POS_CASH_balance
  2.6  Aggregation from credit_card_balance
  2.7  Merge & Validate
  2.8  Encoding & Imputation
  2.11 Interaction Features (Moved up to prevent selection drop errors)
  2.9  Feature Selection (zero-variance + high-correlation)
  2.10 Save train_features.parquet + test_features.parquet
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, time, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
CLEAN_DIR  = str(ROOT_DIR / 'data' / 'cleaned')
OUTPUT_DIR = str(ROOT_DIR / 'data')

def header(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def step(t):   print(f"\n  >> {t}")
def log(t):    print(f"     {t}")


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load all cleaned Parquet files
# ═══════════════════════════════════════════════════════════════
header("STEP 1 — Loading cleaned Parquet files")
t_total = time.time()

step("Loading application train + test ...")
train = pd.read_parquet(os.path.join(CLEAN_DIR, 'application_train_clean.parquet'))
test  = pd.read_parquet(os.path.join(CLEAN_DIR, 'application_test_clean.parquet'))
log(f"train : {train.shape}   test : {test.shape}")

# Gộp train+test để tạo features đồng nhất, sau đó tách ra
train['IS_TRAIN'] = 1
test['IS_TRAIN']  = 0
test['TARGET']    = np.nan
app = pd.concat([train, test], axis=0, ignore_index=True, sort=False)
log(f"combined app : {app.shape}")

step("Loading bureau ...")
bureau = pd.read_parquet(os.path.join(CLEAN_DIR, 'bureau_clean.parquet'))
log(f"bureau : {bureau.shape}")

step("Loading previous_application ...")
prev = pd.read_parquet(os.path.join(CLEAN_DIR, 'previous_application_clean.parquet'))
log(f"previous : {prev.shape}")

step("Loading installments_payments ...")
inst = pd.read_parquet(os.path.join(CLEAN_DIR, 'installments_payments_clean.parquet'))
log(f"installments : {inst.shape}")

step("Loading POS_CASH_balance ...")
pos = pd.read_parquet(os.path.join(CLEAN_DIR, 'POS_CASH_balance_clean.parquet'))
log(f"pos : {pos.shape}")

step("Loading credit_card_balance ...")
cc = pd.read_parquet(os.path.join(CLEAN_DIR, 'credit_card_balance_clean.parquet'))
log(f"credit_card : {cc.shape}")


# ═══════════════════════════════════════════════════════════════
# STEP 2.1 — Features from Application (theo Plan §2.1)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.1 — Application Features (ratios, time, flags)")

step("Financial ratios")
app['CREDIT_INCOME_RATIO']    = app['AMT_CREDIT']      / (app['AMT_INCOME_TOTAL'] + 1e-6)
app['ANNUITY_INCOME_RATIO']   = app['AMT_ANNUITY']     / (app['AMT_INCOME_TOTAL'] + 1e-6)
app['CREDIT_TERM']            = app['AMT_CREDIT']      / (app['AMT_ANNUITY'] + 1e-6)
app['GOODS_CREDIT_RATIO']     = app['AMT_GOODS_PRICE'] / (app['AMT_CREDIT'] + 1e-6)
app['CREDIT_DOWNPAYMENT']     = app['AMT_GOODS_PRICE'] - app['AMT_CREDIT']
log("Created: CREDIT_INCOME_RATIO, ANNUITY_INCOME_RATIO, CREDIT_TERM, GOODS_CREDIT_RATIO, CREDIT_DOWNPAYMENT")

step("Time-based features")
app['AGE_YEARS']               = (-app['DAYS_BIRTH']) / 365
app['YEARS_EMPLOYED']          = (-app['DAYS_EMPLOYED']) / 365
app['DAYS_EMPLOYED_RATIO']     = app['DAYS_EMPLOYED'] / (app['DAYS_BIRTH'] + 1e-6)
app['YEARS_ID_PUBLISH']        = (-app['DAYS_ID_PUBLISH']) / 365
app['YEARS_LAST_PHONE_CHANGE'] = (-app['DAYS_LAST_PHONE_CHANGE']) / 365
log("Created: AGE_YEARS, YEARS_EMPLOYED, DAYS_EMPLOYED_RATIO, YEARS_ID_PUBLISH, YEARS_LAST_PHONE_CHANGE")

step("Document quality indicator")
doc_cols = [c for c in app.columns if 'FLAG_DOCUMENT' in c]
app['FLAG_DOCS_SUM']        = app[doc_cols].sum(axis=1)
app['CNT_CHILDREN_RATIO']   = app['CNT_CHILDREN'] / (app['CNT_FAM_MEMBERS'].clip(lower=1))
log(f"Created: FLAG_DOCS_SUM (from {len(doc_cols)} doc flags), CNT_CHILDREN_RATIO")

log(f"\n  Total app features so far: {app.shape[1]}")


# ═══════════════════════════════════════════════════════════════
# STEP 2.2 — Aggregation from Bureau (theo Plan §2.2)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.2 — Bureau Aggregation")
t0 = time.time()

bureau_agg = bureau.groupby('SK_ID_CURR').agg(
    bureau_loan_count        = ('SK_ID_BUREAU',          'count'),
    bureau_active_count      = ('CREDIT_ACTIVE',         lambda x: (x == 'Active').sum()),
    bureau_closed_count      = ('CREDIT_ACTIVE',         lambda x: (x == 'Closed').sum()),
    bureau_credit_sum        = ('AMT_CREDIT_SUM',        'sum'),
    bureau_credit_mean       = ('AMT_CREDIT_SUM',        'mean'),
    bureau_credit_max        = ('AMT_CREDIT_SUM',        'max'),
    bureau_debt_sum          = ('AMT_CREDIT_SUM_DEBT',   'sum'),
    bureau_overdue_mean      = ('AMT_CREDIT_SUM_OVERDUE','mean'),
    bureau_overdue_max       = ('AMT_CREDIT_SUM_OVERDUE','max'),
    bureau_days_credit_mean  = ('DAYS_CREDIT',           'mean'),
    bureau_days_enddate_max  = ('DAYS_CREDIT_ENDDATE',   'max'),
    bureau_bad_debt_count    = ('CREDIT_DAY_OVERDUE',    lambda x: (x > 0).sum()),
    bureau_overdue_days_max  = ('CREDIT_DAY_OVERDUE',    'max'),
    bureau_prolong_sum       = ('CNT_CREDIT_PROLONG',    'sum'),
).reset_index()

bureau_agg['bureau_bad_debt_flag']     = (bureau_agg['bureau_bad_debt_count'] > 0).astype(int)
bureau_agg['bureau_active_ratio']      = bureau_agg['bureau_active_count'] / (bureau_agg['bureau_loan_count'] + 1e-6)
bureau_agg['bureau_debt_credit_ratio'] = bureau_agg['bureau_debt_sum'] / (bureau_agg['bureau_credit_sum'] + 1e-6)

log(f"bureau_agg shape: {bureau_agg.shape}  ({time.time()-t0:.1f}s)")
del bureau


# ═══════════════════════════════════════════════════════════════
# STEP 2.3 — Aggregation from Previous Application (§2.3)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.3 — Previous Application Aggregation")
t0 = time.time()

prev_agg = prev.groupby('SK_ID_CURR').agg(
    prev_app_count            = ('SK_ID_PREV',              'count'),
    prev_approved_count       = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Approved').sum()),
    prev_refused_count        = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Refused').sum()),
    prev_canceled_count       = ('NAME_CONTRACT_STATUS',    lambda x: (x == 'Canceled').sum()),
    prev_credit_mean          = ('AMT_CREDIT',              'mean'),
    prev_credit_max           = ('AMT_CREDIT',              'max'),
    prev_annuity_mean         = ('AMT_ANNUITY',             'mean'),
    prev_down_payment_mean    = ('AMT_DOWN_PAYMENT',        'mean'),
    prev_days_decision_mean   = ('DAYS_DECISION',           'mean'),
    prev_consumer_count       = ('NAME_CONTRACT_TYPE',      lambda x: (x == 'Consumer loans').sum()),
    prev_cash_count           = ('NAME_CONTRACT_TYPE',      lambda x: (x == 'Cash loans').sum()),
    prev_revolving_count      = ('NAME_CONTRACT_TYPE',      lambda x: (x == 'Revolving loans').sum()),
).reset_index()

prev_agg['prev_refused_ratio']   = prev_agg['prev_refused_count']   / (prev_agg['prev_app_count'] + 1e-6)
prev_agg['prev_approved_ratio']  = prev_agg['prev_approved_count']  / (prev_agg['prev_app_count'] + 1e-6)
prev_agg['prev_canceled_ratio']  = prev_agg['prev_canceled_count']  / (prev_agg['prev_app_count'] + 1e-6)
prev_agg['prev_consumer_ratio']  = prev_agg['prev_consumer_count']  / (prev_agg['prev_app_count'] + 1e-6)

log(f"prev_agg shape: {prev_agg.shape}  ({time.time()-t0:.1f}s)")
del prev


# ═══════════════════════════════════════════════════════════════
# STEP 2.4 — Aggregation from Installments (§2.4)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.4 — Installments Payments Aggregation")
t0 = time.time()

inst_agg = inst.groupby('SK_ID_CURR').agg(
    inst_payment_diff_mean    = ('PAYMENT_DIFF',   'mean'),
    inst_payment_diff_max     = ('PAYMENT_DIFF',   'max'),
    inst_payment_diff_sum     = ('PAYMENT_DIFF',   'sum'),
    inst_days_past_due_mean   = ('DAYS_PAST_DUE',  'mean'),
    inst_days_past_due_max    = ('DAYS_PAST_DUE',  'max'),
    inst_late_payment_count   = ('DAYS_PAST_DUE',  lambda x: (x > 0).sum()),
    inst_on_time_count        = ('DAYS_PAST_DUE',  lambda x: (x == 0).sum()),
    inst_total_count          = ('DAYS_PAST_DUE',  'count'),
    inst_amt_payment_mean     = ('AMT_PAYMENT',    'mean'),
    inst_amt_instalment_mean  = ('AMT_INSTALMENT', 'mean'),
).reset_index()

inst_agg['inst_late_ratio'] = inst_agg['inst_late_payment_count'] / (inst_agg['inst_total_count'] + 1e-6)

log(f"inst_agg shape: {inst_agg.shape}  ({time.time()-t0:.1f}s)")
del inst


# ═══════════════════════════════════════════════════════════════
# STEP 2.5 — Aggregation from POS Cash Balance
# ═══════════════════════════════════════════════════════════════
header("STEP 2.5 — POS Cash Balance Aggregation")
t0 = time.time()

pos_agg = pos.groupby('SK_ID_CURR').agg(
    pos_dpd_mean          = ('SK_DPD',               'mean'),
    pos_dpd_max           = ('SK_DPD',               'max'),
    pos_dpd_def_mean      = ('SK_DPD_DEF',           'mean'),
    pos_dpd_def_max       = ('SK_DPD_DEF',           'max'),
    pos_months_count      = ('MONTHS_BALANCE',       'count'),
    pos_cnt_instalment_mean = ('CNT_INSTALMENT',     'mean'),
    pos_active_count      = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Active').sum()),
    pos_completed_count   = ('NAME_CONTRACT_STATUS', lambda x: (x == 'Completed').sum()),
    pos_late_count        = ('SK_DPD',               lambda x: (x > 0).sum()),
).reset_index()

pos_agg['pos_completed_ratio'] = pos_agg['pos_completed_count'] / (pos_agg['pos_months_count'] + 1e-6)
pos_agg['pos_late_ratio']      = pos_agg['pos_late_count']      / (pos_agg['pos_months_count'] + 1e-6)

log(f"pos_agg shape: {pos_agg.shape}  ({time.time()-t0:.1f}s)")
del pos


# ═══════════════════════════════════════════════════════════════
# STEP 2.6 — Aggregation from Credit Card Balance
# ═══════════════════════════════════════════════════════════════
header("STEP 2.6 — Credit Card Balance Aggregation")
t0 = time.time()

cc_agg = cc.groupby('SK_ID_CURR').agg(
    cc_balance_mean           = ('AMT_BALANCE',          'mean'),
    cc_balance_max            = ('AMT_BALANCE',          'max'),
    cc_limit_mean             = ('AMT_CREDIT_LIMIT_ACTUAL', 'mean'),
    cc_limit_max              = ('AMT_CREDIT_LIMIT_ACTUAL', 'max'),
    cc_utilization_mean       = ('UTILIZATION_RATIO',    'mean'),
    cc_utilization_max        = ('UTILIZATION_RATIO',    'max'),
    cc_dpd_mean               = ('SK_DPD',               'mean'),
    cc_dpd_max                = ('SK_DPD',               'max'),
    cc_dpd_def_mean           = ('SK_DPD_DEF',           'mean'),
    cc_months_count           = ('MONTHS_BALANCE',       'count'),
    cc_drawings_mean          = ('AMT_DRAWINGS_CURRENT', 'mean'),
    cc_drawings_max           = ('AMT_DRAWINGS_CURRENT', 'max'),
    cc_payment_mean           = ('AMT_PAYMENT_CURRENT',  'mean'),
    cc_late_count             = ('SK_DPD',               lambda x: (x > 0).sum()),
).reset_index()

cc_agg['cc_late_ratio'] = cc_agg['cc_late_count'] / (cc_agg['cc_months_count'] + 1e-6)

log(f"cc_agg shape: {cc_agg.shape}  ({time.time()-t0:.1f}s)")
del cc


# ═══════════════════════════════════════════════════════════════
# STEP 2.7 — Merge & Validate (theo Plan §2.5)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.7 — Merge All Tables")
t0 = time.time()

n_before = app.shape[0]
df = app.copy()

df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
log(f"After bureau merge    : {df.shape}")

df = df.merge(prev_agg,   on='SK_ID_CURR', how='left')
log(f"After prev merge      : {df.shape}")

df = df.merge(inst_agg,   on='SK_ID_CURR', how='left')
log(f"After inst merge      : {df.shape}")

df = df.merge(pos_agg,    on='SK_ID_CURR', how='left')
log(f"After pos merge       : {df.shape}")

df = df.merge(cc_agg,     on='SK_ID_CURR', how='left')
log(f"After cc merge        : {df.shape}")

# VALIDATION — row count MUST NOT change
assert df.shape[0] == n_before, \
    f"CRITICAL: Row count changed after merge! {n_before} -> {df.shape[0]}"
log(f"\n  ✓ Row count validation passed: {df.shape[0]:,} rows (unchanged)")
log(f"  Total features after merge: {df.shape[1]}")

# Missing rate for aggregated features (expected — no history = NaN)
agg_cols = (
    [c for c in bureau_agg.columns if c != 'SK_ID_CURR'] +
    [c for c in prev_agg.columns   if c != 'SK_ID_CURR'] +
    [c for c in inst_agg.columns   if c != 'SK_ID_CURR'] +
    [c for c in pos_agg.columns    if c != 'SK_ID_CURR'] +
    [c for c in cc_agg.columns     if c != 'SK_ID_CURR']
)
miss = df[agg_cols].isnull().mean().sort_values(ascending=False)
miss_nonzero = miss[miss > 0]
log(f"\n  Aggregated feature missing rates (NaN = no history, will fill 0):")
for col, rate in miss_nonzero.head(10).items():
    log(f"    {col:<45} {rate*100:.1f}%")

log(f"\n  Done in {time.time()-t0:.1f}s")

del bureau_agg, prev_agg, inst_agg, pos_agg, cc_agg


# ═══════════════════════════════════════════════════════════════
# STEP 2.8 — Encoding & Imputation (theo Plan §2.6)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.8 — Encoding & Imputation")

# Fill NaN from aggregation with 0
df[agg_cols] = df[agg_cols].fillna(0)
log(f"  Remaining NaN in agg cols: {df[agg_cols].isnull().sum().sum()}")

# Binary encoding for 2-category object columns
step("Label encode binary categorical columns")
le = LabelEncoder()
binary_cols = [c for c in df.select_dtypes('object').columns if df[c].nunique() <= 2]
for col in binary_cols:
    df[col] = le.fit_transform(df[col].astype(str))
log(f"  Binary encoded: {binary_cols}")

# One-hot encode remaining multi-category columns
step("One-hot encode remaining categorical columns")
cat_cols = df.select_dtypes('object').columns.tolist()
cat_cols = [c for c in cat_cols if c not in ['SK_ID_CURR']]
log(f"  Columns to OHE: {cat_cols}")
df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
log(f"  Shape after OHE: {df.shape}")

# Impute remaining NaN (from app columns like DAYS_LAST_PHONE_CHANGE etc.)
step("Impute remaining numerical NaN with median")
num_cols = df.select_dtypes(include=np.number).columns.tolist()
exclude_from_impute = ['SK_ID_CURR', 'TARGET', 'IS_TRAIN']
impute_cols = [c for c in num_cols if c not in exclude_from_impute]

# Fit imputer on TRAIN only to avoid data leakage
train_mask = df['IS_TRAIN'] == 1
imputer = SimpleImputer(strategy='median')
imputer.fit(df.loc[train_mask, impute_cols])
df[impute_cols] = imputer.transform(df[impute_cols])

remaining_nan = df[impute_cols].isnull().sum().sum()
log(f"  Remaining NaN after imputation: {remaining_nan}")
assert remaining_nan == 0, "Imputation failed — still has NaN!"


# ═══════════════════════════════════════════════════════════════
# STEP 2.11 — Interaction Features (Moved Up & Fixed)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.11 — Interaction Features")

# Nhóm 1: Financial stress composite
df['STRESS_AGE_X_CREDIT']   = df['AGE_YEARS'] * df['CREDIT_INCOME_RATIO']
df['STRESS_EMP_X_ANNUITY']  = df['YEARS_EMPLOYED'].clip(lower=0) * df['ANNUITY_INCOME_RATIO']
flag_docs_col = [c for c in df.columns if 'FLAG_DOCUMENT' in c]
if flag_docs_col:
    df['FLAG_DOCS_SUM'] = df[flag_docs_col].sum(axis=1)
df['STRESS_DOCS_X_CREDIT']  = df.get('FLAG_DOCS_SUM', pd.Series(0, index=df.index)) * df['CREDIT_INCOME_RATIO']

# Nhóm 2: Bureau quality composite
bureau_overdue_n = df.get('bureau_overdue_mean', pd.Series(0, index=df.index)).clip(0).pipe(
    lambda s: s / (s.quantile(0.99) + 1e-6)).clip(0, 1)
prev_refused_n   = df.get('prev_refused_ratio', pd.Series(0, index=df.index)).clip(0, 1)
inst_late_n      = df.get('inst_late_ratio',    pd.Series(0, index=df.index)).clip(0, 1)
df['BUREAU_QUALITY_COMPOSITE'] = (
    (1 - bureau_overdue_n) * 0.40 +
    (1 - prev_refused_n)   * 0.35 +
    (1 - inst_late_n)      * 0.25
)

# Nhóm 3: False Negative detector
bureau_clean         = (df.get('bureau_bad_debt_flag', pd.Series(0, index=df.index)) == 0).astype(int)
high_lti             = (df['CREDIT_INCOME_RATIO'] > 5.0).astype(int)
df['CLEAN_BUREAU_HIGH_LTI'] = bureau_clean * high_lti

# Nhóm 4: External score composite
ext_cols = [c for c in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3'] if c in df.columns]
if len(ext_cols) >= 2:
    df['EXT_SOURCE_MEAN']    = df[ext_cols].mean(axis=1)
    df['EXT_SOURCE_MIN']     = df[ext_cols].min(axis=1)
    df['EXT_SOURCE_PRODUCT'] = df[ext_cols].product(axis=1)
    df['EXT_SOURCE_STD']     = df[ext_cols].std(axis=1).fillna(0)

log("✓ Created interaction features successfully before feature selection.")


# ═══════════════════════════════════════════════════════════════
# STEP 2.9 — Feature Selection (Khởi tạo danh sách chuẩn xác)
# ═══════════════════════════════════════════════════════════════
header("STEP 2.9 — Feature Selection")

# Định nghĩa danh sách các đặc trưng tham gia huấn luyện (loại bỏ các ID và cột kiểm soát)
feature_cols = [c for c in df.columns if c not in ['SK_ID_CURR', 'TARGET', 'IS_TRAIN']]
n_before_selection = len(feature_cols)

# ── 2.9a. Remove zero-variance columns ────────────────────────
step("Remove near-zero variance features (var < 0.01)")
variances = df[feature_cols].var()
zero_var = variances[variances < 0.01].index.tolist()
df.drop(columns=zero_var, inplace=True)
feature_cols = [c for c in feature_cols if c not in zero_var]
log(f"  Dropped {len(zero_var)} near-zero variance features")
if zero_var:
    log(f"  Examples: {zero_var[:5]}")

# ── 2.9b. Remove highly correlated pairs (r > 0.95) ───────────
step("Remove highly correlated features (|r| > 0.95)")
corr_matrix = df[feature_cols].corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1))
to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.95)]
df.drop(columns=to_drop_corr, inplace=True)
feature_cols = [c for c in feature_cols if c not in to_drop_corr]
log(f"  Dropped {len(to_drop_corr)} highly correlated features")
if to_drop_corr:
    log(f"  Examples: {to_drop_corr[:5]}")

log(f"\n  Feature count: {n_before_selection} → {len(feature_cols)}")
log(f"  Total dropped: {n_before_selection - len(feature_cols)}")


# ═══════════════════════════════════════════════════════════════
# STEP 2.10 — Compute Target Correlation & Save
# ═══════════════════════════════════════════════════════════════
header("STEP 2.10 — Target Correlation & Save")

step("Computing correlation with TARGET (on train set only)")
train_df = df[df['IS_TRAIN'] == 1].copy()
correlations = train_df[feature_cols].corrwith(train_df['TARGET']).abs().sort_values(ascending=False)
top20 = correlations.head(20)
log("Top 20 features by |correlation with TARGET|:")
for i, (feat, corr) in enumerate(top20.items(), 1):
    log(f"  {i:>2}. {feat:<55} {corr:.4f}")

step("Splitting back to train and test")
train_out = df[df['IS_TRAIN'] == 1].drop(columns=['IS_TRAIN'])
test_out  = df[df['IS_TRAIN'] == 0].drop(columns=['IS_TRAIN', 'TARGET'])

log(f"  train_out shape: {train_out.shape}")
log(f"  test_out  shape: {test_out.shape}")

# Final validation
assert train_out.shape[0] == 307511, f"Train row count wrong: {train_out.shape[0]}"
assert test_out.shape[0]  ==  48744, f"Test row count wrong: {test_out.shape[0]}"
assert train_out['TARGET'].isnull().sum() == 0, "TARGET has NaN in train!"
log("  ✓ Row count validation passed")

step("Saving Parquet files")
train_path = os.path.join(OUTPUT_DIR, 'train_features.parquet')
test_path  = os.path.join(OUTPUT_DIR, 'test_features.parquet')
train_out.to_parquet(train_path, index=False)
test_out.to_parquet(test_path, index=False)

train_mb = os.path.getsize(train_path) / 1e6
test_mb  = os.path.getsize(test_path)  / 1e6
log(f"  train_features.parquet : {train_mb:.1f} MB  ({train_out.shape[1]} features)")
log(f"  test_features.parquet  : {test_mb:.1f} MB  ({test_out.shape[1]} features)")

# Save feature list to text file
feat_list_path = os.path.join(OUTPUT_DIR, 'feature_list.txt')
with open(feat_list_path, 'w', encoding='utf-8') as f:
    f.write(f"Total features: {len(feature_cols)}\n\n")
    f.write("=== Top 20 by |correlation with TARGET| ===\n")
    for feat, corr in top20.items():
        f.write(f"{corr:.4f}  {feat}\n")
    f.write("\n=== All features ===\n")
    for feat in feature_cols:
        f.write(feat + "\n")
log(f"  feature_list.txt saved to {feat_list_path}")

print(f"""
{'='*65}
  PHASE 2 COMPLETE
{'='*65}
  Total time   : {time.time()-t_total:.0f}s
  Train rows   : {train_out.shape[0]:,}
  Test rows    : {test_out.shape[0]:,}
  Final features: {len(feature_cols)}
  Output files :
    {train_path}
    {test_path}
    {feat_list_path}
{'='*65}
""")