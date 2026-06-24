"""
modeling.py — Phase 3: Multi-Model Benchmark Pipeline (Enterprise Grade v3.0)
=============================================================================
Home Credit Default Risk — Dự đoán xác suất nợ xấu

Tích hợp và so sánh hiệu năng 4 thuật toán mạnh nhất:
  - LightGBM (Light Gradient Boosting Machine)
  - XGBoost (Extreme Gradient Boosting)
  - CatBoost (Categorical Boosting)
  - Random Forest (Bagging Ensemble)

Đầu ra:
  - reports/model_comparison_leaderboard.csv (Bảng so sánh các chỉ số KPI)
  - models/best_production_model.pkl (Mô hình chiến thắng)
  - models/lgbm_fold1-5.pkl, xgboost_fold1-5.pkl, catboost_fold1-5.pkl
  - models/model_metrics.json (Metadata hiệu năng hệ thống)
  - data/submission.csv (Kết quả dự đoán của mô hình tốt nhất phục vụ Kaggle)
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, time, warnings, pickle, re, json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from pathlib import Path
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
MODELS_DIR  = str(ROOT_DIR / 'models')
REPORTS_DIR = str(ROOT_DIR / 'reports')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*75}\n  {t}\n{'='*75}")
def step(t):   print(f"\n  >> {t}")
def log(t):    print(f"     {t}")

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS — Tính toán chỉ số KS (Kolmogorov-Smirnov)
# ─────────────────────────────────────────────────────────────────────────────
def calculate_ks_statistic(y_true, y_prob):
    """Tính toán chỉ số KS để đo lường khả năng phân tách nhóm Tốt/Xấu"""
    df_ks = pd.DataFrame({'target': y_true, 'prob': y_prob})
    df_ks['bucket'] = pd.qcut(df_ks['prob'], 10, labels=False, duplicates='drop')
    grouped = df_ks.groupby('bucket')
    ks_table = grouped.agg(
        total=('target', 'count'),
        defaults=('target', 'sum')
    ).reset_index()
    ks_table['goods'] = ks_table['total'] - ks_table['defaults']
    ks_table['pct_defaults'] = ks_table['defaults'] / max(1, ks_table['defaults'].sum())
    ks_table['pct_goods'] = ks_table['goods'] / max(1, ks_table['goods'].sum())
    ks_table['cum_defaults'] = ks_table['pct_defaults'].cumsum()
    ks_table['cum_goods'] = ks_table['pct_goods'].cumsum()
    ks_table['ks_diff'] = np.abs(ks_table['cum_goods'] - ks_table['cum_defaults'])
    return float(ks_table['ks_diff'].max())


# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load Data & Prepare Features
# ═══════════════════════════════════════════════════════════════
t_total = time.time()
header("STEP 1 — Loading Train & Test Features")

train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))
test_df  = pd.read_parquet(os.path.join(DATA_DIR, 'test_features.parquet'))
log(f"Train dataset shape: {train_df.shape} | Test dataset shape: {test_df.shape}")

# Chuẩn hóa tên cột (XGBoost/LightGBM yêu cầu không chứa ký tự đặc biệt)
train_df = train_df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))
test_df  = test_df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

FEATURES = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
X = train_df[FEATURES]
y = train_df['TARGET']
X_test = test_df[FEATURES]

log(f"Số lượng đặc trưng (Features) đưa vào mô hình: {len(FEATURES)}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Define Models & Default Configurations
# ═══════════════════════════════════════════════════════════════
header("STEP 2 — Initializing Models & Baseline Hyperparameters")

# Kiểm tra xem có cấu hình Optuna tối ưu sẵn cho LightGBM không
best_params_path = os.path.join(MODELS_DIR, 'best_params.json')
lgbm_init_params = {
    'objective': 'binary', 'metric': 'auc', 'boosting_type': 'gbdt',
    'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
    'random_state': 42, 'n_estimators': 1500, 'verbose': -1, 'n_jobs': -1
}
if os.path.exists(best_params_path):
    try:
        with open(best_params_path, 'r') as f:
            lgbm_init_params.update(json.load(f))
        log("✓ Đã nạp thành công bộ tham số tối ưu từ Optuna cho LightGBM.")
    except Exception: pass

neg, pos = np.bincount(y)
pos_weight = neg / pos

# Tự động lấy các cột phân loại (Categorical Features)
cat_cols = [c for c in X.columns if X[c].dtype == 'object' or X[c].dtype.name == 'category']

model_templates = {
    'LightGBM': {
        'constructor': lambda: lgb.LGBMClassifier(**lgbm_init_params, class_weight='balanced'),
        'fit_params': lambda X_v, y_v: {
            'eval_set': [(X_v, y_v)],
            'callbacks': [lgb.early_stopping(stopping_rounds=100, verbose=False)]
        }
    },
    'XGBoost': {
        'constructor': lambda: xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            tree_method='hist',
            grow_policy='lossguide',           # Leaf-wise growth
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.05,
            reg_lambda=1.0,
            gamma=0.1,
            min_child_weight=5,
            scale_pos_weight=pos_weight,
            random_state=42,
            n_estimators=2000,
            n_jobs=-1,
            # Học thuật: Monotonic Constraints (rất quan trọng credit risk)
            # monotone_constraints=[-1 if 'EXT_SOURCE' in col else 0 for col in FEATURES]
        ),
        'fit_params': lambda X_v, y_v: {
            'eval_set': [(X_v, y_v)],
            'early_stopping_rounds': 100,
            'verbose': False
        }
    },

    'CatBoost_Ordered': {
        'constructor': lambda: CatBoostClassifier(
            iterations=2000,
            learning_rate=0.05,
            depth=8,
            eval_metric='AUC',
            boosting_type='Ordered',           # Ordered Boosting - điểm mạnh lớn nhất
            bootstrap_type='Bayesian',
            bagging_temperature=1.0,
            random_strength=1,
            cat_features=cat_cols,             # Native categorical handling
            one_hot_max_size=10,
            od_type='Iter',
            od_wait=50,
            random_state=42,
            verbose=False,
            thread_count=-1,
            # class_weights=[1, pos_weight]    # nếu cần
        ),
        'fit_params': lambda X_v, y_v: {
            'eval_set': [(X_v, y_v)],
            'early_stopping_rounds': 100,
            'verbose': False
        }
    },
    'RandomForest': {
        'constructor': lambda: RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            random_state=42, n_jobs=-1
        ),
        'fit_params': lambda X_v, y_v: {}
    }
}

for m_name in model_templates.keys():
    log(f"Mô hình tích hợp sẵn: {m_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Unified 5-Fold Cross-Validation Loop
# ═══════════════════════════════════════════════════════════════
header("STEP 3 — Training 4 Models with Stratified 5-Fold CV")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Từ điển lưu trữ dự đoán OOF và Dự đoán tập Test của từng mô hình
dict_oof_preds = {name: np.zeros(len(train_df)) for name in model_templates}
dict_test_preds = {name: np.zeros(len(test_df)) for name in model_templates}
dict_trained_models = {name: [] for name in model_templates}

# Vòng lặp huấn luyện chéo
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y), 1):
    step(f"Processing Fold {fold} / 5...")
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val     = X.iloc[val_idx], y.iloc[val_idx]
    
    # Huấn luyện từng thuật toán trên cùng một tập phân tách dữ liệu (Fold)
    for model_name, config in model_templates.items():
        t_start = time.time()
        
        # Khởi tạo instance mới cho mô hình
        clf = config['constructor']()
        f_params = config['fit_params'](X_val, y_val)
        
        # Xử lý tham số early stopping đặc thù của XGBoost
        if model_name == 'XGBoost':
            clf.early_stopping_rounds = 100
            clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            clf.fit(X_train, y_train, **f_params)
            
        # Dự đoán xác suất cho tập Validation (OOF) và tập Test
        val_p = clf.predict_proba(X_val)[:, 1]
        test_p = clf.predict_proba(X_test)[:, 1]
        
        # Ghi nhận kết quả
        dict_oof_preds[model_name][val_idx] = val_p
        dict_test_preds[model_name] += test_p / folds.n_splits
        dict_trained_models[model_name].append(clf)
        
        f_auc = roc_auc_score(y_val, val_p)
        log(f"{model_name:<12} — Fold {fold} AUC: {f_auc:.4f} | Time: {time.time()-t_start:.1f}s")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Strict Probability Calibration (Chống Leakage)
# ═══════════════════════════════════════════════════════════════
header("STEP 4 — Probability Calibration (Isotonic Regression)")
log("Fitting Isotonic Calibrator strictly on Out-Of-Fold predictions to avoid leakage...")

dict_oof_calibrated = {}
dict_test_calibrated = {}
dict_calibrators = {}

for model_name in model_templates:
    oof_raw = dict_oof_preds[model_name]
    test_raw = dict_test_preds[model_name]
    
    # Khởi tạo và đồng bộ bộ hiệu chuẩn xác suất
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(oof_raw, y) # Fit trên toàn bộ tập OOF hợp lệ
    
    dict_oof_calibrated[model_name] = calibrator.transform(oof_raw)
    dict_test_calibrated[model_name] = calibrator.transform(test_raw)
    dict_calibrators[model_name] = calibrator
    log(f"✓ Hoàn tất hiệu chuẩn xác suất cho mô hình: {model_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Performance Evaluation Leaderboard
# ═══════════════════════════════════════════════════════════════
header("STEP 5 — Generating Model Performance Leaderboard")

leaderboard_data = []

for model_name in model_templates:
    preds_cal = dict_oof_calibrated[model_name]
    
    # Tính toán các chỉ số Core Metrics
    auc_score = roc_auc_score(y, preds_cal)
    gini_score = 2 * auc_score - 1
    ks_score = calculate_ks_statistic(y, preds_cal)
    brier_score = brier_score_loss(y, preds_cal)
    
    leaderboard_data.append({
        'Model': model_name,
        'OOF_AUC': round(auc_score, 4),
        'Gini_Coefficient': round(gini_score, 4),
        'KS_Statistic': round(ks_score * 100, 2), # % đơn vị
        'Brier_Score': round(brier_score, 5)
    })

# Chuyển đổi thành Dataframe để sắp xếp thứ hạng
leaderboard_df = pd.DataFrame(leaderboard_data)
leaderboard_df = leaderboard_df.sort_values(by='OOF_AUC', ascending=False).reset_index(drop=True)

# Hiển thị bảng so sánh trực quan
print("\n" + leaderboard_df.to_string(index=False))

# Xuất bảng xếp hạng ra file CSV báo cáo
leaderboard_path = os.path.join(REPORTS_DIR, 'model_comparison_leaderboard.csv')
leaderboard_df.to_csv(leaderboard_path, index=False)
log(f"\nSaved Leaderboard comparison report to: {leaderboard_path}")

# --- Bổ sung vào modeling.py sau STEP 5 ---
header("PHÂN TÍCH HỌC THUẬT: Calibration Curves")
plt.figure(figsize=(8, 6))

for model_name in model_templates:
    prob_true, prob_pred = calibration_curve(y, dict_oof_calibrated[model_name], n_bins=10)
    plt.plot(prob_pred, prob_true, marker='s', label=f'{model_name}')

plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.title('Reliability Diagram: So sánh độ tin cậy xác suất giữa các thuật toán')
plt.xlabel('Xác suất dự báo trung bình')
plt.ylabel('Tỷ lệ vỡ nợ thực tế')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(os.path.join(REPORTS_DIR, 'calibration_curve.png'), dpi=150)
plt.close()
print("✓ Đã xuất biểu đồ Reliability Diagram.")


# ═══════════════════════════════════════════════════════════════
# STEP 6 — Export Production Artifacts
# ═══════════════════════════════════════════════════════════════
header("STEP 6 — Exporting Winning Production Artifacts & Multi-Model Folds")

all_model_names = list(model_templates.keys())

# Lưu 5-Folds
for model_name in all_model_names:
    if model_name == 'CatBoost_Ordered':
        file_prefix = 'catboost'
    elif model_name == 'LightGBM':
        file_prefix = 'lgbm'
    else:
        file_prefix = model_name.lower()
        
    for fold_idx, clf_fold in enumerate(dict_trained_models[model_name], 1):
        fold_file_name = f"{file_prefix}_fold{fold_idx}.pkl"
        with open(os.path.join(MODELS_DIR, fold_file_name), 'wb') as f:
            pickle.dump(clf_fold, f)

log("✓ Đã lưu trọn bộ 5-Folds của tất cả các mô hình.")

# Lưu Best Production Model (single best fold)
best_model_name = leaderboard_df.loc[0, 'Model']
best_model_list = dict_trained_models[best_model_name] # Lưu list 5 models

with open(os.path.join(MODELS_DIR, 'best_production_model.pkl'), 'wb') as f:
    pickle.dump(best_model_list, f)

with open(os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl'), 'wb') as f:
    pickle.dump(dict_calibrators[best_model_name], f)

with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'wb') as f:
    pickle.dump(FEATURES, f)

log(f"✓ Best Production Model saved: {best_model_name}")


# ═══════════════════════════════════════════════════════════════
# STEP 7 — Plot ROC Curves
# ═══════════════════════════════════════════════════════════════
header("STEP 7 — Plotting ROC Curves")

plt.figure(figsize=(10, 8))
color_map = {
    'LightGBM': 'darkorange', 
    'XGBoost': 'teal', 
    'CatBoost_Ordered': 'indigo',
    'RandomForest': 'crimson'
}

for model_name in all_model_names:
    fpr, tpr, _ = roc_curve(y, dict_oof_calibrated[model_name])
    auc_val = roc_auc_score(y, dict_oof_calibrated[model_name])
    plt.plot(fpr, tpr, color=color_map.get(model_name, 'blue'), lw=2.2,
             label=f'{model_name} (AUC = {auc_val:.4f})')

plt.plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Benchmark Comparison — Out-Of-Fold ROC Curves')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'model_benchmark_roc.png'), dpi=150)
plt.close()

log(f"✓ Saved ROC Curve: reports/model_benchmark_roc.png")


# ═══════════════════════════════════════════════════════════════
# STEP 8 — Export Metrics, Submission & Final Summary
# ═══════════════════════════════════════════════════════════════
header("STEP 8 — Export Final Metrics & Submission")

# Export model_metrics.json
metrics_dict = leaderboard_df.set_index('Model').to_dict(orient='index')
with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w', encoding='utf-8') as f:
    json.dump(metrics_dict, f, indent=4, ensure_ascii=False)

log("✓ Saved model_metrics.json")

# Tạo submission.csv cho Kaggle
submission = pd.DataFrame({
    'SK_ID_CURR': test_df['SK_ID_CURR'].astype(int),
    'TARGET': dict_test_calibrated[best_model_name]
})
submission.to_csv(os.path.join(DATA_DIR, 'submission.csv'), index=False)
log("✓ Saved submission.csv")

best_model_name = leaderboard_df.loc[0, 'Model']
best_model_auc  = leaderboard_df.loc[0, 'OOF_AUC']

print(f"""
{'='*75}
  MULTI-MODEL BENCHMARK PIPELINE COMPLETE
{'='*75}
  Mô hình tối ưu nhất    : {best_model_name} (AUC: {best_model_auc:.4f})
  Tổng thời gian         : {time.time()-t_total:.1f}s
  Các file quan trọng đã được lưu:
    • best_production_model.pkl (5 folds)
    • feature_list.pkl
    • isotonic_calibrator.pkl
    • model_metrics.json
    • submission.csv
{'='*75}
""")