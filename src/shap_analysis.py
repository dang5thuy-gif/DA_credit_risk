"""
shap_analysis.py — Phase 4: Model Interpretation (Enterprise Grade v2.2)
========================================================================
Home Credit Default Risk — Giải thích mô hình hộp đen bằng SHAP & PDP

Đã sửa lỗi: Tự động đồng bộ chính xác danh sách đặc trưng (FEATURES) 
           được lưu từ Phase 3 để tránh lỗi lệch số lượng cột (170 vs 164).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import os, pickle, warnings, re
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from sklearn.inspection import PartialDependenceDisplay

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
MODELS_DIR  = str(ROOT_DIR / 'models')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

def header(t): print(f"\n{'='*65}\n  {t}\n{'='*65}")
def step(t):   print(f"\n  >> {t}")
def log(t):    print(f"     {t}")

# ═══════════════════════════════════════════════════════════════
# STEP 1 — Load Model, Feature List, and Sample Data
# ═══════════════════════════════════════════════════════════════
header("STEP 1 — Load Model and Sample Data")

model_path = os.path.join(MODELS_DIR, 'best_production_model.pkl')
feature_list_path = os.path.join(MODELS_DIR, 'feature_list.pkl')

if not os.path.exists(model_path) or not os.path.exists(feature_list_path):
    print("ERROR: Không tìm thấy file model hoặc feature_list.pkl. Vui lòng chạy modeling.py trước.")
    sys.exit(1)

step("Loading best production model & synchronized feature list...")
with open(model_path, 'rb') as f:
    model = pickle.load(f)
with open(feature_list_path, 'rb') as f:
    FEATURES = pickle.load(f)

log(f"Số lượng đặc trưng mô hình yêu cầu: {len(FEATURES)}")

step("Loading data sample (10,000 rows for SHAP calculation)...")
train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train_features.parquet'))

# Chuẩn hóa lại tên cột y hệt như lúc training trong modeling.py
train_df = train_df.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]+', '_', x))

# Trích xuất tập mẫu 10,000 dòng để tính toán SHAP nhanh hơn
np.random.seed(42)
sample_idx = np.random.choice(train_df.index, size=min(10000, len(train_df)), replace=False)
df_sample = train_df.loc[sample_idx]

# ÉP BUỘC X_sample chỉ chứa chính xác các cột nằm trong danh sách FEATURES đã train
X_sample = df_sample[FEATURES]
y_sample = df_sample['TARGET'] if 'TARGET' in df_sample.columns else None

log(f"Khớp dữ liệu thành công! Sample shape đưa vào SHAP: {X_sample.shape}")


# ═══════════════════════════════════════════════════════════════
# STEP 2 — Calculating Global SHAP Values
# ═══════════════════════════════════════════════════════════════
header("STEP 2 — Calculating Global SHAP Values")
step("Initializing SHAP TreeExplainer...")
explainer = shap.TreeExplainer(model)

step("Computing SHAP values for X_sample (This may take a few minutes)...")
shap_values = explainer(X_sample)

# Hỗ trợ xử lý cho cả cấu trúc SHAP cũ và mới (.values hoặc mảng n-chiều)
if isinstance(shap_values, shap.Explanation):
    # Đối với binary classification, LightGBM trả về SHAP cho class 1 ở chiều cuối nếu có nhiều chiều
    if len(shap_values.shape) == 3: 
        shap_values_matrix = shap_values.values[:, :, 1]
    else:
        shap_values_matrix = shap_values.values
    explanation_obj = shap_values
else:
    shap_values_matrix = shap_values
    explanation_obj = shap.Explanation(values=shap_values, data=X_sample, feature_names=FEATURES)

log("✓ SHAP values calculated successfully.")


# ═══════════════════════════════════════════════════════════════
# STEP 3 — Generating Global Interpretability Plots
# ═══════════════════════════════════════════════════════════════
header("STEP 3 — Generating Global Interpretability Plots")

# 1. Biểu đồ SHAP Summary Bar Chart
step("Plotting SHAP Feature Importance (Bar Chart)...")
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_matrix, X_sample, plot_type="bar", show=False)
plt.title("Global Feature Importance (Absolute Mean SHAP Value)", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_summary_bar.png'), dpi=150)
plt.close()
log("Saved: reports/shap_summary_bar.png")

# 2. Biểu đồ SHAP Beeswarm Plot
step("Plotting SHAP Beeswarm Distribution...")
plt.figure(figsize=(11, 7))
if len(explanation_obj.shape) == 3:
    shap.plots.beeswarm(explanation_obj[:, :, 1], max_display=15, show=False)
else:
    shap.plots.beeswarm(explanation_obj, max_display=15, show=False)
plt.title("SHAP Beeswarm Plot — Feature Impact Distribution on Default Risk", fontsize=12, pad=15)
plt.tight_layout()
plt.savefig(os.path.join(REPORTS_DIR, 'shap_beeswarm.png'), dpi=150)
plt.close()
log("Saved: reports/shap_beeswarm.png")


# ═══════════════════════════════════════════════════════════════
# STEP 4 — Local Interpretation via Waterfall Plots
# ═══════════════════════════════════════════════════════════════
header("STEP 4 — Local Interpretation (Waterfall Plots)")

# Trích xuất cấu trúc nền của Explainer để vẽ đồ thị Waterfall
base_value = explainer.expected_value
if isinstance(base_value, (list, np.ndarray)) and len(base_value) > 1:
    base_value = base_value[1] # Lấy base value của class 1 rủi ro nợ xấu

# Tìm kiếm khách hàng Rủi ro cao thực tế và Rủi ro thấp thực tế trong tập mẫu
if y_sample is not None:
    high_risk_candidates = X_sample[y_sample == 1]
    low_risk_candidates  = X_sample[y_sample == 0]
    
    idx_high = high_risk_candidates.index[0] if not high_risk_candidates.empty else X_sample.index[0]
    idx_low  = low_risk_candidates.index[0] if not low_risk_candidates.empty else X_sample.index[1]
    
    # Định vị vị trí dòng tương ứng trong ma trận SHAP
    pos_high = X_sample.index.get_loc(idx_high)
    pos_low  = X_sample.index.get_loc(idx_low)
    
    # Trích xuất ma trận giá trị đơn lẻ
    val_high = shap_values_matrix[pos_high]
    val_low  = shap_values_matrix[pos_low]
    
    # 1. Khách hàng Rủi ro cao (High-Risk Customer)
    step(f"Generating Waterfall plot for High-Risk Client (Index: {idx_high})...")
    exp_high = shap.Explanation(values=val_high, base_values=base_value, 
                                data=X_sample.iloc[pos_high], feature_names=FEATURES)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_high, max_display=10, show=False)
    plt.title(f"SHAP Local Explanation — High-Risk Client Portfolio", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'shap_waterfall_highrisk.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved: reports/shap_waterfall_highrisk.png")
    
    # 2. Khách hàng Rủi ro thấp (Low-Risk Customer)
    step(f"Generating Waterfall plot for Low-Risk Client (Index: {idx_low})...")
    exp_low = shap.Explanation(values=val_low, base_values=base_value, 
                               data=X_sample.iloc[pos_low], feature_names=FEATURES)
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp_low, max_display=10, show=False)
    plt.title(f"SHAP Local Explanation — Low-Risk Client Portfolio", fontsize=12, pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'shap_waterfall_lowrisk.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved: reports/shap_waterfall_lowrisk.png")


# ═══════════════════════════════════════════════════════════════
# STEP 5 — Partial Dependence Plots (PDP)
# ═══════════════════════════════════════════════════════════════
header("STEP 5 — Partial Dependence Plots (PDP)")
step("Generating PDP for critical core features...")

# Lọc ra các đặc trưng quan trọng nhất có mặt trong dữ liệu thực tế
pdp_candidates = ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'CREDIT_INCOME_RATIO', 'AGE_YEARS']
pdp_features = [f for f in pdp_candidates if f in FEATURES]

if pdp_features:
    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model, X_sample, pdp_features,
        ax=ax, grid_resolution=40, 
        line_kw={"color": "#E74C3C", "linewidth": 2}
    )
    plt.suptitle("Partial Dependence Plots (PDP) — Marginal Effect on Default Probability", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, 'pdp_top_features.png'), dpi=150, bbox_inches='tight')
    plt.close()
    log("Saved: reports/pdp_top_features.png")

print(f"""
{'='*65}
  PHASE 4: SHAP MODEL INTERPRETATION COMPLETE
{'='*65}
  Tất cả các biểu đồ giải thích đã được lưu thành công!
  - reports/shap_summary_bar.png
  - reports/shap_beeswarm.png
  - reports/shap_waterfall_highrisk.png
  - reports/shap_waterfall_lowrisk.png
  - reports/pdp_top_features.png
{'='*65}
""")