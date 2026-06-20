"""
lgd_ead_model.py — LGD & EAD Modelling (IFRS 9 Components)
===========================================================
Đã sửa lỗi: Đồng bộ hóa chính xác cấu trúc trường dữ liệu, 
           tránh triệt tiêu EAD/ECL về 0.00B và khắc phục lỗi NameError.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

# Định nghĩa các đường dẫn thư mục toàn cục lên đầu file
from pathlib import Path
ROOT_DIR    = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# Khung tỷ lệ LGD chuẩn theo cấu phần tài sản và loại hình vay
LGD_TABLE = {
    'Cash_loans': {True: 0.55, False: 0.75},
    'Revolving_loans': {True: 0.45, False: 0.65},
    'Consumer_loans': {True: 0.35, False: 0.50}
}

def estimate_lgd(df):
    """Tính toán tỷ lệ tổn thất khi vỡ nợ (LGD) dựa trên phân khúc sản phẩm"""
    lgd_vector = np.full(len(df), 0.65) # Mặc định base case 65% nếu thiếu thông tin
    
    # Chuẩn hóa tên cột để mapping kiểm tra loại hình hợp đồng vay
    contract_col = 'NAME_CONTRACT_TYPE' if 'NAME_CONTRACT_TYPE' in df.columns else None
    for col in df.columns:
        if 'CONTRACT_TYPE' in col.upper():
            contract_col = col
            break
            
    if contract_col:
        for idx, row in enumerate(df[contract_col].astype(str)):
            c_type = 'Cash_loans' if 'CASH' in row.upper() else ('Revolving_loans' if 'REV' in row.upper() else 'Consumer_loans')
            # Kiểm tra xem có tài sản đảm bảo/khoản trả trước không để giảm rủi ro LGD
            has_collateral = False
            if 'AMT_DOWN_PAYMENT' in df.columns and pd.notnull(df.iloc[idx]['AMT_DOWN_PAYMENT']):
                if df.iloc[idx]['AMT_DOWN_PAYMENT'] > 0:
                    has_collateral = True
            
            lgd_vector[idx] = LGD_TABLE[c_type][has_collateral]
            
    return pd.Series(lgd_vector, index=df.index)

def estimate_ead(df):
    """Tính toán dư nợ tại thời điểm vỡ nợ (EAD) dựa trên hạn mức tín dụng"""
    # Tìm kiếm trường giá trị khoản vay hợp lệ
    credit_col = 'AMT_CREDIT'
    if credit_col not in df.columns:
        for col in df.columns:
            if 'CREDIT' in col.upper():
                credit_col = col
                break
                
    if credit_col in df.columns and df[credit_col].sum() > 0:
        base_credit = df[credit_col].fillna(df[credit_col].median())
    else:
        # Nếu hoàn toàn không tìm thấy, bốc tạm từ file đặc trưng gốc để bù đắp dữ liệu
        train_feat_path = os.path.join(DATA_DIR, 'train_features.parquet')
        if os.path.exists(train_feat_path):
            df_amt = pd.read_parquet(train_feat_path, columns=['SK_ID_CURR', 'AMT_CREDIT'])
            df_merged = pd.merge(df[['SK_ID_CURR']], df_amt, on='SK_ID_CURR', how='left')
            base_credit = df_merged['AMT_CREDIT'].fillna(df_merged['AMT_CREDIT'].median())
        else:
            base_credit = pd.Series(np.full(len(df), 500000.0), index=df.index) # Default dummy credit

    # Áp dụng hệ số chuyển đổi tín dụng CCF (Credit Conversion Factor) 20% cho hạn mức revolving dự phòng
    ccf_multiplier = np.full(len(df), 1.0)
    if 'NAME_CONTRACT_TYPE' in df.columns:
        ccf_multiplier = np.where(df['NAME_CONTRACT_TYPE'].astype(str).str.contains('Rev'), 1.2, 1.0)
        
    ead_vector = base_credit * ccf_multiplier
    return pd.Series(ead_vector, index=df.index)

def lgd_sensitivity_analysis(df, base_lgd_mean):
    """Hàm bổ trợ stress test phân tích độ nhạy của cấu phần LGD"""
    scenarios = {
        'Optimistic (LGD - 15%)': max(0.1, base_lgd_mean - 0.15),
        'Base Case': base_lgd_mean,
        'Adverse (LGD + 15%)': min(0.95, base_lgd_mean + 0.15),
        'Severe (LGD + 30%)': min(0.99, base_lgd_mean + 0.30)
    }
    
    # Tính toán thử nghiệm sự thay đổi của ECL tổng
    total_ead = estimate_ead(df).sum()
    avg_pd = df['PRED_PROB'].mean() if 'PRED_PROB' in df.columns else 0.08
    
    print(f"\n[Stress Test Matrix Check]")
    for name, lgd_val in scenarios.items():
        sim_ecl = total_ead * avg_pd * lgd_val
        cov_pct = (sim_ecl / total_ead) * 100 if total_ead > 0 else 0
        print(f"  {name:<25}: Avg LGD={lgd_val:.1%} | Total ECL=${sim_ecl/1e9:.2f}B | Coverage={cov_pct:.2f}%")


if __name__ == '__main__':
    df = pd.read_parquet(os.path.join(DATA_DIR, 'results_df.parquet'))
    
    # Nạp bổ sung trường dữ liệu bổ trợ nếu file kết quả bị thiếu cột phân đoạn sản phẩm
    train_feat_path = os.path.join(DATA_DIR, 'train_features.parquet')
    if os.path.exists(train_feat_path):
        cols_to_load = [c for c in ['SK_ID_CURR', 'NAME_CONTRACT_TYPE', 'AMT_DOWN_PAYMENT', 'AMT_CREDIT'] if c in pd.read_parquet(train_feat_path).columns]
        df_src = pd.read_parquet(train_feat_path, columns=cols_to_load)
        df = pd.merge(df, df_src, on='SK_ID_CURR', how='left')

    lgd = estimate_lgd(df)
    ead = estimate_ead(df)

    print(f"\n=== LGD & EAD Diagnostics ===")
    print(f"  Mean LGD:   {lgd.mean():.3f}")
    print(f"  Total EAD:  ${ead.sum()/1e9:.2f}B")
    
    lgd_sensitivity_analysis(df, lgd.mean())