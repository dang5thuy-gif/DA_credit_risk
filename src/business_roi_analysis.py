"""
business_roi_analysis.py — Phase E: Business ROI & Threshold Optimization
=========================================================================
Trả lời câu hỏi: "Ở threshold nào thì ngân hàng tối đa hoá lợi nhuận?"

Đã sửa lỗi: Tự động kết hợp (Merge) dữ liệu AMT_CREDIT từ train_features.parquet
           vào kết quả dự đoán kết quả để tránh lỗi KeyError.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os, warnings
warnings.filterwarnings('ignore')

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR    = str(ROOT_DIR / 'data')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(REPORTS_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Business parameters — tunable per bank's P&L assumptions
# ─────────────────────────────────────────────────────────────────────────────
PARAMS = {
    'net_margin_rate':     0.035,  # 3.5% net interest margin trên khoản vay được duyệt
    'lgd_default':         0.65,   # Tổn thất khi vỡ nợ là 65% tổng khoản vay
    'opportunity_cost':    0.015,  # Chi phí cơ hội khi từ chối nhầm KH tốt là 1.5%
    'opex_rate':           0.002,  # Chi phí vận hành phê duyệt: 0.2% trên tổng khoản vay
}

def run_threshold_optimization(df):
    print("\n=================================================================")
    print("  Business ROI — Threshold Optimization")
    print("=================================================================")

    # Xác định giá trị khoản vay trung bình (Sử dụng cột AMT_CREDIT thực tế)
    avg_credit = df['AMT_CREDIT'].mean()
    total_loans = len(df)
    
    thresholds = np.linspace(0.01, 0.99, 100)
    results = []

    y_true = df['TARGET'].values
    y_prob = df['PRED_PROB'].values

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        
        # Tính toán ma trận nhầm lẫn (Confusion Matrix Elements)
        tp = np.sum((y_true == 1) & (y_pred == 1)) # Bị từ chối chính xác (Bad - Denied)
        fp = np.sum((y_true == 0) & (y_pred == 1)) # Bị từ chối nhầm (Good - Denied)
        fn = np.sum((y_true == 1) & (y_pred == 0)) # Bị duyệt nhầm (Bad - Approved)
        tn = np.sum((y_true == 0) & (y_pred == 0)) # Được duyệt chính xác (Good - Approved)

        # Công thức tài chính ngân hàng (P&L Financial Framework)
        revenue      = tn * PARAMS['net_margin_rate'] * avg_credit
        opp_cost     = fp * PARAMS['opportunity_cost'] * avg_credit
        loan_loss    = fn * PARAMS['lgd_default'] * avg_credit
        opex         = (tn + fn) * PARAMS['opex_rate'] * avg_credit
        
        net_profit = revenue - opp_cost - loan_loss - opex
        approval_rate = (tn + fn) / total_loans
        
        precision = tn / (tn + fn) if (tn + fn) > 0 else 0
        recall = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            'threshold': t,
            'net_profit_M': net_profit / 1e6,
            'approval_rate': approval_rate,
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
            'revenue_M': revenue / 1e6,
            'opp_cost_M': opp_cost / 1e6,
            'loan_loss_M': loan_loss / 1e6,
            'opex_M': opex / 1e6
        })

    df_res = pd.DataFrame(results)
    best_row = df_res.loc[df_res['net_profit_M'].idxmax()]

    print(f"  Optimal threshold (max profit):  {best_row['threshold']:.3f}")
    print(f"  Net profit at optimal:           ${best_row['net_profit_M']:.1f}M")
    print(f"  Approval rate at optimal:        {best_row['approval_rate']*100:.1f}%")
    
    print(f"\n  Revenue (approved good):         ${best_row['revenue_M']:.1f}M")
    print(f"  Opportunity cost (denied good):  -${best_row['opp_cost_M']:.1f}M")
    print(f"  Loan loss (missed defaults):     -${best_row['loan_loss_M']:.1f}M")
    print(f"  Operating expense:               -${best_row['opex_M']:.1f}M")

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZATION — Xuất đồ thị phân tích ROI
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Curve 1: Threshold vs Profit
    axes[0].plot(df_res['threshold'], df_res['net_profit_M'], color='blue', lw=2.5, label='Net Profit ($M)')
    axes[0].axvline(x=best_row['threshold'], color='red', linestyle='--', label=f'Optimal T={best_row["threshold"]:.3f}')
    axes[0].set_xlabel('Decision Threshold')
    axes[0].set_ylabel('Net Profit ($ Millions)')
    axes[0].set_title('Profit Optimization vs Decision Threshold')
    axes[0].legend(loc='lower center')
    axes[0].grid(alpha=0.3)

    # Curve 2: Threshold vs Approval Rate
    axes[1].plot(df_res['threshold'], df_res['approval_rate'], color='green', lw=2, label='Approval Rate')
    axes[1].set_xlabel('Decision Threshold')
    axes[1].set_ylabel('Portfolio Approval Rate')
    axes[1].set_title('Portfolio Approval Rate Dynamic')
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.suptitle('Business ROI & Decision Threshold Optimization', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{REPORTS_DIR}/roi_threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: reports/roi_threshold_analysis.png")

    # Tạo bảng Confusion Matrix phân rã lợi nhuận
    df_res.to_csv(f'{REPORTS_DIR}/roi_summary.csv', index=False)
    print(f"  Saved: reports/roi_summary.csv")

    return df_res, best_row


if __name__ == '__main__':
    # 1. Đọc dữ liệu kết quả tích hợp từ các bước trước (Mô hình + IFRS9)
    results_path = f'{DATA_DIR}/results_df.parquet'
    if not os.path.exists(results_path):
        print("ERROR: Không tìm thấy results_df.parquet. Vui lòng chạy modeling.py hoặc ifrs9_ecl_engine.py trước.")
        sys.exit(1)
        
    df = pd.read_parquet(results_path)
    
    # 2. Xử lý gộp TOÀN BỘ các trường dữ liệu từ train_features.parquet
    train_feat_path = f'{DATA_DIR}/train_features.parquet'
    if not os.path.exists(train_feat_path):
        print("ERROR: Không tìm thấy train_features.parquet.")
        sys.exit(1)
        
    print("Đang nạp TOÀN BỘ dữ liệu đặc trưng từ train_features.parquet...")
    df_features = pd.read_parquet(train_feat_path)
    
    # Xác định các cột chỉ có ở file train gốc (loại trừ các cột đã có trong file kết quả)
    cols_to_use = [c for c in df_features.columns if c not in df.columns or c == 'SK_ID_CURR']
    
    print("Thực hiện gộp (Merge) toàn phần dữ liệu gốc và kết quả phân tích...")
    df = pd.merge(df, df_features[cols_to_use], on='SK_ID_CURR', how='inner')
    print("✓ Đã gộp thành công tập dữ liệu đặc trưng.")

    # ═══════════════════════════════════════════════════════════════
    # ĐOẠN SỬA ĐỔI THEO CÁCH B: GỘP NGƯỢC CỘT CHỮ TỪ FILE CSV GỐC
    # ═══════════════════════════════════════════════════════════════
    # Đường dẫn đến file csv gốc chứa văn bản chữ (Bạn kiểm tra lại tên file csv của bạn)
    raw_csv_path = f'{DATA_DIR}/application_train.csv' 
    
    if os.path.exists(raw_csv_path):
        print("  >> Phát hiện application_train.csv. Đang tiến hành gộp ngược các trường văn bản chữ...")
        # Chỉ lấy duy nhất các cột chữ cần thiết để tối ưu bộ nhớ
        df_raw_text = pd.read_csv(raw_csv_path, usecols=['SK_ID_CURR', 'OCCUPATION_TYPE', 'NAME_INCOME_TYPE'])
        
        # Merge các cột chữ vào bảng kết quả chính dựa trên mã khách hàng
        df = pd.merge(df, df_raw_text, on='SK_ID_CURR', how='left')
        print("✓ Đã kéo thành công các trường OCCUPATION_TYPE và NAME_INCOME_TYPE vào file Parquet!")
    else:
        print(f"WARNING: Không tìm thấy file csv gốc tại {raw_csv_path}. Vui lòng kiểm tra lại đường dẫn file.")
    # ═══════════════════════════════════════════════════════════════

    if 'PRED_PROB' not in df.columns:
        print("ERROR: results_df.parquet thiếu trường dữ liệu PRED_PROB.")
        sys.exit(1)

    # 3. Chạy thuật toán tối ưu hóa Threshold
    df_roi, best = run_threshold_optimization(df)

    # 4. Gán nhãn phê duyệt và phân dải điểm tín dụng phục vụ Power BI
    print("\n  >> Đồng bộ kết quả tối ưu hóa ROI vào tệp lưu trữ...")
    df['Decision_Status'] = np.where(df['PRED_PROB'] < best['threshold'], 'Approved', 'Denied')
    
    unique_prob_count = len(df['PRED_PROB'].unique())
    target_labels = ['Very Low Risk', 'Low Risk', 'Medium Risk', 'High Risk']
    
    df['Credit_Band'] = pd.qcut(df['PRED_PROB'], q=4, 
                                labels=target_labels[:unique_prob_count] if unique_prob_count < 4 else target_labels,
                                duplicates='drop')
    
    # 5. Thực hiện ghi đè lưu trữ cuối cùng
    df.to_parquet(results_path, index=False)
    print(f"✓ HOÀN TẤT PIPELINE TỔNG HỢP THEO PHƯƠNG ÁN B!")