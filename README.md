# 🏦 Enterprise Credit Risk Scoring & ECL Provisioning System

> **End-to-end Machine Learning pipeline** dự đoán xác suất vỡ nợ (Probability of Default), tính toán Expected Credit Loss theo **IFRS 9**, và cung cấp actionable insights cho **307,511** hồ sơ vay tiêu dùng — tích hợp kiến trúc **Multi-Model Benchmark** và tối ưu hóa lợi nhuận ngân hàng.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.x-02569B?logo=lightgbm)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.x-EC522A?logo=xgboost&logoColor=white)](https://xgboost.ai)
[![CatBoost](https://img.shields.io/badge/CatBoost-1.x-FF6F00?logo=catboost&logoColor=white)](https://catboost.ai)
[![MLflow](https://img.shields.io/badge/MLOps-MLflow-0194E2?logo=mlflow)](https://mlflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📑 Mục Lục
1. [Kiến Trúc Multi-Model Benchmark](#-kiến-trúc-multi-model-benchmark)
2. [Đóng Góp Kỹ Thuật & Khung Nghiệp Vụ](#-đóng-góp-kỹ-thuật--khung-nghiệp-vụ)
3. [Hiệu Năng Mô Hình (Model Performance)](#-hiệu-năng-mô-hình)
4. [Pipeline Logic](#-pipeline-logic)
5. [Hướng Dẫn Chạy Dự Án](#️-hướng-dẫn-chạy-dự-án)
6. [Cấu Trúc Thư Mục](#-cấu-trúc-thư-mục)

---

## 🧠 Kiến Trúc Multi-Model Benchmark
Hệ thống triển khai kiến trúc **Multi-Model Benchmark** giúp tối đa hóa năng lực phân tách rủi ro trên tập dữ liệu imbalanced:
* **Ensemble Strategy:** So sánh song song sức mạnh của LightGBM, XGBoost, CatBoost và Random Forest thông qua cơ chế 5-Fold Stratified Cross-Validation.
* **K-Fold Isolation:** Cơ chế quản lý bộ nhớ thông minh, cô lập tài nguyên từng fold huấn luyện giúp tránh lỗi tràn RAM trên máy trạm cá nhân.
* **Isotonic Calibration:** Hiệu chuẩn xác suất dự báo (Probability Calibration) dựa trên tập OOF (Out-Of-Fold) để giảm thiểu Brier Score và chống rò rỉ dữ liệu (Data Leakage).



---

## 🛠 Đóng Góp Kỹ Thuật & Khung Nghiệp Vụ
| Hạng Mục | Mô Tả |
|---|---|
| **IFRS 9 Engine** | Tính toán ECL dựa trên PD Term Structure (Lifetime PD) và áp dụng 4 kịch bản vĩ mô (Optimistic, Base, Adverse, Severe). |
| **Stage Classification** | Phân loại hồ sơ tín dụng theo Stage 1, 2, 3 dựa trên PD thresholds và các chỉ số hành vi (late ratio, bureau debt). |
| **ROI Optimization** | Thuật toán quét ngưỡng quyết định (`Optimal Threshold`) để tối đa hóa Net Profit, cân bằng giữa lợi nhuận biên và chi phí rủi ro. |
| **Model Tracking** | Tích hợp MLflow để theo dõi AUC, KS, Gini và xuất các biểu đồ kiểm thử sức căng (Stress Test). |

## 📈 Hiệu Năng Mô Hình (Model Performance)
Hệ thống sử dụng các chỉ số đo lường quy chuẩn của ngành quản trị rủi ro tài chính ngân hàng để đánh giá và so sánh toàn diện năng lực phân tách nhóm Tốt/Xấu của danh mục đầu tư:

| Mô hình (Model) | OOF_AUC | Gini_Coefficient | KS_Statistic | Brier_Score | Trạng thái (Status) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Random Forest** | 0.7592 | 0.5184 | 38.09% | 0.06766 | Benchmark |
| **CatBoost** | 0.7847 | 0.5694 | 42.81% | 0.06595 | Contender |
| **LightGBM** | 0.7856 | 0.5711 | 42.83% | 0.06598 | Contender |
| **XGBoost** | **0.7869** | **0.5739** | **42.85%** | **0.06584** | 🏆 **Champion** |

*Ghi chú: Kết quả trên được trích xuất tự động từ luồng thực nghiệm K-Fold và đồng bộ chính xác với tập tin báo cáo hiệu năng `reports/model_comparison_leaderboard.csv`.*

---

## 🔬 Pipeline Logic
* **`modeling.py`**: Hub trung tâm thực hiện huấn luyện chéo 5-fold, benchmark 4 thuật toán, hiệu chuẩn xác suất và xuất `results_df.parquet`.
* **`ifrs9_ecl_engine.py`**: Nạp dữ liệu dự báo, áp dụng logic IFRS 9 để trích lập dự phòng theo stage và kịch bản kinh tế vĩ mô.
* **`business_roi_analysis.py`**: Kết hợp với `AMT_CREDIT` để tính toán bài toán P&L tài chính, tìm ngưỡng phê duyệt tối ưu.

---

## ⚙️ Hướng Dẫn Chạy Dự Án
Thực hiện tuần tự các bước để hoàn thiện Pipeline tài chính:
```bash
# 1. Huấn luyện Multi-Model Benchmark và Calibration
python src/modeling.py

# 2. Tính toán trích lập dự phòng IFRS 9
python src/ifrs9_ecl_engine.py

# 3. Phân tích ROI và tìm ngưỡng Threshold tối ưu
python src/business_roi_analysis.py
```
## 📂 Cấu Trúc Thư Mục
Enterprise-Credit-Risk-Scoring/
│
├── 🤖 models/                            # Kho lưu trữ các file nhị phân sau huấn luyện
│   ├── lgbm_fold1-5.pkl                  # 5 Mô hình thành phần LightGBM Folds
│   ├── xgboost_fold1-5.pkl               # 5 Mô hình thành phần XGBoost Folds
│   ├── catboost_fold1-5.pkl              # 5 Mô hình thành phần CatBoost Folds 
│   ├── best_production_model.pkl         # Mô hình xuất sắc nhất (Winner) đại diện cho hệ thống
│   ├── isotonic_calibrator.pkl           # Bộ hiệu chuẩn xác suất (Calibration) chống rò rỉ dữ liệu
│   ├── feature_list.pkl                  # Danh sách toàn bộ các đặc trưng (Features) đưa vào mô hình
│   └── model_metrics.json                # File lưu trữ metadata hiệu năng (AUC, Gini, KS, Brier) dạng JSON
│
├── 📈 reports/                           # Tài liệu, chiến lợi phẩm thực nghiệm phục vụ viết báo cáo/slide
│   ├── model_comparison_leaderboard.csv  # Bảng xếp hạng so sánh đối chứng chi tiết KPI các mô hình
│   └── model_benchmark_roc.png           # Đồ thị so sánh đường cong ROC Out-Of-Fold của 4 thuật toán
│
├── 🐍 src/                               # Mã nguồn xử lý các cấu phần của đường ống (Pipeline)
│   ├── modeling.py                       # Script xử lý K-Fold Isolation, Benchmarking & Calibration
│   ├── ifrs9_ecl_engine.py               # Engine trích lập tổn thất dòng tiền theo chuẩn IFRS 9
│   └── business_roi_analysis.py          # Module thuật toán quét Threshold tối ưu lợi nhuận ROI doanh nghiệp
│
└── 📊 data/                              # Thư mục quản lý dữ liệu của dự án
    ├── train_features.parquet            # Tập đặc trưng huấn luyện đầu vào
    ├── test_features.parquet             # Tập đặc trưng kiểm thử đầu vào
    ├── results_df.parquet                # Kết quả dự báo xác suất (PRED_PROB) phục vụ ECL/ROI
    └── submission.csv                    # Kết quả dự đoán định dạng nộp Kaggle của mô hình tốt nhất
```