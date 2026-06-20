-- =========================================================================================
-- HỆ THỐNG TRUY VẤN DỮ LIỆU RỦI RO TÍN DỤNG DÀNH CHO CÁC PHÒNG BAN (BUSINESS USE CASES)
-- =========================================================================================
USE HomeCredit;
GO

-- -----------------------------------------------------------------------------------------
-- 1. BỘ PHẬN THU HỒI NỢ (COLLECTIONS DEPARTMENT)
-- Mục đích: Lấy danh sách khách hàng có rủi ro vỡ nợ cao nhất để ưu tiên gọi điện nhắc nhở sớm.
-- -----------------------------------------------------------------------------------------
SELECT TOP 1000
    p.SK_ID_CURR AS Ma_Khach_Hang,
    a.CODE_GENDER AS Gioi_Tinh,
    a.NAME_INCOME_TYPE AS Nguon_Thu_Nhap,
    a.AMT_INCOME_TOTAL AS Thu_Nhap_Hang_Thang,
    a.AMT_CREDIT AS Tong_Tien_Vay,
    ROUND(p.DEFAULT_PROBABILITY * 100, 2) AS Xac_Suat_Vo_No_Phan_Tram,
    p.RISK_TIER AS Phan_Loai_Rui_Ro,
    ROUND(p.ECL, 2) AS Du_Kien_Thiet_Hai_USD,
    -- Kaggle ẩn thông tin cá nhân (PII). Ta dùng Mock Data để minh họa thực tế:
    CONCAT('09', RIGHT(CAST(CAST(p.SK_ID_CURR AS BIGINT) * 1234567 AS VARCHAR(20)), 8)) AS So_Dien_Thoai_Lien_He,
    CONCAT('khachhang_', p.SK_ID_CURR, '@gmail.com') AS Email_Lien_He
FROM model.Predictions p
JOIN raw.Application a ON p.SK_ID_CURR = a.SK_ID_CURR
WHERE p.RISK_TIER = 'High' AND p.DEFAULT_PROBABILITY > 0.48
ORDER BY p.DEFAULT_PROBABILITY DESC;
GO

-- -----------------------------------------------------------------------------------------
-- 2. BỘ PHẬN PHÊ DUYỆT TÍN DỤNG (UNDERWRITING / CREDIT APPROVAL)
-- Mục đích: Tự động phê duyệt (Auto-Approve) cho các khách hàng có rủi ro cực thấp (< 15%).
-- -----------------------------------------------------------------------------------------
SELECT 
    p.SK_ID_CURR AS Ma_Khach_Hang,
    a.AMT_CREDIT AS Khoan_Vay_Yeu_Cau,
    a.AMT_INCOME_TOTAL AS Thu_Nhap_Chung_Minh,
    ROUND(p.DEFAULT_PROBABILITY * 100, 2) AS Ty_Le_Rui_Ro,
    'AUTO-APPROVE' AS Quyet_Dinh_He_Thong,
    GETDATE() AS Thoi_Gian_Duyet
FROM model.Predictions p
JOIN raw.Application a ON p.SK_ID_CURR = a.SK_ID_CURR
WHERE p.DEFAULT_PROBABILITY < 0.15 
  AND p.RISK_TIER IN ('Very Low', 'Low')
ORDER BY p.DEFAULT_PROBABILITY ASC;
GO

-- -----------------------------------------------------------------------------------------
-- 3. BỘ PHẬN QUẢN TRỊ RỦI RO & TÀI CHÍNH (RISK MANAGEMENT & FINANCE)
-- Mục đích: Báo cáo tổng số tiền cần trích lập dự phòng theo từng Nhóm Nợ (Chuẩn IFRS 9).
-- -----------------------------------------------------------------------------------------
SELECT 
    p.Stage AS Nhom_No_IFRS9,
    COUNT(p.SK_ID_CURR) AS Tong_So_Khoan_Vay,
    SUM(p.EAD) AS Tong_Du_No_Chi_Chiu_Rui_Ro_EAD,
    SUM(p.ECL) AS Tong_Tien_Trich_Lap_Du_Phong_ECL,
    ROUND(SUM(p.ECL) / SUM(p.EAD) * 100, 2) AS Ty_Le_Bao_Phu_No_Xau_Phan_Tram
FROM model.Predictions p
WHERE p.Stage IS NOT NULL
GROUP BY p.Stage
ORDER BY p.Stage;
GO

-- -----------------------------------------------------------------------------------------
-- 4. BỘ PHẬN KINH DOANH & MARKETING (SALE & MARKETING)
-- Mục đích: Tìm tập khách hàng VIP (Thu nhập cao, Rủi ro thấp) để chạy quảng cáo Upsell Thẻ Tín Dụng.
-- -----------------------------------------------------------------------------------------
SELECT 
    p.SK_ID_CURR AS Ma_Khach_Hang,
    a.AMT_INCOME_TOTAL AS Thu_Nhap_Hien_Tai,
    a.NAME_INCOME_TYPE AS Loai_Hinh_Cong_Viec,
    ROUND(p.DEFAULT_PROBABILITY * 100, 2) AS Xac_Suat_Vo_No,
    'Telesale - Mo The Tin Dung' AS Chien_Dich_Marketing
FROM model.Predictions p
JOIN raw.Application a ON p.SK_ID_CURR = a.SK_ID_CURR
WHERE p.DEFAULT_PROBABILITY < 0.05 -- Rủi ro dưới 5%
  AND a.AMT_INCOME_TOTAL > 200000  -- Thu nhập thuộc nhóm cao
ORDER BY a.AMT_INCOME_TOTAL DESC;
GO
