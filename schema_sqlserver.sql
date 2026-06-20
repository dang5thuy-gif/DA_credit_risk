-- ============================================================
-- Home Credit Default Risk — SQL Server Schema
-- Database : HomeCredit
-- Created  : 2026
-- Purpose  : Import all 8 cleaned source files for SQL analysis
--            + reporting + BI dashboarding
-- Notes    :
--   - All DAYS_ columns store raw negative values (days before application)
--   - After cleaning, DAYS_EMPLOYED no longer contains 365243
--   - EMP_ANOMALY is an engineered column added during cleaning
--   - UTILIZATION_RATIO is engineered in credit_card_balance (cleaned)
--   - Run sections in ORDER (FK dependencies)
-- ============================================================

USE master;
GO

-- ── Drop & recreate database ──────────────────────────────────────────────────
IF EXISTS (SELECT name FROM sys.databases WHERE name = 'HomeCredit')
BEGIN
    ALTER DATABASE HomeCredit SET SINGLE_USER WITH ROLLBACK IMMEDIATE;
    DROP DATABASE HomeCredit;
END
GO

CREATE DATABASE HomeCredit
    COLLATE Vietnamese_CI_AS;   -- supports Vietnamese characters in future reports
GO

USE HomeCredit;
GO


-- ============================================================
-- SCHEMA SETUP
-- ============================================================
IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'raw')
    EXEC('CREATE SCHEMA raw');   -- original source tables (cleaned)

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'feat')
    EXEC('CREATE SCHEMA feat');  -- feature-engineered aggregation tables (Phase 2)

IF NOT EXISTS (SELECT * FROM sys.schemas WHERE name = 'model')
    EXEC('CREATE SCHEMA model'); -- model results, predictions, risk tiers
GO


-- ============================================================
-- TABLE 1: raw.Application
-- Source  : application_train_clean.parquet
--           application_test_clean.parquet  (IS_TEST = 1)
-- Grain   : 1 row per applicant
-- PK      : SK_ID_CURR
-- Rows    : ~307,511 train + 48,744 test
-- ============================================================
IF OBJECT_ID('raw.Application', 'U') IS NOT NULL DROP TABLE raw.Application;
GO

CREATE TABLE raw.Application (

    -- ── Identity ────────────────────────────────────────────
    SK_ID_CURR              INT             NOT NULL,   -- applicant unique ID
    IS_TEST                 TINYINT         NOT NULL DEFAULT 0,  -- 0=train, 1=test
    TARGET                  TINYINT         NULL,       -- 1=default, 0=repaid, NULL for test

    -- ── Loan characteristics ────────────────────────────────
    NAME_CONTRACT_TYPE      NVARCHAR(30)    NOT NULL,   -- 'Cash loans' | 'Revolving loans'
    AMT_CREDIT              DECIMAL(18,2)   NOT NULL,   -- total loan amount
    AMT_ANNUITY             DECIMAL(18,2)   NULL,       -- monthly payment
    AMT_GOODS_PRICE         DECIMAL(18,2)   NULL,       -- goods price financed
    AMT_INCOME_TOTAL        DECIMAL(18,2)   NOT NULL,   -- annual income (capped at p99)

    -- ── Demographics ────────────────────────────────────────
    CODE_GENDER             NVARCHAR(10)    NOT NULL,   -- 'M' | 'F' | 'Unknown'
    FLAG_OWN_CAR            CHAR(1)         NOT NULL,   -- 'Y' | 'N'
    FLAG_OWN_REALTY         CHAR(1)         NOT NULL,   -- 'Y' | 'N'
    CNT_CHILDREN            SMALLINT        NOT NULL,
    CNT_FAM_MEMBERS         SMALLINT        NULL,

    -- ── Education, employment, housing ──────────────────────
    NAME_INCOME_TYPE        NVARCHAR(50)    NOT NULL,
    NAME_EDUCATION_TYPE     NVARCHAR(60)    NOT NULL,
    NAME_FAMILY_STATUS      NVARCHAR(30)    NOT NULL,
    NAME_HOUSING_TYPE       NVARCHAR(30)    NOT NULL,
    OCCUPATION_TYPE         NVARCHAR(50)    NULL,       -- 31.3% missing in source
    ORGANIZATION_TYPE       NVARCHAR(60)    NOT NULL,

    -- ── Time features (days before application date) ────────
    DAYS_BIRTH              INT             NOT NULL,   -- negative, e.g. -14000
    DAYS_EMPLOYED           INT             NULL,       -- NULL after cleaning (was 365243)
    DAYS_REGISTRATION       DECIMAL(10,2)   NULL,
    DAYS_ID_PUBLISH         INT             NOT NULL,   -- days since ID published
    DAYS_LAST_PHONE_CHANGE  DECIMAL(10,2)   NULL,
    OWN_CAR_AGE             DECIMAL(6,2)    NULL,

    -- ── Engineered anomaly flag (added during cleaning) ─────
    EMP_ANOMALY             TINYINT         NOT NULL DEFAULT 0,  -- 1 = was unemployed

    -- ── External credit scores (strongest predictors) ────────
    EXT_SOURCE_1            DECIMAL(8,6)    NULL,       -- 56.4% filled with median
    EXT_SOURCE_2            DECIMAL(8,6)    NULL,
    EXT_SOURCE_3            DECIMAL(8,6)    NULL,

    -- ── Contact & regional flags ─────────────────────────────
    FLAG_MOBIL              TINYINT         NOT NULL DEFAULT 0,
    FLAG_EMP_PHONE          TINYINT         NOT NULL DEFAULT 0,
    FLAG_WORK_PHONE         TINYINT         NOT NULL DEFAULT 0,
    FLAG_CONT_MOBILE        TINYINT         NOT NULL DEFAULT 0,
    FLAG_PHONE              TINYINT         NOT NULL DEFAULT 0,
    FLAG_EMAIL              TINYINT         NOT NULL DEFAULT 0,

    REGION_RATING_CLIENT            TINYINT         NULL,
    REGION_RATING_CLIENT_W_CITY     TINYINT         NULL,
    REGION_POPULATION_RELATIVE      DECIMAL(10,8)   NULL,
    REG_REGION_NOT_LIVE_REGION      TINYINT         NULL,
    REG_REGION_NOT_WORK_REGION      TINYINT         NULL,
    LIVE_REGION_NOT_WORK_REGION     TINYINT         NULL,
    REG_CITY_NOT_LIVE_CITY          TINYINT         NULL,
    REG_CITY_NOT_WORK_CITY          TINYINT         NULL,
    LIVE_CITY_NOT_WORK_CITY         TINYINT         NULL,

    -- ── Application process ──────────────────────────────────
    WEEKDAY_APPR_PROCESS_START  NVARCHAR(15)    NULL,
    HOUR_APPR_PROCESS_START     TINYINT         NULL,
    NAME_TYPE_SUITE             NVARCHAR(30)    NULL,

    -- ── Social circle (capped at 20/100 during cleaning) ────
    OBS_30_CNT_SOCIAL_CIRCLE    SMALLINT        NULL,
    DEF_30_CNT_SOCIAL_CIRCLE    TINYINT         NULL,
    OBS_60_CNT_SOCIAL_CIRCLE    SMALLINT        NULL,
    DEF_60_CNT_SOCIAL_CIRCLE    TINYINT         NULL,

    -- ── Credit bureau inquiry counts ─────────────────────────
    AMT_REQ_CREDIT_BUREAU_HOUR  TINYINT         NULL,
    AMT_REQ_CREDIT_BUREAU_DAY   TINYINT         NULL,
    AMT_REQ_CREDIT_BUREAU_WEEK  TINYINT         NULL,
    AMT_REQ_CREDIT_BUREAU_MON   TINYINT         NULL,
    AMT_REQ_CREDIT_BUREAU_QRT   TINYINT         NULL,   -- capped at 12
    AMT_REQ_CREDIT_BUREAU_YEAR  TINYINT         NULL,

    -- ── Document flags (20 binary flags) ─────────────────────
    FLAG_DOCUMENT_2     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_3     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_4     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_5     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_6     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_7     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_8     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_9     TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_10    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_11    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_12    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_13    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_14    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_15    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_16    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_17    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_18    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_19    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_20    TINYINT NOT NULL DEFAULT 0,
    FLAG_DOCUMENT_21    TINYINT NOT NULL DEFAULT 0,

    -- ── Metadata ─────────────────────────────────────────────
    CREATED_AT          DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_Application PRIMARY KEY CLUSTERED (SK_ID_CURR)
);
GO

-- Indexes for common query patterns
CREATE NONCLUSTERED INDEX IX_App_Target        ON raw.Application (TARGET)      INCLUDE (SK_ID_CURR, AMT_CREDIT, EXT_SOURCE_2);
CREATE NONCLUSTERED INDEX IX_App_Gender        ON raw.Application (CODE_GENDER) INCLUDE (TARGET, AMT_CREDIT);
CREATE NONCLUSTERED INDEX IX_App_IncomeType    ON raw.Application (NAME_INCOME_TYPE) INCLUDE (TARGET);
CREATE NONCLUSTERED INDEX IX_App_OccupationType ON raw.Application (OCCUPATION_TYPE) INCLUDE (TARGET);
GO


-- ============================================================
-- TABLE 2: raw.Bureau
-- Source  : bureau_clean.parquet
-- Grain   : 1 row per external credit bureau entry
-- FK      : SK_ID_CURR -> raw.Application
-- Rows    : ~1,716,428
-- ============================================================
IF OBJECT_ID('raw.Bureau', 'U') IS NOT NULL DROP TABLE raw.Bureau;
GO

CREATE TABLE raw.Bureau (

    SK_ID_BUREAU            INT             NOT NULL,   -- bureau loan unique ID
    SK_ID_CURR              INT             NOT NULL,   -- FK to Application

    CREDIT_ACTIVE           NVARCHAR(20)    NOT NULL,   -- 'Active' | 'Closed' | 'Sold' | 'Bad debt'
    CREDIT_CURRENCY         NVARCHAR(10)    NULL,
    CREDIT_TYPE             NVARCHAR(60)    NOT NULL,   -- 14 types (Consumer, Credit card, ...)

    -- ── Days (all negative = past, 0 = application date) ────
    DAYS_CREDIT             INT             NOT NULL,   -- when bureau credit was applied
    DAYS_CREDIT_ENDDATE     INT             NULL,       -- planned end date (can be positive = future)
    DAYS_ENDDATE_FACT       INT             NOT NULL DEFAULT 0,  -- actual end (0 = still active)
    DAYS_CREDIT_UPDATE      INT             NOT NULL,   -- last update

    -- ── Overdue ──────────────────────────────────────────────
    CREDIT_DAY_OVERDUE      SMALLINT        NOT NULL DEFAULT 0,  -- capped at 365
    AMT_CREDIT_MAX_OVERDUE  DECIMAL(18,2)   NOT NULL DEFAULT 0,  -- filled 0 if no overdue

    -- ── Amounts ──────────────────────────────────────────────
    AMT_CREDIT_SUM          DECIMAL(18,2)   NULL,       -- total credit amount
    AMT_CREDIT_SUM_DEBT     DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_CREDIT_SUM_LIMIT    DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_CREDIT_SUM_OVERDUE  DECIMAL(18,2)   NOT NULL DEFAULT 0,
    CNT_CREDIT_PROLONG      SMALLINT        NOT NULL DEFAULT 0,  -- # times extended

    CREATED_AT              DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_Bureau PRIMARY KEY CLUSTERED (SK_ID_BUREAU),
    CONSTRAINT FK_Bureau_Application
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO

CREATE NONCLUSTERED INDEX IX_Bureau_Client    ON raw.Bureau (SK_ID_CURR) INCLUDE (CREDIT_ACTIVE, AMT_CREDIT_SUM, CREDIT_DAY_OVERDUE);
CREATE NONCLUSTERED INDEX IX_Bureau_Active    ON raw.Bureau (CREDIT_ACTIVE);
GO


-- ============================================================
-- TABLE 3: raw.PreviousApplication
-- Source  : previous_application_clean.parquet
-- Grain   : 1 row per previous loan application at Home Credit
-- FK      : SK_ID_CURR -> raw.Application
-- Rows    : ~1,670,214
-- ============================================================
IF OBJECT_ID('raw.PreviousApplication', 'U') IS NOT NULL DROP TABLE raw.PreviousApplication;
GO

CREATE TABLE raw.PreviousApplication (

    SK_ID_PREV              INT             NOT NULL,
    SK_ID_CURR              INT             NOT NULL,

    -- ── Contract info ────────────────────────────────────────
    NAME_CONTRACT_TYPE      NVARCHAR(30)    NULL,
    NAME_CONTRACT_STATUS    NVARCHAR(20)    NOT NULL,  -- 'Approved'|'Canceled'|'Refused'|'Unused offer'
    NAME_CASH_LOAN_PURPOSE  NVARCHAR(50)    NULL,
    NAME_PAYMENT_TYPE       NVARCHAR(50)    NULL,
    CODE_REJECT_REASON      NVARCHAR(30)    NULL,
    NAME_CLIENT_TYPE        NVARCHAR(30)    NULL,
    NAME_GOODS_CATEGORY     NVARCHAR(50)    NULL,
    NAME_PORTFOLIO          NVARCHAR(30)    NULL,
    NAME_PRODUCT_TYPE       NVARCHAR(30)    NULL,
    CHANNEL_TYPE            NVARCHAR(50)    NULL,
    NAME_SELLER_INDUSTRY    NVARCHAR(50)    NULL,
    NAME_YIELD_GROUP        NVARCHAR(30)    NULL,
    PRODUCT_COMBINATION     NVARCHAR(60)    NULL,
    FLAG_LAST_APPL_PER_CONTRACT NVARCHAR(5) NULL,
    NFLAG_LAST_APPL_IN_DAY  TINYINT         NULL,
    NFLAG_INSURED_ON_APPROVAL TINYINT       NULL,

    -- ── Amounts ──────────────────────────────────────────────
    AMT_ANNUITY             DECIMAL(18,2)   NULL,
    AMT_APPLICATION         DECIMAL(18,2)   NULL,       -- originally requested
    AMT_CREDIT              DECIMAL(18,2)   NULL,       -- actually approved
    AMT_DOWN_PAYMENT        DECIMAL(18,2)   NOT NULL DEFAULT 0,  -- clipped >= 0
    AMT_GOODS_PRICE         DECIMAL(18,2)   NULL,
    RATE_DOWN_PAYMENT       DECIMAL(8,6)    NULL,

    -- ── Days ─────────────────────────────────────────────────
    DAYS_DECISION           INT             NOT NULL,   -- days before current app
    DAYS_FIRST_DRAWING      INT             NULL,       -- anomaly 365243 -> NaN -> imputed
    DAYS_FIRST_DUE          INT             NULL,
    DAYS_LAST_DUE_1ST_VERSION INT           NULL,
    DAYS_LAST_DUE           INT             NULL,
    DAYS_TERMINATION        INT             NULL,

    -- ── Other ─────────────────────────────────────────────────
    WEEKDAY_APPR_PROCESS_START NVARCHAR(15) NULL,
    HOUR_APPR_PROCESS_START TINYINT         NULL,
    SELLERPLACE_AREA        INT             NULL,
    CNT_PAYMENT             SMALLINT        NULL,

    CREATED_AT              DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_PreviousApplication PRIMARY KEY CLUSTERED (SK_ID_PREV),
    CONSTRAINT FK_PrevApp_Application
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO

CREATE NONCLUSTERED INDEX IX_PrevApp_Client  ON raw.PreviousApplication (SK_ID_CURR) INCLUDE (NAME_CONTRACT_STATUS, AMT_CREDIT, DAYS_DECISION);
CREATE NONCLUSTERED INDEX IX_PrevApp_Status  ON raw.PreviousApplication (NAME_CONTRACT_STATUS);
GO


-- ============================================================
-- TABLE 4: raw.InstallmentsPayments
-- Source  : installments_payments_clean.parquet
-- Grain   : 1 row per scheduled installment payment
-- FK      : SK_ID_CURR -> raw.Application
-- Rows    : ~13,605,401
-- Engineered: PAYMENT_DIFF, DAYS_PAST_DUE
-- ============================================================
IF OBJECT_ID('raw.InstallmentsPayments', 'U') IS NOT NULL DROP TABLE raw.InstallmentsPayments;
GO

CREATE TABLE raw.InstallmentsPayments (

    SK_ID_PREV                  INT             NOT NULL,
    SK_ID_CURR                  INT             NOT NULL,
    NUM_INSTALMENT_VERSION      TINYINT         NULL,
    NUM_INSTALMENT_NUMBER       SMALLINT        NOT NULL,

    -- ── Scheduled vs actual payment ──────────────────────────
    DAYS_INSTALMENT             DECIMAL(10,2)   NULL,   -- scheduled payment date
    DAYS_ENTRY_PAYMENT          DECIMAL(10,2)   NULL,   -- actual payment date
    AMT_INSTALMENT              DECIMAL(14,4)   NULL,   -- scheduled amount
    AMT_PAYMENT                 DECIMAL(14,4)   NULL,   -- actual amount paid

    -- ── Engineered (added during cleaning) ───────────────────
    PAYMENT_DIFF                DECIMAL(14,4)   NULL,   -- AMT_INSTALMENT - AMT_PAYMENT (>0 = underpaid)
    DAYS_PAST_DUE               DECIMAL(10,2)   NULL,   -- clipped >= 0 (0 = on time or early)

    CREATED_AT                  DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT FK_Inst_Application
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO

-- No PK (composite key would be SK_ID_PREV + NUM_INSTALMENT_NUMBER but not guaranteed unique)
CREATE NONCLUSTERED INDEX IX_Inst_Client  ON raw.InstallmentsPayments (SK_ID_CURR) INCLUDE (AMT_PAYMENT, DAYS_PAST_DUE, PAYMENT_DIFF);
CREATE NONCLUSTERED INDEX IX_Inst_Prev    ON raw.InstallmentsPayments (SK_ID_PREV);
GO


-- ============================================================
-- TABLE 5: raw.POSCashBalance
-- Source  : POS_CASH_balance_clean.parquet
-- Grain   : 1 row per loan per month (monthly snapshot)
-- FK      : SK_ID_CURR -> raw.Application
-- Rows    : ~10,001,358
-- ============================================================
IF OBJECT_ID('raw.POSCashBalance', 'U') IS NOT NULL DROP TABLE raw.POSCashBalance;
GO

CREATE TABLE raw.POSCashBalance (

    SK_ID_PREV              INT             NOT NULL,
    SK_ID_CURR              INT             NOT NULL,
    MONTHS_BALANCE          SMALLINT        NOT NULL,   -- negative = months before snapshot

    CNT_INSTALMENT          DECIMAL(6,2)    NULL,       -- total installments in loan
    CNT_INSTALMENT_FUTURE   DECIMAL(6,2)    NULL,       -- remaining installments

    NAME_CONTRACT_STATUS    NVARCHAR(30)    NOT NULL,   -- 'Active'|'Completed'|'Signed'|...

    -- ── Delinquency (capped at 365) ──────────────────────────
    SK_DPD                  SMALLINT        NOT NULL DEFAULT 0,
    SK_DPD_DEF              SMALLINT        NOT NULL DEFAULT 0,

    CREATED_AT              DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT FK_POS_Application
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO

CREATE NONCLUSTERED INDEX IX_POS_Client  ON raw.POSCashBalance (SK_ID_CURR) INCLUDE (SK_DPD, NAME_CONTRACT_STATUS, MONTHS_BALANCE);
CREATE NONCLUSTERED INDEX IX_POS_Prev    ON raw.POSCashBalance (SK_ID_PREV, MONTHS_BALANCE);
GO


-- ============================================================
-- TABLE 6: raw.CreditCardBalance
-- Source  : credit_card_balance_clean.parquet
-- Grain   : 1 row per credit card per month (monthly snapshot)
-- FK      : SK_ID_CURR -> raw.Application
-- Rows    : ~3,840,312  (only 33.7% of applicants have CC)
-- Engineered: UTILIZATION_RATIO
-- ============================================================
IF OBJECT_ID('raw.CreditCardBalance', 'U') IS NOT NULL DROP TABLE raw.CreditCardBalance;
GO

CREATE TABLE raw.CreditCardBalance (

    SK_ID_PREV                  INT             NOT NULL,
    SK_ID_CURR                  INT             NOT NULL,
    MONTHS_BALANCE              SMALLINT        NOT NULL,

    -- ── Balance & limit ──────────────────────────────────────
    AMT_BALANCE                 DECIMAL(18,2)   NOT NULL DEFAULT 0,  -- can be negative (credit)
    AMT_CREDIT_LIMIT_ACTUAL     INT             NOT NULL DEFAULT 0,

    -- ── Drawings (0 = no activity that month) ────────────────
    AMT_DRAWINGS_ATM_CURRENT    DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_DRAWINGS_CURRENT        DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_DRAWINGS_OTHER_CURRENT  DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_DRAWINGS_POS_CURRENT    DECIMAL(18,2)   NOT NULL DEFAULT 0,
    CNT_DRAWINGS_ATM_CURRENT    DECIMAL(8,2)    NOT NULL DEFAULT 0,
    CNT_DRAWINGS_CURRENT        SMALLINT        NOT NULL DEFAULT 0,
    CNT_DRAWINGS_OTHER_CURRENT  DECIMAL(8,2)    NOT NULL DEFAULT 0,
    CNT_DRAWINGS_POS_CURRENT    DECIMAL(8,2)    NOT NULL DEFAULT 0,

    -- ── Payments ─────────────────────────────────────────────
    AMT_INST_MIN_REGULARITY     DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_PAYMENT_CURRENT         DECIMAL(18,2)   NOT NULL DEFAULT 0,
    AMT_PAYMENT_TOTAL_CURRENT   DECIMAL(18,2)   NULL,

    -- ── Receivables ──────────────────────────────────────────
    AMT_RECEIVABLE_PRINCIPAL    DECIMAL(18,2)   NULL,
    AMT_RECIVABLE               DECIMAL(18,2)   NULL,   -- note: typo in source data
    AMT_TOTAL_RECEIVABLE        DECIMAL(18,2)   NULL,
    CNT_INSTALMENT_MATURE_CUM   DECIMAL(8,2)    NOT NULL DEFAULT 0,

    NAME_CONTRACT_STATUS        NVARCHAR(30)    NULL,

    -- ── Delinquency (capped at 365) ──────────────────────────
    SK_DPD                      SMALLINT        NOT NULL DEFAULT 0,
    SK_DPD_DEF                  SMALLINT        NOT NULL DEFAULT 0,

    -- ── Engineered (added during cleaning) ───────────────────
    UTILIZATION_RATIO           DECIMAL(8,6)    NOT NULL DEFAULT 0,
    -- = AMT_BALANCE / AMT_CREDIT_LIMIT_ACTUAL, capped at 2.0
    -- NULL when limit=0, then filled 0

    CREATED_AT                  DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT FK_CC_Application
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO

CREATE NONCLUSTERED INDEX IX_CC_Client  ON raw.CreditCardBalance (SK_ID_CURR) INCLUDE (AMT_BALANCE, UTILIZATION_RATIO, SK_DPD, MONTHS_BALANCE);
CREATE NONCLUSTERED INDEX IX_CC_Prev    ON raw.CreditCardBalance (SK_ID_PREV, MONTHS_BALANCE);
GO


-- ============================================================
-- FEATURE TABLES (Phase 2 output — populated after Python runs)
-- ============================================================

-- feat.ApplicationFeatures: one row per applicant with all engineered features
IF OBJECT_ID('feat.ApplicationFeatures', 'U') IS NOT NULL DROP TABLE feat.ApplicationFeatures;
GO

CREATE TABLE feat.ApplicationFeatures (

    SK_ID_CURR              INT             NOT NULL,

    -- ── Financial ratios ─────────────────────────────────────
    CREDIT_INCOME_RATIO     DECIMAL(10,4)   NULL,   -- AMT_CREDIT / AMT_INCOME_TOTAL
    ANNUITY_INCOME_RATIO    DECIMAL(10,4)   NULL,   -- AMT_ANNUITY / AMT_INCOME_TOTAL
    CREDIT_TERM             DECIMAL(10,4)   NULL,   -- AMT_CREDIT / AMT_ANNUITY (months)
    GOODS_CREDIT_RATIO      DECIMAL(10,4)   NULL,   -- AMT_GOODS_PRICE / AMT_CREDIT
    CREDIT_DOWNPAYMENT      DECIMAL(18,2)   NULL,   -- AMT_GOODS_PRICE - AMT_CREDIT

    -- ── Time-based ───────────────────────────────────────────
    AGE_YEARS               DECIMAL(6,2)    NULL,
    YEARS_EMPLOYED          DECIMAL(6,2)    NULL,
    DAYS_EMPLOYED_RATIO     DECIMAL(10,6)   NULL,
    YEARS_ID_PUBLISH        DECIMAL(6,2)    NULL,
    YEARS_LAST_PHONE_CHANGE DECIMAL(6,2)    NULL,

    -- ── Document quality ─────────────────────────────────────
    FLAG_DOCS_SUM           TINYINT         NULL,
    CNT_CHILDREN_RATIO      DECIMAL(6,4)    NULL,
    EMP_ANOMALY             TINYINT         NULL,

    -- ── Bureau aggregations ──────────────────────────────────
    BUREAU_LOAN_COUNT       SMALLINT        NULL,
    BUREAU_ACTIVE_COUNT     SMALLINT        NULL,
    BUREAU_CLOSED_COUNT     SMALLINT        NULL,
    BUREAU_CREDIT_SUM       DECIMAL(18,2)   NULL,
    BUREAU_CREDIT_MEAN      DECIMAL(18,2)   NULL,
    BUREAU_DEBT_SUM         DECIMAL(18,2)   NULL,
    BUREAU_OVERDUE_MEAN     DECIMAL(14,4)   NULL,
    BUREAU_OVERDUE_MAX      DECIMAL(14,4)   NULL,
    BUREAU_BAD_DEBT_COUNT   SMALLINT        NULL,
    BUREAU_BAD_DEBT_FLAG    TINYINT         NULL,
    BUREAU_ACTIVE_RATIO     DECIMAL(8,6)    NULL,
    BUREAU_DEBT_CREDIT_RATIO DECIMAL(8,6)   NULL,

    -- ── Previous application aggregations ────────────────────
    PREV_APP_COUNT          SMALLINT        NULL,
    PREV_APPROVED_COUNT     SMALLINT        NULL,
    PREV_REFUSED_COUNT      SMALLINT        NULL,
    PREV_REFUSED_RATIO      DECIMAL(8,6)    NULL,
    PREV_APPROVED_RATIO     DECIMAL(8,6)    NULL,
    PREV_CREDIT_MEAN        DECIMAL(18,2)   NULL,
    PREV_CREDIT_MAX         DECIMAL(18,2)   NULL,
    PREV_ANNUITY_MEAN       DECIMAL(14,4)   NULL,

    -- ── Installment aggregations ─────────────────────────────
    INST_PAYMENT_DIFF_MEAN  DECIMAL(14,4)   NULL,
    INST_PAYMENT_DIFF_MAX   DECIMAL(14,4)   NULL,
    INST_DPD_MEAN           DECIMAL(10,4)   NULL,
    INST_DPD_MAX            DECIMAL(10,4)   NULL,
    INST_LATE_COUNT         INT             NULL,

    -- ── POS Cash aggregations ─────────────────────────────────
    POS_DPD_MEAN            DECIMAL(10,4)   NULL,
    POS_DPD_MAX             SMALLINT        NULL,
    POS_MONTHS_COUNT        SMALLINT        NULL,
    POS_COMPLETED_RATIO     DECIMAL(8,6)    NULL,

    -- ── Credit card aggregations ─────────────────────────────
    CC_BALANCE_MEAN         DECIMAL(18,2)   NULL,
    CC_BALANCE_MAX          DECIMAL(18,2)   NULL,
    CC_LIMIT_MEAN           DECIMAL(18,2)   NULL,
    CC_UTILIZATION_MEAN     DECIMAL(8,6)    NULL,
    CC_UTILIZATION_MAX      DECIMAL(8,6)    NULL,
    CC_DPD_MEAN             DECIMAL(10,4)   NULL,
    CC_DPD_MAX              SMALLINT        NULL,
    CC_MONTHS_COUNT         SMALLINT        NULL,

    CREATED_AT              DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_AppFeatures  PRIMARY KEY CLUSTERED (SK_ID_CURR),
    CONSTRAINT FK_AppFeatures_App
        FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR)
);
GO


-- ============================================================
-- MODEL RESULTS TABLE (Phase 3 output)
-- ============================================================
IF OBJECT_ID('model.Predictions', 'U') IS NOT NULL DROP TABLE model.Predictions;
GO

CREATE TABLE model.Predictions (

    SK_ID_CURR              INT             NOT NULL,
    MODEL_VERSION           NVARCHAR(30)    NOT NULL DEFAULT 'lgbm_v1',

    -- ── Scores ───────────────────────────────────────────────
    DEFAULT_PROBABILITY     DECIMAL(8,6)    NOT NULL,   -- 0.0 to 1.0
    RISK_TIER               NVARCHAR(15)    NOT NULL,   -- 'Very Low'|'Low'|'Medium'|'High'
    OOF_FOLD                TINYINT         NULL,       -- which CV fold this came from

    -- ── Actuals (NULL for test set) ───────────────────────────
    ACTUAL_TARGET           TINYINT         NULL,
    IS_CORRECT              AS CASE
                                WHEN ACTUAL_TARGET IS NULL THEN NULL
                                WHEN (DEFAULT_PROBABILITY >= 0.5 AND ACTUAL_TARGET = 1) THEN 1
                                WHEN (DEFAULT_PROBABILITY <  0.5 AND ACTUAL_TARGET = 0) THEN 1
                                ELSE 0
                              END,

    PREDICTED_AT            DATETIME2       NOT NULL DEFAULT GETDATE(),

    CONSTRAINT PK_Predictions  PRIMARY KEY CLUSTERED (SK_ID_CURR, MODEL_VERSION),
    CONSTRAINT FK_Pred_App      FOREIGN KEY (SK_ID_CURR) REFERENCES raw.Application(SK_ID_CURR),
    CONSTRAINT CK_RiskTier      CHECK (RISK_TIER IN ('Very Low', 'Low', 'Medium', 'High')),
    CONSTRAINT CK_Prob          CHECK (DEFAULT_PROBABILITY BETWEEN 0 AND 1)
);
GO

CREATE NONCLUSTERED INDEX IX_Pred_Tier  ON model.Predictions (RISK_TIER) INCLUDE (DEFAULT_PROBABILITY, SK_ID_CURR);
GO


-- ============================================================
-- USEFUL VIEWS FOR REPORTING / POWER BI
-- ============================================================

-- View 1: Full applicant profile with prediction (for dashboard)
CREATE OR ALTER VIEW model.vw_ApplicantRiskProfile AS
SELECT
    a.SK_ID_CURR,
    a.TARGET,
    a.CODE_GENDER,
    a.NAME_INCOME_TYPE,
    a.NAME_EDUCATION_TYPE,
    a.OCCUPATION_TYPE,
    a.NAME_CONTRACT_TYPE,
    a.AMT_CREDIT,
    a.AMT_INCOME_TOTAL,
    f.AGE_YEARS,
    f.CREDIT_INCOME_RATIO,
    f.BUREAU_BAD_DEBT_FLAG,
    f.PREV_REFUSED_RATIO,
    a.EXT_SOURCE_2,
    a.EXT_SOURCE_3,
    p.DEFAULT_PROBABILITY,
    p.RISK_TIER
FROM raw.Application       a
LEFT JOIN feat.ApplicationFeatures f ON f.SK_ID_CURR = a.SK_ID_CURR
LEFT JOIN model.Predictions        p ON p.SK_ID_CURR = a.SK_ID_CURR
                                     AND p.MODEL_VERSION = 'lgbm_v1'
WHERE a.IS_TEST = 0;
GO

-- View 2: Default rate by segment (for EDA / business insights)
CREATE OR ALTER VIEW model.vw_DefaultRateBySegment AS
SELECT
    NAME_INCOME_TYPE,
    NAME_EDUCATION_TYPE,
    CODE_GENDER,
    COUNT(*)                        AS applicant_count,
    SUM(CAST(TARGET AS INT))        AS default_count,
    AVG(CAST(TARGET AS FLOAT))*100  AS default_rate_pct,
    AVG(AMT_CREDIT)                 AS avg_credit_amount,
    AVG(AMT_INCOME_TOTAL)           AS avg_income
FROM raw.Application
WHERE IS_TEST = 0
GROUP BY NAME_INCOME_TYPE, NAME_EDUCATION_TYPE, CODE_GENDER;
GO


-- ============================================================
-- BULK IMPORT TEMPLATE (run after Python cleaning is done)
-- Replace paths as needed
-- ============================================================
/*
-- Option A: BULK INSERT from CSV (if you export cleaned CSVs)
BULK INSERT raw.Application
FROM 'd:\Risk\data\cleaned\application_train_clean.csv'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    TABLOCK
);

-- Option B: Use Python pandas + pyodbc / sqlalchemy (recommended for Parquet)
-- In Python:
--   from sqlalchemy import create_engine
--   engine = create_engine("mssql+pyodbc://server/HomeCredit?driver=ODBC+Driver+17+for+SQL+Server")
--   df.to_sql('Application', engine, schema='raw', if_exists='append', index=False, chunksize=5000)
*/


-- ============================================================
-- QUICK VALIDATION QUERIES (run after import)
-- ============================================================
/*
-- Check row counts
SELECT 'Application'           AS tbl, COUNT(*) AS rows FROM raw.Application
UNION ALL SELECT 'Bureau',                COUNT(*) FROM raw.Bureau
UNION ALL SELECT 'PreviousApplication',   COUNT(*) FROM raw.PreviousApplication
UNION ALL SELECT 'InstallmentsPayments',  COUNT(*) FROM raw.InstallmentsPayments
UNION ALL SELECT 'POSCashBalance',        COUNT(*) FROM raw.POSCashBalance
UNION ALL SELECT 'CreditCardBalance',     COUNT(*) FROM raw.CreditCardBalance;

-- Check default rate
SELECT
    TARGET,
    COUNT(*) AS cnt,
    COUNT(*)*100.0 / SUM(COUNT(*)) OVER() AS pct
FROM raw.Application WHERE IS_TEST = 0
GROUP BY TARGET;

-- Check no remaining anomaly
SELECT COUNT(*) AS emp_anomaly_unfixed
FROM raw.Application
WHERE DAYS_EMPLOYED > 0 AND EMP_ANOMALY = 0;
-- Should return 0
*/

PRINT 'HomeCredit schema created successfully.';
GO
