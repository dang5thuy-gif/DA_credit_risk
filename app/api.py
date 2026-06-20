"""
app/api.py — Phase F: FastAPI REST API for Credit Risk Scoring
=============================================================
Production REST API wrapping the 5-fold LightGBM ensemble.

Endpoints:
    POST /score          — Score a single applicant
    POST /score/batch    — Score up to 1000 applicants
    GET  /health         — Health check
    GET  /model/info     — Model metadata
    GET  /model/features — Feature list

Run:
    uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload

Test:
    curl -X POST http://localhost:8000/score \
         -H "Content-Type: application/json" \
         -d '{"EXT_SOURCE_2": 0.75, "CREDIT_INCOME_RATIO": 2.5}'
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import pickle, json, os, re, time
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

app = FastAPI(
    title="Enterprise Credit Risk Scoring API",
    description=(
        "Production-grade REST API for loan default probability scoring. "
        "Powered by LightGBM 5-fold ensemble with Isotonic Regression calibration. "
        "Compliant with IFRS 9 ECL framework."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Load models at startup
# ─────────────────────────────────────────────────────────────────────────────
_models     = []
_features   = []
_calibrator = None
_tier_cfg   = {}
_imp_stats  = {}
_metrics    = {}
_load_time  = None


def load_artifacts():
    global _models, _features, _calibrator, _tier_cfg, _imp_stats, _metrics, _load_time

    for i in range(1, 6):
        path = os.path.join(MODELS_DIR, f'lgbm_fold{i}.pkl')
        with open(path, 'rb') as f:
            _models.append(pickle.load(f))

    with open(os.path.join(MODELS_DIR, 'feature_list.pkl'), 'rb') as f:
        raw_features = pickle.load(f)
    _features = [re.sub(r'[^A-Za-z0-9_]+', '_', x) for x in raw_features]

    cal_path = os.path.join(MODELS_DIR, 'isotonic_calibrator.pkl')
    if os.path.exists(cal_path):
        with open(cal_path, 'rb') as f:
            _calibrator = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'tier_config.json')) as f:
        _tier_cfg = json.load(f)

    imp_path = os.path.join(MODELS_DIR, 'imputation_stats.json')
    if os.path.exists(imp_path):
        with open(imp_path) as f:
            _imp_stats = json.load(f)

    metrics_path = os.path.join(MODELS_DIR, 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            _metrics = json.load(f)

    _load_time = datetime.utcnow().isoformat()
    return True


try:
    load_artifacts()
except Exception as e:
    print(f"WARNING: Could not load models: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas
# ─────────────────────────────────────────────────────────────────────────────

class ApplicantInput(BaseModel):
    """Input schema for a single loan applicant.
    All fields are optional — missing values are imputed from training medians.
    """
    # Core financial features (most important per SHAP)
    EXT_SOURCE_2:          Optional[float] = Field(None, ge=0, le=1, description="External credit score 2 (0-1)")
    EXT_SOURCE_3:          Optional[float] = Field(None, ge=0, le=1, description="External credit score 3 (0-1)")
    EXT_SOURCE_1:          Optional[float] = Field(None, ge=0, le=1, description="External credit score 1 (0-1)")
    AMT_CREDIT:            Optional[float] = Field(None, gt=0, description="Loan amount requested ($)")
    AMT_INCOME_TOTAL:      Optional[float] = Field(None, gt=0, description="Annual income ($)")
    AMT_ANNUITY:           Optional[float] = Field(None, gt=0, description="Monthly annuity amount ($)")
    AMT_GOODS_PRICE:       Optional[float] = Field(None, gt=0, description="Goods price (for consumer loans)")
    AGE_YEARS:             Optional[float] = Field(None, ge=18, le=70, description="Applicant age (years)")
    YEARS_EMPLOYED:        Optional[float] = Field(None, ge=0, description="Years at current employer")
    CNT_CHILDREN:          Optional[int]   = Field(None, ge=0, le=15)
    # Derived ratios (computed automatically if AMT_* fields provided)
    CREDIT_INCOME_RATIO:   Optional[float] = Field(None, description="Loan-to-income ratio (auto-computed)")
    ANNUITY_INCOME_RATIO:  Optional[float] = Field(None)
    CREDIT_TERM:           Optional[float] = Field(None, description="Loan term in months")
    # Bureau / behavioural signals
    bureau_bad_debt_flag:  Optional[int]   = Field(None, ge=0, le=1, description="1=any bureau bad debt event")
    prev_refused_ratio:    Optional[float] = Field(None, ge=0, le=1, description="Fraction of prev applications refused")
    inst_late_ratio:       Optional[float] = Field(None, ge=0, le=1, description="Fraction of installments paid late")
    cc_utilization_mean:   Optional[float] = Field(None, ge=0, description="Average credit card utilization")
    bureau_loan_count:     Optional[int]   = Field(None, ge=0)
    # Other
    extra_features: Optional[Dict[str, Any]] = Field(None, description="Any additional raw feature values")

    class Config:
        json_schema_extra = {
            "example": {
                "EXT_SOURCE_2": 0.72,
                "EXT_SOURCE_3": 0.65,
                "AMT_CREDIT": 350000,
                "AMT_INCOME_TOTAL": 120000,
                "AMT_ANNUITY": 18000,
                "AGE_YEARS": 38,
                "YEARS_EMPLOYED": 6,
                "bureau_bad_debt_flag": 0,
                "prev_refused_ratio": 0.0,
                "inst_late_ratio": 0.02,
            }
        }


class ScoreResponse(BaseModel):
    pd_raw:         float = Field(description="Raw model PD (uncalibrated, 0-1)")
    pd_calibrated:  float = Field(description="Calibrated PD — reliable probability (0-1)")
    risk_tier:      str   = Field(description="Risk tier: Very Low / Low / Medium / High")
    ecl_estimate:   float = Field(description="Estimated ECL in dollars (PD x LGD x EAD)")
    recommendation: str   = Field(description="Lending recommendation")
    top_risk_factors: List[str] = Field(description="Top 3 features driving the risk score")
    processing_ms:  float


class BatchRequest(BaseModel):
    applicants: List[ApplicantInput] = Field(..., max_length=1000)


class BatchResponse(BaseModel):
    count: int
    results: List[ScoreResponse]
    processing_ms: float


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring logic
# ─────────────────────────────────────────────────────────────────────────────

def _build_feature_row(inp: ApplicantInput) -> pd.DataFrame:
    """Build a full feature vector, imputing missing values from training medians."""
    row = {feat: _imp_stats.get(feat, {}).get('median', 0) for feat in _features}

    # Override with explicit inputs
    overrides = {}
    if inp.EXT_SOURCE_2       is not None: overrides['EXT_SOURCE_2']       = inp.EXT_SOURCE_2
    if inp.EXT_SOURCE_3       is not None: overrides['EXT_SOURCE_3']       = inp.EXT_SOURCE_3
    if inp.EXT_SOURCE_1       is not None: overrides['EXT_SOURCE_1']       = inp.EXT_SOURCE_1
    if inp.AMT_CREDIT         is not None: overrides['AMT_CREDIT']         = inp.AMT_CREDIT
    if inp.AMT_INCOME_TOTAL   is not None: overrides['AMT_INCOME_TOTAL']   = inp.AMT_INCOME_TOTAL
    if inp.AMT_ANNUITY        is not None: overrides['AMT_ANNUITY']        = inp.AMT_ANNUITY
    if inp.AMT_GOODS_PRICE    is not None: overrides['AMT_GOODS_PRICE']    = inp.AMT_GOODS_PRICE
    if inp.AGE_YEARS          is not None: overrides['AGE_YEARS']          = inp.AGE_YEARS
    if inp.YEARS_EMPLOYED     is not None: overrides['YEARS_EMPLOYED']     = inp.YEARS_EMPLOYED
    if inp.CNT_CHILDREN       is not None: overrides['CNT_CHILDREN']       = inp.CNT_CHILDREN
    if inp.bureau_bad_debt_flag is not None: overrides['bureau_bad_debt_flag'] = inp.bureau_bad_debt_flag
    if inp.prev_refused_ratio is not None: overrides['prev_refused_ratio'] = inp.prev_refused_ratio
    if inp.inst_late_ratio    is not None: overrides['inst_late_ratio']    = inp.inst_late_ratio
    if inp.cc_utilization_mean is not None: overrides['cc_utilization_mean'] = inp.cc_utilization_mean
    if inp.bureau_loan_count  is not None: overrides['bureau_loan_count']  = inp.bureau_loan_count
    if inp.extra_features:                 overrides.update(inp.extra_features)

    row.update({k: v for k, v in overrides.items() if k in row})

    # Auto-compute derived ratios
    credit = row.get('AMT_CREDIT', 1)
    income = row.get('AMT_INCOME_TOTAL', 1)
    annuity = row.get('AMT_ANNUITY', 1)
    if inp.AMT_CREDIT and inp.AMT_INCOME_TOTAL and inp.CREDIT_INCOME_RATIO is None:
        row['CREDIT_INCOME_RATIO']  = credit / (income + 1e-6)
    if inp.AMT_ANNUITY and inp.AMT_INCOME_TOTAL and inp.ANNUITY_INCOME_RATIO is None:
        row['ANNUITY_INCOME_RATIO'] = annuity / (income + 1e-6)
    if inp.AMT_CREDIT and inp.AMT_ANNUITY and inp.CREDIT_TERM is None:
        row['CREDIT_TERM']          = credit / (annuity + 1e-6)

    return pd.DataFrame([row])[_features]


def _assign_tier(pd_val: float) -> str:
    bins   = _tier_cfg.get('bins', [0, 0.279, 0.457, 0.649, 1.0])
    labels = ['Very Low', 'Low', 'Medium', 'High']
    for i in range(len(bins) - 1):
        if bins[i] <= pd_val <= bins[i + 1]:
            return labels[i]
    return 'High'


def _get_recommendation(tier: str) -> str:
    recs = {
        'Very Low': 'AUTO-APPROVE: Eligible for fast-track approval and preferential rate.',
        'Low':      'STANDARD: Approve via standard process. Income verification recommended.',
        'Medium':   'CONDITIONAL: Approve with conditions — reduce limit by 20%, require documentation.',
        'High':     'REVIEW: Manual underwriter review required. Consider collateral or co-borrower.',
    }
    return recs.get(tier, 'REVIEW')


def _score_row(client_df: pd.DataFrame) -> tuple:
    """Returns (raw_prob, calibrated_prob)."""
    raw_preds = np.mean([m.predict_proba(client_df)[:, 1] for m in _models], axis=0)
    raw_prob  = float(raw_preds[0])
    if _calibrator is not None:
        cal_prob = float(np.clip(_calibrator.predict(np.array([[raw_prob]])), 0.001, 0.999)[0])
    else:
        cal_prob = raw_prob
    return raw_prob, cal_prob


def _top_risk_features(client_df: pd.DataFrame, top_n: int = 3) -> List[str]:
    """Return top N features by absolute value vs population median."""
    pop_medians = {feat: _imp_stats.get(feat, {}).get('median', 0) for feat in _features}
    deviations  = {}
    for feat in _features:
        client_val = float(client_df[feat].iloc[0])
        median_val = pop_medians.get(feat, 0)
        if median_val != 0:
            deviations[feat] = abs((client_val - median_val) / (abs(median_val) + 1e-9))
    top = sorted(deviations, key=deviations.get, reverse=True)[:top_n]
    return top


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    return {
        "status":       "healthy",
        "models_loaded": len(_models),
        "load_time":    _load_time,
        "timestamp":    datetime.utcnow().isoformat(),
    }


@app.get("/model/info")
async def model_info():
    return {
        "model_type":    "LightGBM 5-Fold Ensemble + Isotonic Regression Calibration",
        "n_folds":       len(_models),
        "n_features":    len(_features),
        "calibrated":    _calibrator is not None,
        "oof_auc":       _metrics.get('oof_auc'),
        "ks_statistic":  _metrics.get('ks_statistic'),
        "gini":          _metrics.get('gini'),
        "brier_calibrated": _metrics.get('brier_calibrated'),
        "framework":     "IFRS 9 ECL | Basel II IRB",
        "version":       "2.0.0",
    }


@app.get("/model/features")
async def get_features():
    return {"n_features": len(_features), "features": _features}


@app.post("/score", response_model=ScoreResponse)
async def score_applicant(inp: ApplicantInput):
    """Score a single loan applicant. Missing features are imputed from training medians."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded. Check server logs.")

    t0 = time.time()
    try:
        client_df    = _build_feature_row(inp)
        raw_prob, cal_prob = _score_row(client_df)
        tier         = _assign_tier(cal_prob)
        amt_credit   = inp.AMT_CREDIT or 0
        lgd          = 0.65
        ecl          = cal_prob * lgd * amt_credit
        top_feats    = _top_risk_features(client_df)
        rec          = _get_recommendation(tier)
        elapsed_ms   = (time.time() - t0) * 1000
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)}")

    return ScoreResponse(
        pd_raw=round(raw_prob, 6),
        pd_calibrated=round(cal_prob, 6),
        risk_tier=tier,
        ecl_estimate=round(ecl, 2),
        recommendation=rec,
        top_risk_factors=top_feats,
        processing_ms=round(elapsed_ms, 2),
    )


@app.post("/score/batch", response_model=BatchResponse)
async def score_batch(req: BatchRequest):
    """Score up to 1000 applicants in a single request."""
    if not _models:
        raise HTTPException(status_code=503, detail="Models not loaded.")

    t0 = time.time()
    results = []
    for inp in req.applicants:
        try:
            client_df         = _build_feature_row(inp)
            raw_prob, cal_prob = _score_row(client_df)
            tier              = _assign_tier(cal_prob)
            amt_credit        = inp.AMT_CREDIT or 0
            ecl               = cal_prob * 0.65 * amt_credit
            rec               = _get_recommendation(tier)
            results.append(ScoreResponse(
                pd_raw=round(raw_prob, 6), pd_calibrated=round(cal_prob, 6),
                risk_tier=tier, ecl_estimate=round(ecl, 2),
                recommendation=rec, top_risk_factors=_top_risk_features(client_df),
                processing_ms=0,
            ))
        except Exception as e:
            results.append(ScoreResponse(
                pd_raw=0, pd_calibrated=0, risk_tier='Unknown',
                ecl_estimate=0, recommendation=f'ERROR: {e}',
                top_risk_factors=[], processing_ms=0,
            ))

    elapsed_ms = (time.time() - t0) * 1000
    return BatchResponse(count=len(results), results=results,
                         processing_ms=round(elapsed_ms, 2))
