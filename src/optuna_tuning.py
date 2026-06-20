"""
optuna_tuning.py — Bayesian Hyperparameter Optimization
=======================================================
Dùng Optuna + TPE Sampler để tối ưu LightGBM params.
Output: models/best_params.json

Chạy 1 lần trước modeling.py.
Thời gian: ~90-120 phút với n_trials=100 trên CPU.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, os, time, warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR   = str(ROOT_DIR / 'data')
MODELS_DIR = str(ROOT_DIR / 'models')
REPORTS_DIR = str(ROOT_DIR / 'reports')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def objective(trial, X: pd.DataFrame, y: pd.Series, n_splits: int = 3) -> float:
    """
    Optuna objective: maximise OOF AUC với 3-fold CV.
    3-fold (không phải 5-fold) để giảm search time trong tuning phase.
    Final training dùng 5-fold với params đã tìm được.
    """
    params = {
        'objective':        'binary',
        'metric':           'auc',
        'boosting_type':    'gbdt',
        'verbose':          -1,
        'n_jobs':           -1,
        'random_state':     42,
        'scale_pos_weight': float((y == 0).sum() / (y == 1).sum()),

        # Search space — bounded để tránh overfitting
        'n_estimators':      trial.suggest_int  ('n_estimators',      200,  2000, step=100),
        'learning_rate':     trial.suggest_float ('learning_rate',     0.005, 0.15, log=True),
        'num_leaves':        trial.suggest_int   ('num_leaves',        16,   127),
        'max_depth':         trial.suggest_int   ('max_depth',         3,    10),
        'min_child_samples': trial.suggest_int   ('min_child_samples', 10,   200),
        'subsample':         trial.suggest_float ('subsample',         0.50, 1.00),
        'subsample_freq':    trial.suggest_int   ('subsample_freq',    1,    7),
        'colsample_bytree':  trial.suggest_float ('colsample_bytree',  0.30, 1.00),
        'reg_alpha':         trial.suggest_float ('reg_alpha',         1e-5, 10.0, log=True),
        'reg_lambda':        trial.suggest_float ('reg_lambda',        1e-5, 10.0, log=True),
        'min_split_gain':    trial.suggest_float ('min_split_gain',    0.0,  1.0),
        'min_child_weight':  trial.suggest_float ('min_child_weight',  1e-3, 10.0, log=True),
    }

    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(X))

    for train_idx, val_idx in skf.split(X, y):
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X.iloc[train_idx], y.iloc[train_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False)]
        )
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

    return roc_auc_score(y, oof_preds)


def run_optuna_search(X: pd.DataFrame, y: pd.Series, n_trials: int = 100) -> dict:
    """
    Chạy Optuna hyperparameter search.
    Recommended: n_trials=100 (đủ để converge với LightGBM search space).
    """
    print(f"\n{'='*65}")
    print(f"  Optuna Hyperparameter Search — LightGBM Credit Risk")
    print(f"  Trials: {n_trials} | Sampler: TPE | Pruner: Hyperband")
    print(f"{'='*65}")

    study = optuna.create_study(
        direction='maximize',
        study_name='lgbm_credit_risk',
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=50, max_resource=500, reduction_factor=3
        )
    )

    t_start = time.time()
    study.optimize(
        lambda trial: objective(trial, X, y, n_splits=3),
        n_trials=n_trials,
        show_progress_bar=True,
        n_jobs=1,
    )
    elapsed = time.time() - t_start

    best_params = study.best_params
    best_auc    = study.best_value

    print(f"\n  Search complete in {elapsed/60:.1f} minutes")
    print(f"  Best OOF AUC (3-fold): {best_auc:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in sorted(best_params.items()):
        print(f"    {k:<30} = {v}")

    output = {
        'best_auc_3fold':  float(best_auc),
        'n_trials':        n_trials,
        'search_time_min': round(elapsed / 60, 1),
        'params':          best_params,
    }
    with open(f'{MODELS_DIR}/best_params.json', 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: models/best_params.json")

    # Plot optimization history + param importance
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Optimization history
        trial_vals  = [t.value for t in study.trials if t.value is not None]
        best_so_far = [max(trial_vals[:i+1]) for i in range(len(trial_vals))]
        axes[0].plot(trial_vals,  alpha=0.35, color='steelblue', label='Trial AUC')
        axes[0].plot(best_so_far, color='red', linewidth=2.5, label='Best so far')
        axes[0].axhline(best_auc, color='darkred', linestyle='--', alpha=0.6)
        axes[0].set_xlabel('Trial')
        axes[0].set_ylabel('OOF AUC (3-fold)')
        axes[0].set_title(f'Optuna Optimization History\nBest AUC: {best_auc:.4f}')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Hyperparameter importance
        importances   = optuna.importance.get_param_importances(study)
        params_sorted = list(importances.keys())[:10]
        vals_sorted   = [importances[p] for p in params_sorted]
        axes[1].barh(params_sorted[::-1], vals_sorted[::-1], color='steelblue')
        axes[1].set_xlabel('Importance Score')
        axes[1].set_title('Hyperparameter Importance (Optuna FAnova)')
        axes[1].grid(alpha=0.3, axis='x')

        plt.suptitle(f'Optuna Search — Best AUC: {best_auc:.4f}', fontsize=13)
        plt.tight_layout()
        plt.savefig(f'{REPORTS_DIR}/optuna_results.png', dpi=150)
        plt.close()
        print(f"  Saved: reports/optuna_results.png")
    except Exception as e:
        print(f"  Plot skipped: {e}")

    return best_params


if __name__ == '__main__':
    t0 = time.time()
    print("Loading data...")
    train_df = pd.read_parquet(f'{DATA_DIR}/train_features.parquet')
    FEATURES = [c for c in train_df.columns if c not in ['TARGET', 'SK_ID_CURR']]
    X = train_df[FEATURES]
    y = train_df['TARGET']
    print(f"Dataset: {X.shape} | Default rate: {y.mean()*100:.2f}%")

    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    best = run_optuna_search(X, y, n_trials=n_trials)

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Next step: results saved to models/best_params.json")
    print("           Update params dict in modeling.py accordingly.")
