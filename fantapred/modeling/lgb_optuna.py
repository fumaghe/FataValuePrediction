"""
LightGBM + Optuna:
addestra (o ri-usa) un modello per ciascun ruolo e produce le colonne
 *_pred  su gf / assist / mv / fmv / clean_sheet.

• Per mv / fmv include come feature gf_pred e assist_pred.
• Per gf / assist il modello lavora su rate per-90, poi riconverte
  l’output in conteggio stagionale moltiplicando per min_playing_time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib, optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS

# ------------------------------------------------------------------ #
#  Optuna: spazio d’ipermetri
# ------------------------------------------------------------------ #
def _suggest_params(trial: optuna.trial.Trial, target: str) -> Dict:
    params = dict(
        num_leaves       = trial.suggest_int ("leaves",     31, 95, step=16),
        learning_rate    = trial.suggest_float("lr",        0.03, 0.10, log=True),
        n_estimators     = trial.suggest_int ("n_estim",    300, 600, step=100),
        max_depth        = trial.suggest_int ("depth",       4,  8),
        subsample        = 0.8,
        colsample_bytree = 0.8,
        random_state     = 42,
        **GPU_PARAMS,
    )
    if target in {"gf", "assist"}:
        params.update(objective="poisson", metric="poisson")
    else:                                       # mv / fmv / clean_sheet
        params.update(objective="regression_l1", metric="mae")
    return params


# ------------------------------------------------------------------ #
#  Train-&-predict separato per ruolo
# ------------------------------------------------------------------ #
def fit_predict_by_role(
    df_feat: pd.DataFrame,
    target: str,
    train_until: str,
    models_dir: Path,
    n_trials: int = 20,
) -> pd.Series:

    models_dir.mkdir(parents=True, exist_ok=True)
    pred_list: List[pd.Series] = []

    for role, role_df in df_feat.groupby("role"):

        train_df = role_df[role_df.season <= train_until]
        pred_df  = role_df[role_df.season >  train_until]
        if train_df.empty or pred_df.empty:     # ← FIX: senza parentesi!
            continue

        # ----- target -------------------------------------------------- #
        if target in {"gf", "assist"}:
            y = train_df[f"{target}_per90"].fillna(0)
        else:
            y = train_df[target].fillna(6.0)

        # ----- feature set -------------------------------------------- #
        num_cols = [
            c for c in train_df.select_dtypes(include="number").columns
            if c not in LEAK_COLS
               and c not in {f"{target}_pred", "presenze_pred"}   # evita duplicati
        ]

        extra_cols: List[str] = []
        if target in {"mv", "fmv"}:
            extra_cols = [c for c in ("gf_pred", "assist_pred") if c in role_df.columns]

        feature_cols = list(dict.fromkeys(num_cols + extra_cols + CAT_COLS + ["role"]))

        X_train = train_df[feature_cols].copy()
        X_pred  = pred_df [feature_cols].copy()

        for c in CAT_COLS + ["role"]:
            X_train[c] = X_train[c].astype("category")
            X_pred[c]  = X_pred[c].astype("category")

        sample_w = train_df["min_playing_time"].fillna(0) / 90.0 + 1e-3

        # ----- load oppure Optuna ------------------------------------- #
        mdl_path = models_dir / f"{target}_{role}.pkl"
        if mdl_path.exists():
            model = joblib.load(mdl_path)
        else:

            def objective(trial: optuna.trial.Trial) -> float:
                params = _suggest_params(trial, target)
                cv     = GroupKFold(4)
                mae: List[float] = []
                for tr_idx, va_idx in cv.split(X_train, y, groups=train_df["season"]):
                    m = lgb.LGBMRegressor(**params,
                                          categorical_feature=CAT_COLS,
                                          verbose=-1)
                    m.fit(X_train.iloc[tr_idx], y.iloc[tr_idx],
                          sample_weight=sample_w.iloc[tr_idx])
                    mae.append(
                        mean_absolute_error(y.iloc[va_idx],
                                            m.predict(X_train.iloc[va_idx]))
                    )
                return float(np.mean(mae))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best_par = _suggest_params(study.best_trial, target)
            model = lgb.LGBMRegressor(**best_par,
                                      categorical_feature=CAT_COLS,
                                      verbose=-1)
            model.fit(X_train, y, sample_weight=sample_w)
            joblib.dump(model, mdl_path)

        # ----- prediction --------------------------------------------- #
        p = model.predict(X_pred)
        if target in {"gf", "assist"}:         # rate → conteggio
            p = p * pred_df["min_playing_time"] / 90.0

        pred_list.append(pd.Series(p, index=pred_df.index,
                                   name=f"{target}_pred"))

    return pd.concat(pred_list).sort_index()
