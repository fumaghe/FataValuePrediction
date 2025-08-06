"""
LightGBM + Optuna: training e predizione per-target e per-ruolo.

Fix inclusi:
- Nessun `categorical_feature` dentro `params`; lo passiamo a `.fit()` per evitare warning.
- Poisson: fallback automatico se la somma delle etichette nel fold è 0 (evita LightGBMError).
- Esclusione rigorosa delle colonne "leaky" (target e *_per90) + per mv/fmv niente gf/assist/clean_sheet grezzi.
- `role` aggiunto alle categoriche.
- GroupKFold con n_splits sicuro e Optuna con seed per riproducibilità.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

from sklearn.model_selection import GroupKFold
from optuna.samplers import TPESampler

# importa costanti dal pacchetto
from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS

# ------------------------------------------------------------------ #
#  Utils
# ------------------------------------------------------------------ #
TARGETS_RATE = {"gf", "assist"}  # questi li alleniamo su rate per-90

def _safe_rate(df: pd.DataFrame, num: str, minutes_col: str = "min_playing_time") -> pd.Series:
    m = df[minutes_col].fillna(0).astype(float)
    n = df[num].fillna(0).astype(float)
    denom = np.where(m > 0, m, 1.0)
    return 90.0 * n / denom

def _pick_objective(target: str, y_train: pd.Series) -> str:
    """Sceglie l'objective: Poisson per gf/assist se somma>0, altrimenti 'mae'."""
    if target in TARGETS_RATE and float(np.nansum(y_train)) > 0.0:
        return "poisson"
    return "mae"

def _numeric_features(df: pd.DataFrame, target: str) -> List[str]:
    base_num = [c for c in df.select_dtypes(include="number").columns if c not in LEAK_COLS]
    # escludi target e target_per90
    leak = {target, f"{target}_per90"}
    # per mv/fmv, non usare gf/assist/clean_sheet (né le versioni per-90) come feature grezze
    if target in {"mv", "fmv"}:
        leak |= {"gf", "assist", "clean_sheet", "gf_per90", "assist_per90", "clean_sheet_per90"}
    return [c for c in base_num if c not in leak]

def _build_Xy(role_df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepara X (train) e y per il ruolo corrente."""
    if target in TARGETS_RATE:
        if f"{target}_per90" in role_df.columns:
            y = role_df[f"{target}_per90"].fillna(0).clip(lower=0)
        else:
            y = _safe_rate(role_df, target).clip(lower=0)
    else:
        y = role_df[target].fillna(0)
        if target == "clean_sheet":
            y = y.clip(lower=0)
    # feature columns
    num_cols = _numeric_features(role_df, target)
    extra_cols: List[str] = []
    if target in {"mv", "fmv"}:
        for c in ("gf_pred", "assist_pred"):
            if c in role_df.columns:
                extra_cols.append(c)
    # categoriche
    feat_cols = list(dict.fromkeys(num_cols + extra_cols + CAT_COLS + ["role"]))
    X = role_df[feat_cols].copy()
    for c in CAT_COLS + ["role"]:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, y

def _lgb_default_params() -> Dict:
    return dict(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=400,
        num_leaves=63,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=42,
        **GPU_PARAMS,
    )

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
    """Allena (con Optuna) e predice per ciascun `role` presente in df_feat.

    Ritorna una Series indicizzata come le righe future con il nome f"{target}_pred".
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # split train / future
    tr_df = df_feat[df_feat["season"] <= train_until].copy()
    fu_df = df_feat[df_feat["season"]  > train_until].copy()

    preds: List[pd.Series] = []

    for role, role_df in tr_df.groupby("role", dropna=False):
        role = str(role)
        # se target è clean_sheet e ruolo != P, skippa e predici 0
        if target == "clean_sheet" and role != "P":
            idx = fu_df[fu_df["role"].astype(str) == role].index
            preds.append(pd.Series(0.0, index=idx, name=f"{target}_pred"))
            continue

        # prepara X,y sul train e X_pred sul future
        X_train, y = _build_Xy(role_df, target)
        X_pred, _ = _build_Xy(fu_df[fu_df["role"].astype(str) == role], target)

        # se non ci sono righe future per questo ruolo, passa oltre
        if X_pred.shape[0] == 0:
            continue

        # Poisson fallback: se somma(y)==0 a livello di *intero train* → modello costante 0
        use_objective = _pick_objective(target, y)
        constant_zero = False
        if target in TARGETS_RATE and use_objective != "poisson":
            constant_zero = True

        # sample weight: proporzionale ai minuti (se presenti)
        sample_w = None
        if "min_playing_time" in role_df.columns:
            sample_w = role_df.loc[X_train.index, "min_playing_time"].fillna(0).values

        # Optuna
        model_path = models_dir / f"{target}_{role}.joblib"
        if constant_zero:
            model = None  # usiamo predizioni nulle
        else:
            def objective(trial: optuna.Trial) -> float:
                params = _lgb_default_params()
                params.update(
                    num_leaves=trial.suggest_int("num_leaves", 31, 127),
                    learning_rate=trial.suggest_float("learning_rate", 0.01, 0.10),
                    n_estimators=trial.suggest_int("n_estimators", 200, 700),
                    max_depth=trial.suggest_int("max_depth", -1, 10),
                )
                cv = GroupKFold(n_splits=min(4, max(2, role_df["season"].nunique())))
                oof = []
                for tr_idx, va_idx in cv.split(X_train, groups=role_df.loc[X_train.index, "season"]):
                    Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
                    ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

                    # se in questo fold somma(y)==0, salta la valutazione (loss grande)
                    if target in TARGETS_RATE and float(np.nansum(ytr)) == 0.0:
                        oof.append(1e3)
                        continue

                    m = lgb.LGBMRegressor(
                        **params,
                        objective=use_objective,
                        verbose=-1,
                    )
                    m.fit(
                        Xtr, ytr,
                        sample_weight=None if sample_w is None else sample_w[tr_idx],
                        categorical_feature=CAT_COLS + ["role"],
                    )
                    p = m.predict(Xva)
                    # metric: MAE sulle *rate* (o sul target continuo)
                    oof.append(np.mean(np.abs(p - yva)))
                return float(np.mean(oof))

            study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

            best = study.best_trial.params if len(study.trials) else {}
            params = _lgb_default_params()
            params.update(best)
            model = lgb.LGBMRegressor(**params, objective=use_objective, verbose=-1)
            model.fit(
                X_train, y,
                sample_weight=sample_w,
                categorical_feature=CAT_COLS + ["role"],
            )
            joblib.dump(model, model_path)

        # predizioni
        if model is None:
            p = np.zeros(X_pred.shape[0], dtype=float)
        else:
            p = model.predict(X_pred)

        # per gf/assist: da rate → conteggio stagionale usando i minuti previsti
        if target in TARGETS_RATE:
            mins = fu_df.loc[X_pred.index, "min_playing_time"].fillna(0).values
            p = p * (mins / 90.0)

        preds.append(pd.Series(p, index=X_pred.index, name=f"{target}_pred"))

    # riordina per indice complessivo
    if not preds:
        return pd.Series(dtype=float, name=f"{target}_pred")
    return pd.concat(preds).sort_index()
