from __future__ import annotations
from pathlib import Path
from typing import Dict, List

import joblib, optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance
from sklearn.model_selection import GroupKFold

from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS

# ===== parametri pesi adattivi =====
W_MIN        = 0.05
ALPHA_Z      = 0.8
TAU_LOW      = 0.60
RECENCY_LMB  = 0.25
EPS_MAD      = 120.0

# Blend severity (riuso della stessa logica di minutes)
BLEND_MAX          = 0.75
BLEND_STEEPNESS    = 3.0
BLEND_CENTER       = 0.50
OUTLIER_LAST_CUTOFF = 1000.0
HEALTHY_OK_MIN      = 1800.0
HEALTHY_RATIO_CLUB  = 0.60
HEALTHY_FLOOR_MIN   = 1200.0

def _season_year(s: pd.Series) -> pd.Series:
    y = s.astype(str).str.extract(r"s_(\d{2})_\d{2}")[0].astype(int)
    return (2000 + y).astype(int)

def _adaptive_sample_weights(df: pd.DataFrame, minutes_col: str = "min_playing_time") -> pd.Series:
    m = df[minutes_col].fillna(0.0)

    if not {"player_id", "team_name_short", "season"}.issubset(df.columns):
        base = m / 90.0 + 1e-3
        return base.clip(lower=W_MIN)

    g = df.groupby(["player_id", "team_name_short"], observed=True)
    med = g[minutes_col].transform("median").fillna(0.0)
    mad = g[minutes_col].transform(lambda x: (x - x.median()).abs().median()).fillna(0.0)
    mad = 1.4826 * mad + EPS_MAD

    z = (m - med).abs() / mad
    ratio = m / np.maximum(med, 1.0)
    low_pen = np.clip(ratio / TAU_LOW, 0.0, 1.0) ** 2
    strength = 1 / (1 + np.exp(-(med - 1200.0) / 300.0))

    outlier_w = W_MIN + (1 - W_MIN) * np.exp(-ALPHA_Z * z) * (low_pen * strength + (1 - strength))
    outlier_w = outlier_w.clip(lower=W_MIN, upper=1.0)

    year = _season_year(df["season"])
    last_year = g["season"].transform(lambda s: _season_year(s).max())
    steps = (last_year - year).clip(lower=0)
    recency = np.exp(-RECENCY_LMB * steps)

    base = m / 90.0 + 1e-3
    w = base * outlier_w * recency
    return w.clip(lower=W_MIN)

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
    if target in {"gf", "assist", "clean_sheet"}:
        params.update(objective="poisson", metric="poisson")
    else:
        params.update(objective="regression_l1", metric="mae")
    return params

# ------------------------- RATE PRIORS ------------------------------ #
def _healthy_mask(minutes: pd.Series, club_median: pd.Series) -> pd.Series:
    thr = np.maximum(HEALTHY_RATIO_CLUB * club_median.fillna(0.0), HEALTHY_FLOOR_MIN)
    return minutes.fillna(0.0).ge(thr)

def _shifted_median(vals: pd.Series, ok_mask: pd.Series) -> pd.Series:
    vals = vals.astype(float)
    ok_mask = ok_mask.astype(bool)
    out = pd.Series(np.nan, index=vals.index, dtype=float)
    acc: List[float] = []
    for i, idx in enumerate(vals.index):
        out.iloc[i] = np.nan if len(acc) == 0 else float(np.nanmedian(acc))
        v = vals.loc[idx]
        ok = bool(ok_mask.loc[idx])
        if ok and pd.notna(v):
            acc.append(v)
    return out

def _build_rate_priors(df: pd.DataFrame, rate_col: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = _season_year(df["season"]).astype(int)
    df.sort_values(["player_id", "team_name_short", "year"], inplace=True)

    gpc = df.groupby(["player_id", "team_name_short"], observed=True)
    club_med_minutes = gpc["min_playing_time"].transform("median").fillna(0.0)
    healthy = _healthy_mask(df["min_playing_time"], club_med_minutes)

    out = pd.DataFrame(index=df.index)
    out["club_rate_prev"] = (
        df.groupby(["player_id", "team_name_short"], observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(g[rate_col], healthy.loc[g.index]))
    )
    out["player_rate_prev"] = (
        df.groupby("player_id", observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(g[rate_col], healthy.loc[g.index]))
    )

    out["club_healthy_minutes_prev"] = (
        df.groupby(["player_id","team_name_short"], observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(g["min_playing_time"], healthy.loc[g.index]))
    )
    out["last_minutes_same_club"] = (
        df.groupby(["player_id","team_name_short"], observed=True)["min_playing_time"].shift(1)
    )
    out["tenure_at_club"] = gpc.cumcount()

    return out
# ------------------------------------------------------------------- #

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

        train_df = role_df[role_df.season <= train_until].copy()
        pred_df  = role_df[role_df.season >  train_until].copy()
        if train_df.empty or pred_df.empty:
            continue

        # target e definizione "rate" (per90)
        is_rate_target = target in {"gf", "assist", "clean_sheet"}

        # per clean_sheet: alleno SOLO i portieri
        if target == "clean_sheet":
            train_df = train_df[train_df.role == "P"].copy()
            pred_df  = pred_df [pred_df.role  == "P"].copy()
            if train_df.empty and not pred_df.empty:
                # se non ho storia, restituisco 0
                pred_list.append(pd.Series(0.0, index=pred_df.index, name=f"{target}_pred"))
                continue
            if pred_df.empty:
                continue

        # y
        if is_rate_target:
            rate_col = f"{target}_per90"
            y = train_df[rate_col].fillna(0)
        else:
            y = train_df[target].fillna(6.0)

        # skip banale se tutto zero (per rate_target)
        if is_rate_target and float(y.sum()) == 0.0:
            out = pd.Series(0.0, index=pred_df.index, name=f"{target}_pred")
            pred_list.append(out)
            continue

        # features: NO leakage (escludo target e target_per90)
        banned = set(LEAK_COLS + [target, f"{target}_per90", f"{target}_pred", "presenze_pred"])
        num_cols = [
            c for c in train_df.select_dtypes(include="number").columns
            if c not in banned
        ]
        extra_cols: List[str] = []
        if target in {"mv", "fmv"}:
            extra_cols = [c for c in ("gf_pred", "assist_pred") if c in role_df.columns]

        feature_cols = list(dict.fromkeys(num_cols + extra_cols + CAT_COLS + ["role"]))
        X_train = train_df[feature_cols].copy()
        X_pred  = pred_df [feature_cols].copy()
        for c in CAT_COLS + ["role"]:
            if c in X_train:
                X_train[c] = X_train[c].astype("category")
                X_pred[c]  = X_pred[c].astype("category")

        # pesi adattivi
        sample_w = _adaptive_sample_weights(train_df, minutes_col="min_playing_time")

        # training
        mdl_path = models_dir / f"{target}_{role}.pkl"
        if mdl_path.exists():
            model = joblib.load(mdl_path)
        else:
            def objective(trial: optuna.trial.Trial) -> float:
                params = _suggest_params(trial, target)
                cv     = GroupKFold(4)
                fold_losses: List[float] = []

                for tr_idx, va_idx in cv.split(X_train, y, groups=train_df["season"]):
                    y_tr = y.iloc[tr_idx]

                    if is_rate_target and float(y_tr.sum()) == 0.0:
                        # se il fold ha tutto zero, loss ~0 predicendo tutto 0
                        y_hat = np.zeros(len(va_idx), dtype=float)
                        w_val = sample_w.iloc[va_idx]
                        loss = mean_poisson_deviance(
                            y.iloc[va_idx].clip(lower=0),
                            np.clip(y_hat, 1e-9, np.inf),
                            sample_weight=w_val
                        )
                        fold_losses.append(float(loss))
                        continue

                    m = lgb.LGBMRegressor(**params, verbose=-1)
                    m.fit(
                        X_train.iloc[tr_idx], y_tr,
                        sample_weight=sample_w.iloc[tr_idx]
                    )
                    y_pred = m.predict(X_train.iloc[va_idx])

                    if is_rate_target:
                        w_val = sample_w.iloc[va_idx]
                        loss = mean_poisson_deviance(
                            y.iloc[va_idx].clip(lower=0),
                            np.clip(y_pred, 1e-9, np.inf),
                            sample_weight=w_val
                        )
                    else:
                        w_val = sample_w.iloc[va_idx]
                        loss = mean_absolute_error(y.iloc[va_idx], y_pred, sample_weight=w_val)

                    fold_losses.append(float(loss))

                return float(np.mean(fold_losses))

            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            best_par = _suggest_params(study.best_trial, target)

            model = lgb.LGBMRegressor(**best_par, verbose=-1)
            model.fit(X_train, y, sample_weight=sample_w)
            joblib.dump(model, mdl_path)

        # prediction
        p = model.predict(X_pred)

        if is_rate_target:
            # -------- blend con prior "sano" del rate per90 --------
            # Priors calcolati su tutto role_df (train+pred), poi seleziono pred_idx
            pri_all = _build_rate_priors(role_df, rate_col=f"{target}_per90")
            pri = pri_all.loc[pred_df.index, :].copy()

            healthy_mins = pri["club_healthy_minutes_prev"].fillna(0.0)
            last_club    = pri["last_minutes_same_club"].fillna(np.nan)
            tenure       = pri["tenure_at_club"].fillna(0).astype(float)

            override_mask = (
                (tenure >= 2)
                & (last_club.notna())
                & (last_club < OUTLIER_LAST_CUTOFF)
                & (healthy_mins >= HEALTHY_OK_MIN)
            )

            prior_rate = pri["club_rate_prev"].copy()
            missing = prior_rate.isna()
            prior_rate[missing] = pri["player_rate_prev"][missing]

            # fallback: mediana per ruolo nel train
            role_median = float(train_df[f"{target}_per90"].median()) if f"{target}_per90" in train_df else 0.0
            prior_rate = prior_rate.fillna(role_median).clip(lower=0)

            # severità vs healthy_mins (stessa logica di minutes)
            last_for_severity = last_club.fillna(0.0)
            denom = np.maximum(healthy_mins, 1.0)
            severity = np.clip((denom - last_for_severity) / denom, 0.0, 1.0)
            k = BLEND_MAX * (1.0 / (1.0 + np.exp(-BLEND_STEEPNESS * (severity - BLEND_CENTER))))

            base_rate = np.clip(p, 1e-9, np.inf)
            blended_rate = (1 - k) * base_rate + k * prior_rate
            blended_rate[override_mask] = prior_rate[override_mask]

            # rate -> conteggio usando i minuti PREVISTI
            minutes_pred = pred_df["min_playing_time"].fillna(0.0)
            p = blended_rate * (minutes_pred / 90.0)

        pred_list.append(pd.Series(p, index=pred_df.index, name=f"{target}_pred"))

    return pd.concat(pred_list).sort_index()
