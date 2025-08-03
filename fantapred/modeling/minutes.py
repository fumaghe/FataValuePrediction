from __future__ import annotations
from pathlib import Path
from typing import List

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS

# ------------------------------------------------------------------ #
#  Depth-features = titolari (≥1800′) dell’anno precedente
# ------------------------------------------------------------------ #
def _depth_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_season = (
        df["season"]
        .str.extract(r"s_(\d+)_\d+")[0].astype(int).sub(1).astype(str)
        .radd("s_")
        .str.cat(df["season"].str[-2:], sep="_")
    )
    base = df.assign(season_prev=prev_season)

    tit = base[base["min_playing_time"] >= 1800]

    return (
        tit.groupby(["team_name_short", "season_prev", "role"], as_index=False)
           .agg(
               players1800_prev=("player_id", "size"),
               mv_mean_prev    =("mv", "mean"),
           )
    )

# ------------------------------------------------------------------ #
#  Utility per addestrare un singolo LightGBM e fare predict
# ------------------------------------------------------------------ #
def _fit_predict(
    Xtr: pd.DataFrame, ytr: pd.Series, Xfu: pd.DataFrame,
    n_estim: int = 500, seed: int = 42
) -> np.ndarray:
    mdl = lgb.LGBMRegressor(
        num_leaves=63,
        learning_rate=0.05,
        n_estimators=n_estim,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        **GPU_PARAMS,
    )
    mdl.fit(Xtr, ytr)
    return mdl.predict(Xfu)

# ------------------------------------------------------------------ #
#  MAIN: minutes + starts + presenze
# ------------------------------------------------------------------ #
def minutes_regressor(
    combined: pd.DataFrame,
    train_until: str,
    models_dir: Path | None = None,   # (non persistiamo qui i modelli)
) -> pd.DataFrame:

    df = combined.copy()

    # ---------------- depth-features ---------------- #
    depth = _depth_features(df)
    df = (
        df.merge(
            depth,
            left_on=["team_name_short", "season", "role"],
            right_on=["team_name_short", "season_prev", "role"],
            how="left",
        )
        .drop(columns="season_prev")
    )
    df["players1800_prev"] = df["players1800_prev"].fillna(0)
    df["mv_mean_prev"]     = df["mv_mean_prev"].fillna(df["mv"].median())

    # ---------------- gestione starts_eleven basata su presenze ---------------- #
    # se manca starts_eleven ma presenze > 0, usiamo presenze/2
    mask = df["starts_eleven"].isna() & df["presenze"].fillna(0) > 0
    df.loc[mask, "starts_eleven"] = df.loc[mask, "presenze"] / 2
    # lasciamo intatti eventuali NaN di starts_eleven se presenze è NaN o =0

    # ---------------- split train / future ---------------- #
    tr_df = df[df.season <= train_until].copy()
    fu_df = df[df.season >  train_until].copy()

    if tr_df.empty or fu_df.empty:
        df["presenze_pred"]  = df["presenze"].fillna(0)
        df["starts_pred"]    = df["starts_eleven"].fillna(0)
        df["titolare_pred"]  = (df["starts_pred"] >= 19).astype(int)
        return df

    # ---------------- feature set ---------------- #
    base_num: List[str] = [
        c for c in tr_df.select_dtypes(include="number").columns
        if c not in LEAK_COLS + ["min_playing_time", "starts_eleven", "presenze"]
    ]
    num_cols  = list(dict.fromkeys(base_num + ["players1800_prev", "mv_mean_prev"]))

    feat_cols = num_cols + CAT_COLS + ["role"]

    Xtr = tr_df[feat_cols].copy()
    Xfu = fu_df[feat_cols].copy()
    for c in CAT_COLS + ["role"]:
        Xtr[c] = Xtr[c].astype("category")
        Xfu[c] = Xfu[c].astype("category")

    # 1) minuti giocati
    y_min = tr_df["min_playing_time"].fillna(0)
    df.loc[fu_df.index, "min_playing_time"] = _fit_predict(Xtr, y_min, Xfu)

    # 2) starts_eleven
    # (i NaN rimasti, ovvero con presenze=0 o mancanti, diventeranno 0 in fillna)
    y_st = tr_df["starts_eleven"].fillna(0)
    df.loc[fu_df.index, "starts_pred"] = _fit_predict(
        Xtr, y_st, Xfu, n_estim=400, seed=99
    )

    # 3) presenze totali (dalla colonna “presenze” originale)
    y_pr = tr_df["presenze"].fillna(0)
    df.loc[fu_df.index, "presenze_pred"] = _fit_predict(
        Xtr, y_pr, Xfu, n_estim=450, seed=123
    )

    # flag titolare
    df["starts_pred"]     = df["starts_pred"].round(0)
    df["presenze_pred"]   = df["presenze_pred"].round(0).clip(lower=0)
    df["titolare_pred"]   = (df["starts_pred"] >= 19).astype(int)

    return df
