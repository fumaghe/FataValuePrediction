from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
import lightgbm as lgb

from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS

__all__ = ["minutes_regressor", "fit_predict_minutes"]

# ------------------------------------------------------------------ #
#  Depth-features = titolari (≥1800′) dell'anno precedente
# ------------------------------------------------------------------ #
def _depth_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea feature sull'anno precedente per ciascun (team, season) con padding corretto.
    - players1800_prev: # di giocatori con >=1800' nella stagione precedente
    - mv_mean_prev: media voto della stagione precedente
    """
    # estrae YY iniziale, calcola YY precedente e mantiene il padding a 2 cifre
    yy_start = df["season"].str.extract(r"s_(\d{2})_\d{2}")[0].astype(int)
    yy_prev  = (yy_start - 1) % 100
    prev_season = "s_" + yy_prev.astype(str).str.zfill(2) + "_" + df["season"].str[-2:]

    key_prev = df["team_name_short"].astype(str) + "_" + prev_season

    # costruiamo indici team_season per l'anno "corrente" delle righe esistenti,
    # così possiamo aggregare ciò che è successo in quell'anno (che diventa "prev" per la riga target)
    team_season = df["team_name_short"].astype(str) + "_" + df["season"].astype(str)

    # 1) numero titolari (>= 1800')
    starters_prev = (
        df.assign(team_season=team_season)
          .loc[lambda x: x["min_playing_time"].fillna(0) >= 1800]
          .groupby("team_season")["player_id"].nunique()
    )
    df["players1800_prev"] = key_prev.map(starters_prev).fillna(0).astype(int)

    # 2) media voto dell'anno precedente
    mv_prev = (
        df.assign(team_season=team_season)
          .groupby("team_season")["mv"].mean()
    )
    df["mv_mean_prev"] = key_prev.map(mv_prev).fillna(6.0)

    return df

# ------------------------------------------------------------------ #
#  LGBM helper
# ------------------------------------------------------------------ #
def _fit_predict(
    Xtr: pd.DataFrame, ytr: pd.Series, Xfu: pd.DataFrame, *, n_estim: int = 400, seed: int = 42
) -> np.ndarray:
    params = dict(
        boosting_type="gbdt",
        learning_rate=0.05,
        n_estimators=n_estim,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        **GPU_PARAMS,
    )
    mdl = lgb.LGBMRegressor(**params, objective="mae", verbose=-1)
    # IMPORTANT: passiamo le categoriche a .fit(), NON nei params
    mdl.fit(Xtr, ytr, categorical_feature=CAT_COLS + ["role"])
    p = mdl.predict(Xfu)
    return np.clip(p, 0, None)

# ------------------------------------------------------------------ #
#  Regressione minuti, titolarità e presenze (sulla stagione futura)
# ------------------------------------------------------------------ #
def fit_predict_minutes(df: pd.DataFrame, train_until: str) -> pd.DataFrame:
    """
    Predice:
      - min_playing_time (sostituisce la colonna sulla stagione futura)
      - starts_pred     (num. titolarità)
      - presenze_pred   (num. presenze)
      - titolare_pred   (flag: starts_pred >= 19)

    Ritorna un nuovo DataFrame con i campi sopra popolati per le righe future.
    """
    # depth features su tutto df (servono mappe "prev" → "curr")
    df = _depth_features(df.copy())

    # split
    tr_df = df[df["season"] <= train_until].copy()
    fu_df = df[df["season"]  > train_until].copy()

    # selezione colonne numeriche senza leak
    num_cols = [c for c in tr_df.select_dtypes(include="number").columns if c not in LEAK_COLS]
    feat_cols = list(dict.fromkeys(num_cols + CAT_COLS + ["role"]))

    Xtr = tr_df[feat_cols].copy()
    Xfu = fu_df[feat_cols].copy()

    for c in CAT_COLS + ["role"]:
        if c in Xtr.columns:
            Xtr[c] = Xtr[c].astype("category")
            Xfu[c] = Xfu[c].astype("category")

    # 1) minuti giocati → scriviamo direttamente nella colonna usata a valle
    y_min = tr_df["min_playing_time"].fillna(0)
    df.loc[fu_df.index, "min_playing_time"] = _fit_predict(Xtr, y_min, Xfu)

    # 2) titolarità (starts_eleven)
    y_st = tr_df["starts_eleven"].fillna(0)
    df.loc[fu_df.index, "starts_pred"] = _fit_predict(Xtr, y_st, Xfu, n_estim=400, seed=99).round(0)

    # 3) presenze
    y_pr = tr_df["presenze"].fillna(0)
    df.loc[fu_df.index, "presenze_pred"] = _fit_predict(Xtr, y_pr, Xfu, n_estim=450, seed=123).round(0)

    # flags & clip
    df["presenze_pred"] = df["presenze_pred"].clip(lower=0)
    df["titolare_pred"] = (df["starts_pred"] >= 19).astype(int)

    return df

# ------------------------------------------------------------------ #
#  BACKWARD-COMPAT: vecchio nome usato da cli.py e modeling/__init__.py
# ------------------------------------------------------------------ #
def minutes_regressor(df: pd.DataFrame, train_until: str) -> pd.DataFrame:
    """
    Wrapper retro-compatibile. Mantiene la vecchia API chiamata dal CLI.
    """
    return fit_predict_minutes(df, train_until)
