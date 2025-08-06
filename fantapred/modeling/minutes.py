from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import lightgbm as lgb

from ..settings import CAT_COLS, LEAK_COLS, GPU_PARAMS


# =========================
# Parametri regolabili
# =========================
HEALTHY_RATIO_CLUB = 0.60   # >= 60% della mediana club -> "sano"
HEALTHY_FLOOR_MIN  = 1200.0 # oppure almeno 1200'
OUTLIER_LAST_CUTOFF = 1000.0# ultima stagione nel club <1000' => probabile infortunio/outlier
HEALTHY_OK_MIN     = 1800.0 # healthy-median club abbastanza alto da fidarsi

# blending (quanto tirare verso il prior quando l'ultima è bassa ma non scatta override)
BLEND_MAX          = 0.75   # tiro massimo verso il prior
BLEND_STEEPNESS    = 3.0    # ripidità della sigmoide
BLEND_CENTER       = 0.50   # centro della sigmoide sulla severità

# training weights (morbidi, servono a non farsi "tirare giù" dagli outlier)
W_MIN        = 0.05
ALPHA_Z      = 0.8
EPS_MAD      = 120.0
RECENCY_LMB  = 0.25

# tasso da titolare di fallback per ruolo
DEFAULT_START_RATE_ROLE = {"P": 0.92, "D": 0.72, "C": 0.68, "A": 0.62}

# ============================================================
# Utility
# ============================================================
def _season_year(s: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(s)
    y = s.astype(str).str.extract(r"s_(\d{2})_\d{2}")[0].astype(float)
    return (2000 + y).astype("Int64")

def _healthy_mask(minutes: pd.Series, club_median: pd.Series) -> pd.Series:
    """
    Series booleana allineata: True se la stagione è "sana".
    """
    thr = np.maximum(HEALTHY_RATIO_CLUB * club_median.fillna(0.0), HEALTHY_FLOOR_MIN)
    mask = minutes.fillna(0.0).ge(thr)
    return mask.astype(bool)

def _adaptive_sample_weights(df: pd.DataFrame, minutes_col: str = "min_playing_time") -> pd.Series:
    """ Pesi continui: base (minuti) * robustezza (z rispetto mediana club) * recency """
    m   = df[minutes_col].fillna(0.0)
    grp = df.groupby(["player_id", "team_name_short"], observed=True)
    med = grp[minutes_col].transform("median").fillna(0.0)
    mad = grp[minutes_col].transform(lambda x: (x - x.median()).abs().median()).fillna(0.0)
    mad = 1.4826 * mad + EPS_MAD

    z   = (m - med).abs() / mad
    outlier_w = W_MIN + (1 - W_MIN) * np.exp(-ALPHA_Z * z)
    outlier_w = outlier_w.clip(lower=W_MIN, upper=1.0)

    year      = _season_year(df["season"]).astype(float)
    last_year = grp["season"].transform(lambda s: _season_year(s).max()).astype(float)
    steps     = (last_year - year).clip(lower=0)
    recency   = np.exp(-RECENCY_LMB * steps)

    base = m / 90.0 + 1e-3
    return (base * outlier_w * recency).clip(lower=W_MIN)


# ============================================================
# Priors "sani" (shifted) di club e giocatore
# ============================================================
def _shifted_median(vals: pd.Series, ok_mask: pd.Series) -> pd.Series:
    """
    Mediana delle sole osservazioni "sane" PRECEDENTI (shift by 1) per ogni riga.
    vals e ok_mask devono avere stesso index/ordine.
    """
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

def _safe_rate(starts: pd.Series, pres: pd.Series) -> pd.Series:
    return (starts.fillna(0) / pres.replace(0, np.nan)).clip(0, 1)

def _build_healthy_priors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge:
      - club_healthy_median_prev
      - player_healthy_median_prev
      - starter_rate_club_prev
      - starter_rate_player_prev
      - last_minutes_same_club
      - tenure_at_club (n stagioni prima di questa nel club)
    Tutto SHIFTED: per ogni riga consideriamo solo stagioni precedenti.
    """
    df = df.copy()
    df["year"] = _season_year(df["season"]).astype(int)
    df.sort_values(["player_id", "team_name_short", "year"], inplace=True)

    grp_pc = df.groupby(["player_id", "team_name_short"], observed=True)

    # mediana minuti (serve a costruire la mask)
    club_median = grp_pc["min_playing_time"].transform("median").fillna(0.0)
    healthy = _healthy_mask(df["min_playing_time"], club_median)

    # club_healthy_median_prev (shifted mediana "sana" nel club)
    df["club_healthy_median_prev"] = (
        df.groupby(["player_id", "team_name_short"], observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(g["min_playing_time"], healthy.loc[g.index]))
    )

    # player_healthy_median_prev (shifted mediana "sana" su tutti i club)
    df["player_healthy_median_prev"] = (
        df.groupby("player_id", observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(g["min_playing_time"], healthy.loc[g.index]))
    )

    # starter_rate (shifted median solo su stagioni sane)
    rate_series = _safe_rate(df["starts_eleven"], df["presenze"])
    df["starter_rate_club_prev"] = (
        df.groupby(["player_id", "team_name_short"], observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(rate_series.loc[g.index], healthy.loc[g.index]))
    )
    df["starter_rate_player_prev"] = (
        df.groupby("player_id", observed=True, group_keys=False)
          .apply(lambda g: _shifted_median(rate_series.loc[g.index], healthy.loc[g.index]))
    )

    # last minutes same club (shift semplice nel club)
    df["last_minutes_same_club"] = (
        df.groupby(["player_id", "team_name_short"], observed=True)["min_playing_time"]
          .shift(1)
    )

    # tenure nel club (quante stagioni precedenti)
    df["tenure_at_club"] = grp_pc.cumcount()

    return df


# ============================================================
# Depth-features: titolari (≥1800′) dell’anno precedente
# ============================================================
def _depth_features(df: pd.DataFrame) -> pd.DataFrame:
    prev_season = (
        df["season"].str.extract(r"s_(\d{2})_\d{2}")[0].astype(int).sub(1).astype(str)
        .radd("s_").str.cat(df["season"].str[-2:], sep="_")
    )
    base = df.assign(season_prev=prev_season)
    tit = base[base["min_playing_time"] >= 1800]
    return (
        tit.groupby(["team_name_short", "season_prev", "role"], as_index=False)
           .agg(players1800_prev=("player_id", "size"),
                mv_mean_prev=("mv", "mean"))
    )


# ============================================================
# Fit/predict utility
# ============================================================
def _fit_predict(
    Xtr: pd.DataFrame, ytr: pd.Series, Xfu: pd.DataFrame,
    sample_weight: pd.Series | None = None,
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
    mdl.fit(Xtr, ytr, sample_weight=sample_weight)
    return mdl.predict(Xfu)


# ============================================================
# MAIN
# ============================================================
def minutes_regressor(
    combined: pd.DataFrame,
    train_until: str,
    models_dir: Path | None = None,
) -> pd.DataFrame:

    df = combined.copy()

    # depth-features
    depth = _depth_features(df)
    df = (
        df.merge(depth,
                 left_on=["team_name_short", "season", "role"],
                 right_on=["team_name_short", "season_prev", "role"],
                 how="left")
          .drop(columns="season_prev")
    )
    df["players1800_prev"] = df["players1800_prev"].fillna(0)
    df["mv_mean_prev"]     = df["mv_mean_prev"].fillna(df["mv"].median())

    # starts_eleven derivato se mancante
    mask = df["starts_eleven"].isna() & df["presenze"].fillna(0).gt(0)
    df.loc[mask, "starts_eleven"] = df.loc[mask, "presenze"] / 2

    # ---------- priors sani SHIFTED + info ultima stagione club ----------
    pri = _build_healthy_priors(df)
    for c in [
        "club_healthy_median_prev", "player_healthy_median_prev",
        "starter_rate_club_prev", "starter_rate_player_prev",
        "last_minutes_same_club", "tenure_at_club"
    ]:
        df[c] = pri[c]

    # split
    tr_df = df[df.season <= train_until].copy()
    fu_df = df[df.season >  train_until].copy()

    if tr_df.empty or fu_df.empty:
        df["presenze_pred"]  = df["presenze"].fillna(0)
        df["starts_pred"]    = df["starts_eleven"].fillna(0)
        df["titolare_pred"]  = (df["starts_pred"] >= 19).astype(int)
        return df

    # feature set
    base_num: List[str] = [
        c for c in tr_df.select_dtypes(include="number").columns
        if c not in LEAK_COLS + ["min_playing_time", "starts_eleven", "presenze"]
    ]
    num_cols = list(dict.fromkeys(base_num + [
        "players1800_prev", "mv_mean_prev",
        "club_healthy_median_prev", "player_healthy_median_prev", "tenure_at_club"
    ]))
    feat_cols = num_cols + CAT_COLS + ["role"]

    Xtr = tr_df[feat_cols].copy()
    Xfu = fu_df[feat_cols].copy()
    for c in CAT_COLS + ["role"]:
        if c in Xtr:
            Xtr[c] = Xtr[c].astype("category")
            Xfu[c] = Xfu[c].astype("category")

    # pesi adattivi per il training (anti-outlier)
    w_tr = _adaptive_sample_weights(tr_df, minutes_col="min_playing_time")

    # 1) minuti giocati
    y_min = tr_df["min_playing_time"].fillna(0)
    fu_minutes = _fit_predict(Xtr, y_min, Xfu, sample_weight=w_tr)

    # 2) starts_eleven
    y_st = tr_df["starts_eleven"].fillna(0)
    fu_starts = _fit_predict(Xtr, y_st, Xfu, sample_weight=w_tr, n_estim=420, seed=99)

    # 3) presenze totali
    y_pr = tr_df["presenze"].fillna(0)
    fu_pres = _fit_predict(Xtr, y_pr, Xfu, sample_weight=w_tr, n_estim=450, seed=123)

    # =====================================================
    # Post-adjustment sui FUTURE usando i priors sani
    # =====================================================
    fu = df.loc[fu_df.index, [
        "player_id", "team_name_short", "role",
        "club_healthy_median_prev", "player_healthy_median_prev",
        "starter_rate_club_prev", "starter_rate_player_prev",
        "last_minutes_same_club", "tenure_at_club"
    ]].copy()

    # 1) override netto quando pattern club è chiarissimo (per presenze)
    healthy_club = fu["club_healthy_median_prev"].fillna(0.0)
    last_club    = fu["last_minutes_same_club"].fillna(np.nan)
    tenure       = fu["tenure_at_club"].fillna(0).astype(float)

    override_mask = (
        (tenure >= 2)
        & (last_club.notna())
        & (last_club < OUTLIER_LAST_CUTOFF)
        & (healthy_club >= HEALTHY_OK_MIN)
    )
    override_matches = np.round(healthy_club[override_mask] / 90.0).clip(0, 38)

    # 2) blending adattivo per presenze (prior club se c'è, altrimenti player)
    prior_minutes = healthy_club.copy()
    use_player_prior = prior_minutes.isna() | (prior_minutes <= 0)
    prior_minutes[use_player_prior] = fu["player_healthy_median_prev"][use_player_prior].fillna(0.0)

    last_for_severity = last_club.fillna(0.0)  # se nuovo club, 0 => spinta verso prior
    denom = np.maximum(prior_minutes, 1.0)
    severity = np.clip((denom - last_for_severity) / denom, 0.0, 1.0)  # 0=ok, 1=molto bassa
    k = BLEND_MAX * (1.0 / (1.0 + np.exp(-BLEND_STEEPNESS * (severity - BLEND_CENTER))))

    base_matches  = np.clip(np.round(fu_pres.copy()), 0, 38).astype(float)
    prior_matches = np.clip(np.round(prior_minutes / 90.0), 0, 38).astype(float)
    blended_matches = np.round((1 - k) * base_matches + k * prior_matches).clip(0, 38)

    # applica override dove attivo
    final_matches = blended_matches.copy()
    final_matches[override_mask] = override_matches

    # ----------- STARTS: blend del tasso da titolare -----------
    base_rate = (fu_starts / np.maximum(fu_pres, 1e-6)).clip(0, 1)
    prior_rate = fu["starter_rate_club_prev"].copy()
    missing = prior_rate.isna()
    prior_rate[missing] = fu["starter_rate_player_prev"][missing]
    # fallback per ruolo
    role_map = fu["role"].map(DEFAULT_START_RATE_ROLE).fillna(0.65)
    prior_rate = prior_rate.fillna(role_map).clip(0, 1)

    k_s = k  # stessa severità delle presenze
    blended_rate = ((1 - k_s) * base_rate + k_s * prior_rate).clip(0, 1)
    # se override attivo, usa direttamente il prior rate
    blended_rate[override_mask] = prior_rate[override_mask]

    final_starts = np.round(final_matches * blended_rate).clip(0, 38)
    final_starts = np.minimum(final_starts, final_matches)

    # scrivi nel df
    df.loc[fu_df.index, "presenze_pred"] = final_matches
    df.loc[fu_df.index, "starts_pred"]   = final_starts
    df.loc[fu_df.index, "min_playing_time"] = np.clip(np.round(final_matches * 90.0), 0, 38 * 90.0)
    df["titolare_pred"] = (df["starts_pred"] >= 19).astype(int)

    return df
