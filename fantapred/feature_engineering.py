from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List

from .settings import (
    RECENCY_WEIGHTS,
    AGE_DECAY,
    GROWTH_CAP,
    DECLINE_CAP,
    LEAGUE_COEF,
    DEFAULT_LEAGUE_COEF,
    TEAM_ROLE_SCORE,
    TEAM_DEFAULT_ROLE_COEF,  # lasciato se usato altrove
    SCORE_SCALE,
)

# ------------------------------------------------------------------ #
#  Helper functions
# ------------------------------------------------------------------ #
def _to_numeric_clean(s: pd.Series) -> pd.Series:
    """
    Converte una serie potenzialmente 'sporca' in float:
    - gestisce virgola decimale,
    - estrae solo la parte numerica da stringhe tipo '30 min',
    - ritorna NaN per non-numero.
    """
    if s is None:
        return pd.Series(dtype="float64")
    s = s.astype(str).str.replace(",", ".", regex=False)
    s = s.str.extract(r"([-+]?\d*\.?\d+)")[0]
    return pd.to_numeric(s, errors="coerce")


def _exp_weighted_mean(values: List[float], weights: List[float]) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return 0.0
    w = np.array(weights[: len(vals)][::-1], dtype=float)
    v = vals[::-1]
    sw = w.sum()
    if sw == 0:
        return float(np.nanmean(v)) if v.size else 0.0
    w /= sw
    return float(np.dot(v, w))


def _apply_growth_decline_cap(value: float, last_val: float | None, stat: str) -> float:
    if last_val is None or np.isnan(last_val):
        return float(value)
    up   = last_val * (1 + GROWTH_CAP.get(stat, 0.3))
    down = last_val * (1 - DECLINE_CAP.get(stat, 0.5))
    return float(np.clip(value, down, up))


# ------------------------------------------------------------------ #
#  Team-role coefficient lookup (normalized 0–5 into 1±scale)
# ------------------------------------------------------------------ #
def _team_role_coef(team: str, role: str) -> float:
    tr = TEAM_ROLE_SCORE.get(team, {}).get(role, (0, 0))
    try:
        adv, dis = tr
    except Exception:
        adv, dis = 0, 0
    delta = (adv or 0) - (dis or 0)
    return 1.0 + SCORE_SCALE * delta


# ------------------------------------------------------------------ #
#  Feature factory
# ------------------------------------------------------------------ #
def build_features(df: pd.DataFrame, today_year: int = 2025) -> pd.DataFrame:
    df = df.copy()

    # --- Anagrafica ---
    df["birth_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year
    df["age"]        = today_year - df["birth_year"]
    df["age_sq"]     = df["age"] ** 2

    # --- Coefficiente difficoltà lega ---
    df["league_coef"] = (
        df["tournament_name"].map(LEAGUE_COEF).fillna(DEFAULT_LEAGUE_COEF)
    )

    # --- Coefficiente allenatore × ruolo ---
    df["team_role_coef"] = df.apply(
        lambda r: _team_role_coef(r.get("team_name_short", ""), r.get("role", "")),
        axis=1,
    )

    # --- Normalizza colonne numeriche usate a valle ---
    numeric_candidates = {
        "gf", "assist", "shots", "xg", "xg_on_target",
        "passes", "cross", "duels", "clean_sheet",
        "presenze", "starts_eleven",
        "shots_on_target", "total_shots",
        "fmv", "mv", "min_playing_time",
    } & set(df.columns)

    for col in numeric_candidates:
        if df[col].dtype == object:
            df[col] = _to_numeric_clean(df[col])
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- Per-90 ---
    mins = df.get("min_playing_time")
    if mins is None:
        mins = pd.Series(0.0, index=df.index, dtype="float64")
    df["minutes90"] = (mins.fillna(0.0) / 90.0).astype(float)
    denom = df["minutes90"].replace(0, np.nan)  # evita div/0

    per90_base_cols = {
        "gf", "assist", "shots", "xg", "xg_on_target",
        "passes", "cross", "duels", "clean_sheet",
    } & set(df.columns)

    for col in per90_base_cols:
        vals = df[col].astype(float)
        df[f"{col}_per90"] = vals / denom

    # --- Forza squadra grezza (media gf per team e stagione) ---
    if "gf" in df.columns:
        df["team_season_gf_mean"] = (
            df.groupby(["season", "team_name_short"])["gf"].transform("mean")
        )
    else:
        df["team_season_gf_mean"] = 0.0

    # rating attacco con pesi di recency (ultime 3 stagioni note per team)
    df["team_attack_rating"] = (
        df.groupby("team_name_short")["team_season_gf_mean"]
          .transform(lambda s: _exp_weighted_mean(s.tail(3).tolist(), RECENCY_WEIGHTS))
    )

    # --- Forza attacco aggiustata ---
    df["attack_rating_adj"] = (
        df["team_attack_rating"] * df["league_coef"] * df["team_role_coef"]
    )

    # --- Trend personale fmv ---
    df = df.sort_values(["player_id", "season"])
    if "fmv" in df.columns:
        df["fmv_prev"]  = df.groupby("player_id")["fmv"].shift(1)
        df["fmv_delta"] = df["fmv"] - df["fmv_prev"]
    else:
        df["fmv_prev"] = 0.0
        df["fmv_delta"] = 0.0

    # --- Precisione tiro ---
    if {"shots_on_target", "total_shots"}.issubset(df.columns):
        denom_sh = df["total_shots"].replace(0, np.nan)
        df["shot_accuracy"] = df["shots_on_target"] / denom_sh
    else:
        df["shot_accuracy"] = 0.0

    # Riporta NaN “innocui” a 0 dove ha senso per la fase successiva
    return df.fillna(0)


# ------------------------------------------------------------------ #
#  Projection helpers (unchanged in logica, robusti nei tipi)
# ------------------------------------------------------------------ #
def project_player_row(hist: pd.DataFrame, next_season: str) -> pd.Series:
    last = hist.iloc[-1].copy()
    role = last.get("role", "")
    age_n = (last.get("age") or 0) + 1
    proj = last.copy()
    proj["season"] = next_season
    proj["age"]    = age_n
    proj["age_sq"] = age_n ** 2

    # Assicurati numerici per le principali
    for col in ["fmv", "mv", "gf", "assist", "clean_sheet", "min_playing_time", "team_season_gf_mean"]:
        if col in hist.columns:
            if hist[col].dtype == object:
                hist[col] = _to_numeric_clean(hist[col])
            else:
                hist[col] = pd.to_numeric(hist[col], errors="coerce")

    main_stats = ["fmv", "mv", "gf", "assist", "clean_sheet"]
    for stat in main_stats:
        if stat in hist.columns:
            series = hist[stat].dropna().tolist()
        else:
            series = []
        val = _exp_weighted_mean(series[-3:], RECENCY_WEIGHTS)
        if len(series) >= 3:
            mu, sigma = float(np.mean(series[-3:])), float(np.std(series[-3:]))
            if sigma > 0 and abs(series[-1] - mu) > 3 * sigma:
                val = mu + np.sign(series[-1] - mu) * sigma
        last_val = series[-1] if series else None
        val = _apply_growth_decline_cap(val, last_val, stat)
        proj[stat] = val

    dcfg = AGE_DECAY.get(role)
    if dcfg and age_n > dcfg.get("age_thr", 10**9):
        factor = max(0.0, 1 - dcfg.get("decay", 0.0) * (age_n - dcfg.get("age_thr", 0)))
        for stat in ["fmv", "mv", "gf", "assist"]:
            if stat in proj:
                proj[stat] = float(proj[stat]) * factor

    if "min_playing_time" in hist.columns:
        proj["min_playing_time"] = _exp_weighted_mean(
            hist["min_playing_time"].tail(3).tolist(), RECENCY_WEIGHTS
        )

    if len(hist) >= 2:
        last_team = hist.iloc[-1].get("team_name_short", "")
        prev_team = hist.iloc[-2].get("team_name_short", "")
        proj["moved_team"] = int(last_team != prev_team)
        tsgm_last = float(hist.iloc[-1].get("team_season_gf_mean", 0) or 0)
        tsgm_prev = float(hist.iloc[-2].get("team_season_gf_mean", 0) or 0)
        proj["team_attack_delta"] = tsgm_last - tsgm_prev
    else:
        proj["moved_team"] = 0
        proj["team_attack_delta"] = 0.0

    return proj


def build_future_dataframe(
    df: pd.DataFrame, train_until: str, next_season: str
) -> pd.DataFrame:
    """
    Crea la proiezione della riga futura SOLO per i giocatori che
    **non** hanno già una riga per `next_season` (evita duplicati
    quando il CSV contiene già placeholder della stagione successiva).
    """
    rows: list[pd.Series] = []

    already_present = set(df.loc[df["season"] == next_season, "player_id"])

    for pid, hist in df.sort_values(["player_id", "season"]).groupby("player_id"):
        if pid in already_present:
            continue
        htrain = hist[hist["season"] <= train_until]
        if htrain.empty:
            continue
        rows.append(project_player_row(htrain, next_season))

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=df.columns)
