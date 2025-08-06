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
    TEAM_DEFAULT_ROLE_COEF,
    SCORE_SCALE,
)

# ------------------------------------------------------------------ #
#  Helper functions
# ------------------------------------------------------------------ #
def _exp_weighted_mean(values: List[float], weights: List[float]) -> float:
    if not values:
        return 0.0
    w = np.array(weights[: len(values)][::-1], dtype=float)
    v = np.array(values[::-1], dtype=float)
    w /= w.sum()
    return float(np.dot(v, w))


def _apply_growth_decline_cap(value: float, last_val: float | None, stat: str) -> float:
    if last_val is None or np.isnan(last_val):
        return value
    up   = last_val * (1 + GROWTH_CAP.get(stat, 0.3))
    down = last_val * (1 - DECLINE_CAP.get(stat, 0.5))
    return float(np.clip(value, down, up))


# ------------------------------------------------------------------ #
#  Team-role coefficient lookup (normalized 0–5 into 1±scale)
# ------------------------------------------------------------------ #
def _team_role_coef(team: str, role: str) -> float:
    adv, dis = TEAM_ROLE_SCORE.get(team, {}).get(role, (0, 0))
    delta = adv - dis
    return 1.0 + SCORE_SCALE * delta


# ------------------------------------------------------------------ #
#  Feature factory
# ------------------------------------------------------------------ #
def build_features(df: pd.DataFrame, today_year: int = 2025) -> pd.DataFrame:
    df = df.copy()

    # Anagrafica
    df["birth_year"] = pd.to_datetime(df["date_of_birth"], errors="coerce").dt.year
    df["age"]        = today_year - df["birth_year"]
    df["age_sq"]     = df["age"] ** 2

    # Coefficiente difficoltà lega
    df["league_coef"] = (
        df["tournament_name"]
          .map(LEAGUE_COEF)
          .fillna(DEFAULT_LEAGUE_COEF)
    )

    # Coefficiente allenatore × ruolo
    df["team_role_coef"] = df.apply(
        lambda r: _team_role_coef(r["team_name_short"], r["role"]),
        axis=1,
    )

    # Per-90
    df["minutes90"] = df["min_playing_time"].fillna(0) / 90.0
    per90_suffixes = ["gf", "assist", "shots", "xg", "xg_on_target",
                      "passes", "cross", "duels", "clean_sheet"]
    for col in [c for c in df.columns if any(c.endswith(s) for s in per90_suffixes)]:
        df[f"{col}_per90"] = df[col] / df["minutes90"].replace(0, np.nan)

    # Forza squadra grezza
    if "gf" in df.columns:
        df["team_season_gf_mean"] = (
            df.groupby(["season", "team_name_short"])["gf"].transform("mean")
        )
    df["team_attack_rating"] = (
        df.groupby("team_name_short")["team_season_gf_mean"]
          .transform(lambda s: _exp_weighted_mean(s.tail(3).tolist(), RECENCY_WEIGHTS))
    )

    # Forza attacco aggiustata
    df["attack_rating_adj"] = (
        df["team_attack_rating"]
        * df["league_coef"]
        * df["team_role_coef"]
    )

    # Trend personale fmv
    df = df.sort_values(["player_id", "season"])
    df["fmv_prev"]  = df.groupby("player_id")["fmv"].shift(1)
    df["fmv_delta"] = df["fmv"] - df["fmv_prev"]

    # Precisione tiro
    if {"shots_on_target", "total_shots"}.issubset(df.columns):
        df["shot_accuracy"] = (
            df["shots_on_target"] / df["total_shots"].replace(0, np.nan)
        )

    return df.fillna(0)


# ------------------------------------------------------------------ #
#  Projection helpers (unchanged)
# ------------------------------------------------------------------ #
def project_player_row(hist: pd.DataFrame, next_season: str) -> pd.Series:
    last = hist.iloc[-1].copy()
    role = last["role"]
    age_n = last["age"] + 1
    proj = last.copy()
    proj["season"] = next_season
    proj["age"]    = age_n
    proj["age_sq"] = age_n ** 2

    main_stats = ["fmv", "mv", "gf", "assist", "clean_sheet"]
    for stat in main_stats:
        series = hist[stat].dropna().tolist()
        val = _exp_weighted_mean(series[-3:], RECENCY_WEIGHTS)
        if len(series) >= 3:
            mu, sigma = np.mean(series[-3:]), np.std(series[-3:])
            if sigma > 0 and abs(series[-1] - mu) > 3 * sigma:
                val = mu + np.sign(series[-1] - mu) * sigma
        val = _apply_growth_decline_cap(val, series[-1] if series else None, stat)
        proj[stat] = val

    dcfg = AGE_DECAY.get(role)
    if dcfg and age_n > dcfg["age_thr"]:
        factor = max(0.0, 1 - dcfg["decay"] * (age_n - dcfg["age_thr"]))
        for stat in ["fmv", "mv", "gf", "assist"]:
            proj[stat] *= factor

    if "min_playing_time" in hist.columns:
        proj["min_playing_time"] = _exp_weighted_mean(
            hist["min_playing_time"].tail(3).tolist(), RECENCY_WEIGHTS
        )

    if len(hist) >= 2:
        last_team = hist.iloc[-1]["team_name_short"]
        prev_team = hist.iloc[-2]["team_name_short"]
        proj["moved_team"] = int(last_team != prev_team)
        proj["team_attack_delta"] = (
            hist.iloc[-1]["team_season_gf_mean"]
            - hist.iloc[-2]["team_season_gf_mean"]
        )
    else:
        proj["moved_team"] = 0
        proj["team_attack_delta"] = 0.0

    return proj


def build_future_dataframe(
    df: pd.DataFrame, train_until: str, next_season: str
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for pid, hist in df.sort_values(["player_id", "season"]).groupby("player_id"):
        htrain = hist[hist["season"] <= train_until]
        if htrain.empty:
            continue
        rows.append(project_player_row(htrain, next_season))
    return pd.DataFrame(rows)
