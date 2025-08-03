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
#  Costanti per filtro stagioni infortunio
# ------------------------------------------------------------------ #
SEASON_POSSIBLE_MINUTES = 38 * 90        # minuti totali in una stagione completa
MIN_PLAY_RATIO         = 0.3            # soglia minima (30%) per considerare "stagione valida"
MIN_PLAYING_THRESHOLD  = SEASON_POSSIBLE_MINUTES * MIN_PLAY_RATIO


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
#  Projection helpers (modificata per escludere stagioni corte)
# ------------------------------------------------------------------ #
def project_player_row(hist: pd.DataFrame, next_season: str) -> pd.Series:
    # 1) Filtra solo le stagioni con almeno MIN_PLAYING_THRESHOLD minuti
    hist_valid = hist[hist["min_playing_time"] >= MIN_PLAYING_THRESHOLD]
    if hist_valid.empty:
        # se non ce ne sono, usa tutto lo storico
        hist_valid = hist

    last = hist_valid.iloc[-1].copy()
    role = last["role"]
    age_n = last["age"] + 1

    # base della proiezione
    proj = last.copy()
    proj["season"] = next_season
    proj["age"]    = age_n
    proj["age_sq"] = age_n ** 2

    # 2) Flag infortunio se la stagione precedente è sotto soglia
    proj["injured_last_season"] = int(last["min_playing_time"] < MIN_PLAYING_THRESHOLD)

    # 3) Calcolo stats principali con exp‐weighted mean, outlier cap, growth/decline
    main_stats = ["fmv", "mv", "gf", "assist", "clean_sheet"]
    for stat in main_stats:
        series = hist_valid[stat].dropna().tolist()

        # a) Exp‐weighted mean ultimi 3 anni
        val = _exp_weighted_mean(series[-3:], RECENCY_WEIGHTS)

        # b) Clipping outlier >3σ sugli ultimi 3 anni
        if len(series) >= 3:
            mu, sigma = np.mean(series[-3:]), np.std(series[-3:])
            if sigma > 0 and abs(series[-1] - mu) > 3 * sigma:
                val = mu + np.sign(series[-1] - mu) * sigma

        # c) Growth/decline cap
        val = _apply_growth_decline_cap(val, series[-1] if series else None, stat)
        proj[stat] = val

    # 4) Decay per età (unchanged)
    dcfg = AGE_DECAY.get(role)
    if dcfg and age_n > dcfg["age_thr"]:
        factor = max(0.0, 1 - dcfg["decay"] * (age_n - dcfg["age_thr"]))
        for stat in ["fmv", "mv", "gf", "assist"]:
            proj[stat] *= factor

    # 5) Minuti futuri (exp‐weighted mean)
    if "min_playing_time" in hist_valid.columns:
        proj["min_playing_time"] = _exp_weighted_mean(
            hist_valid["min_playing_time"].tail(3).tolist(),
            RECENCY_WEIGHTS
        )

    # 6) Moved team & delta attacco
    if len(hist_valid) >= 2:
        last_team = hist_valid.iloc[-1]["team_name_short"]
        prev_team = hist_valid.iloc[-2]["team_name_short"]
        proj["moved_team"] = int(last_team != prev_team)
        proj["team_attack_delta"] = (
            hist_valid.iloc[-1]["team_season_gf_mean"]
            - hist_valid.iloc[-2]["team_season_gf_mean"]
        )
    else:
        proj["moved_team"] = 0
        proj["team_attack_delta"] = 0.0

    return proj


# feature_engineering.py
def build_future_dataframe(df: pd.DataFrame, train_until: str, next_season: str) -> pd.DataFrame:
    # mappa “roster” della stagione futura, se presente nel dataset
    roster_cols = ["player_id", "team_name_short", "tournament_name", "role"]
    roster = (df[df["season"] == next_season][roster_cols]
              .drop_duplicates("player_id")
              .set_index("player_id"))

    rows = []
    # usa SOLO storico fino a train_until per proiettare le statistiche
    for pid, hist in df[df["season"] <= train_until].sort_values(["player_id","season"]).groupby("player_id"):
        if hist.empty:
            continue
        proj = project_player_row(hist, next_season)

        # se abbiamo già la riga s_25_26 nel dataset, prendi squadra/lega/ruolo aggiornati
        if pid in roster.index:
            for col in ["team_name_short", "tournament_name", "role"]:
                if col in roster.columns:
                    proj[col] = roster.loc[pid, col]

        rows.append(proj)

    fut_df = pd.DataFrame(rows)

    # feature “cambio squadra” & delta forza attacco basate sulla nuova squadra
    last_team_prev = (df[df["season"] == train_until]
                      .groupby("player_id")["team_name_short"].last())
    team_gf_prev   = (df[df["season"] == train_until]
                      .groupby("team_name_short")["team_season_gf_mean"].mean())

    fut_df["last_team_prev"]   = fut_df["player_id"].map(last_team_prev.to_dict())
    fut_df["moved_team"]       = (fut_df["team_name_short"] != fut_df["last_team_prev"]).astype(int)
    gf_map = team_gf_prev.to_dict()
    fut_df["team_attack_delta"] = (
        fut_df["team_name_short"].map(gf_map).fillna(0.0)
        - fut_df["last_team_prev"].map(gf_map).fillna(0.0)
    )
    fut_df = fut_df.drop(columns=["last_team_prev"])

    return fut_df

