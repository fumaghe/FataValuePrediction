"""
Post-processing finale delle predizioni.

• Boost assist (× 1.33)
• Calcola fmv_pred = mv_pred + bonus SOLO per D/C/A
• Soft-clip outlier e fix default rating
"""

from __future__ import annotations
import numpy as np
import pandas as pd

try:
    from scipy.special import expit           # sigmoide logistica
except ModuleNotFoundError:                   # fallback se SciPy manca
    expit = lambda x: 1 / (1 + np.exp(-x))    # type: ignore

from ..settings import MIN_MATCHES_FOR_RATING, DEFAULT_RATING

# ------------------- parametri regolabili -------------------------- #
ASSIST_MULT = 1.03
BONUS_K     = 1.05
# ------------------------------------------------------------------- #


def _goal_curve(g: pd.Series | np.ndarray) -> pd.Series:
    """ Bonus non lineare per i gol (3*gol fino a ~10 reti). """
    return 2.2 * expit(0.25 * (g - 6)) * g


def postprocess(
    df: pd.DataFrame,
    bonus_mode: str = "curve",        # “curve” | “linear” | “off”
    round_stats: bool = True,
) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------------
    # 1) Boost assist
    # ------------------------------------------------------------------
    if "assist_pred" in df.columns:
        df["assist_pred"] *= ASSIST_MULT

    # ------------------------------------------------------------------
    # 2) Bonus fmv SOLO per ruoli di movimento
    # ------------------------------------------------------------------
    mov_mask = df["role"] != "P"                  # D / C / A

    games = df["min_playing_time"].clip(lower=1) / 90.0

    if bonus_mode == "off":
        bonus_pg = 0.0
    elif bonus_mode == "linear":
        bonus_pg = (df["gf_pred"] * 3.0 + df["assist_pred"]) / games
    else:                                          # “curve”
        bonus_pg = (_goal_curve(df["gf_pred"]) + df["assist_pred"]) / games

    # aggiorna solo dove mov_mask è True
    df.loc[mov_mask, "fmv_pred"] = (
        df.loc[mov_mask, "mv_pred"] + BONUS_K * bonus_pg[mov_mask]
    )
    # Portieri: fmv_pred resta quella uscita dal modello

    # ------------------------------------------------------------------
    # 3) Pulizia & soft-clip outlier
    # ------------------------------------------------------------------
    num_stats  = ["gf_pred", "assist_pred", "clean_sheet_pred"]
    rate_stats = ["mv_pred", "fmv_pred"]

    df[num_stats]  = df[num_stats].clip(lower=0).fillna(0)
    df[rate_stats] = df[rate_stats].fillna(DEFAULT_RATING)

    hi = df["fmv_pred"].quantile(0.995)
    lo = df["fmv_pred"].quantile(0.005)
    df["fmv_pred"] = df["fmv_pred"].clip(lo, hi)

    # azzera clean-sheet per ruoli non-portiere
    df.loc[df.role != "P", "clean_sheet_pred"] = 0

    # rating default se poche presenze
    few = df["min_playing_time"] < MIN_MATCHES_FOR_RATING * 90
    df.loc[few, rate_stats] = DEFAULT_RATING

    if round_stats:
        df[num_stats]  = df[num_stats].round(2)
        df[rate_stats] = df[rate_stats].round(2)

    return df
