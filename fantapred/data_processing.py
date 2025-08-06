from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

from .utils.cache import memory

# ---------------------------------------------------------------------
# 1) IMPUTAZIONE GERARCHICA – già presente
# ---------------------------------------------------------------------
@memory.cache
def hierarchical_impute(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hierarchical median → role median → full-data IterativeImputer.
    (Cachato per evitare ricomputazioni costose).
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = (
            df.groupby("player_id")[col].transform(lambda s: s.fillna(s.median()))
              .fillna(
                  df.groupby(["team_name_short", "role"])[col].transform(
                      lambda s: s.median()
                  )
              )
              .fillna(df.groupby("role")[col].transform(lambda s: s.median()))
        )
    df[num_cols] = IterativeImputer(max_iter=10, random_state=0).fit_transform(
        df[num_cols]
    )
    return df


# ---------------------------------------------------------------------
# 2) AGGREGAZIONE RIGHE DOPPIE (trasferimenti nella stessa stagione)
# ---------------------------------------------------------------------
# Colonne puramente additive (gol, assist, clean-sheet, minuti, ecc.)
ADDITIVE_COLS = {
    "gf",
    "assist",
    "clean_sheet",
    "presenze",
    "starts_eleven",
    "shots",
    "xg",
    "xg_on_target",
    "passes",
    "cross",
    "duels",
    "min_playing_time",
}

# Colonne di voto/media che vanno mediate pesando per i minuti
RATING_COLS = {"mv", "fmv", "fvm"}


def aggregate_midseason_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprimi i duplicati (player_id, season) dovuti a trasferimenti invernali.

    * Somma ADDITIVE_COLS (gol, assist, minuti…).
    * Media ponderata sui minuti per RATING_COLS (mv, fmv, fvm).
    * Mantiene come squadra/campionato/lega la RIGA **più recente** (post-trasferimento).
    """
    # Se nessun duplicato → ritorna subito
    if df.duplicated(["player_id", "season"]).sum() == 0:
        return df

    def _agg(grp: pd.DataFrame) -> pd.Series:
        out = grp.iloc[-1].copy()  # tieni l’ultima riga (squadra finale)
        # Somme
        for col in ADDITIVE_COLS & set(grp.columns):
            out[col] = grp[col].sum(min_count=1)
        # Medie ponderate voto
        for col in RATING_COLS & set(grp.columns):
            w = grp["min_playing_time"].fillna(0).clip(lower=1)
            out[col] = np.average(grp[col], weights=w)
        return out

    aggregated = (
        df.groupby(["player_id", "season"], as_index=False, sort=False)
        .apply(_agg)
        .reset_index(drop=True)
    )
    return aggregated
