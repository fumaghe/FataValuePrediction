#!/usr/bin/env python3
"""
Trasforma le predizioni in prezzi d’asta consigliati.
Funziona *senza* retrain: usa solo il CSV di output.

python -m fantapred.tools.auction_pricing --pred_csv future_predictions_s_25_26.csv --teams 8
"""

from __future__ import annotations
import argparse, math, sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ #
#  Rendi importabile “fantapred” anche se esegui lo script da tools/
# ------------------------------------------------------------------ #
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from fantapred.settings import TEAM_ROLE_SCORE as TEAM_ROLE_SCORE_ORIG, LEAGUE_COEF

SCORE_SCALE = 1.7

# ===================== PARAMETRI DI TUNING ======================== #
#  Regola SOLO queste costanti; il resto del codice non va toccato. #
# ------------------------------------------------------------------ #

# ---- Slot di rosa e budget medio slot titolare ------------------- #
ROLE_SLOTS = {"P": 3, "D": 8, "C": 8, "A": 6}
AVG_SLOT   = {"P": 16, "D": 30, "C": 25, "A": 60}   # crediti medi per slot TIT

# ---- Limiti e forma della curva prezzo --------------------------- #
CAP_ROLE = {"P": 55, "D": 60, "C": 70, "A": 220}     # hard-cap
FREE_PERC = 0.10                                     # quota 1 credito
P_LOW, P_HIGH = 5, 95                                # percentili normalizzazione
SHARP_R = {"P": 1.5, "D": 1.3, "C": 1.5, "A": 1.5}   # esponenti curva

# ---- Disponibilità (starts / presenze) --------------------------- #
MAX_MATCHES = 38                 # partite di campionato
STARTS_W    = 0.7                # peso starts vs presenze (0-1)
AVAIL_K1    = 0.5                # coeff. minimo (panchinaro cronico)
AVAIL_K2    = 0.7                # ampiezza (→ max = K1 + K2)

# ==================== OVERRIDE SERIE A 2025-26 ==================== #
# Abbozzo di punteggi tattici (advantage, disadvantage) per ruolo.
#  0   = neutro;  0.2 ≈ “molto favorevole”;  -0.2 ≈ “molto sfavorevole”.
TEAM_ROLE_SCORE_25_26 = {
    # Top 4 “powerhouse”: +20/+25/+15/+20
    "Inter":      {"P": (0.20,0), "D": (0.25,0), "C": (0.15,0), "A": (0.20,0)},
    "AC Milan":   {"P": (0.20,0), "D": (0.25,0), "C": (0.15,0), "A": (0.20,0)},
    "Juventus":   {"P": (0.20,0), "D": (0.20,0), "C": (0.15,0), "A": (0.18,0)},
    "Napoli":     {"P": (0.20,0), "D": (0.20,0), "C": (0.18,0), "A": (0.20,0)},

    # Contendenti di alto livello: +15/+20/+15/+18
    "Roma":       {"P": (0.15,0), "D": (0.20,0), "C": (0.15,0), "A": (0.18,0)},
    "Atalanta":   {"P": (0.15,0), "D": (0.18,0), "C": (0.15,0), "A": (0.18,0)},
    "Fiorentina": {"P": (0.15,0), "D": (0.15,0), "C": (0.15,0), "A": (0.18,0)},
    "Lazio":      {"P": (0.15,0), "D": (0.15,0), "C": (0.15,0), "A": (0.15,0)},

    # Squadre di metà classifica: +5/+10/+5/+10
    "Bologna":    {"P": (0.05,0), "D": (0.10,0), "C": (0.05,0), "A": (0.10,0)},
    "Torino":     {"P": (0.05,0), "D": (0.10,0), "C": (0.05,0), "A": (0.10,0)},
    "Udinese":    {"P": (0.05,0), "D": (0.10,0), "C": (0.05,0), "A": (0.05,0)},
    "Genoa":      {"P": (0.05,0), "D": (0.10,0), "C": (0.05,0), "A": (0.10,0)},
    "Sassuolo":   {"P": (0.00,0), "D": (0.05,0), "C": (0.05,0), "A": (0.10,0)},

    # Squadre in cerca di conferme: 0/+5/0/+5
    "Monza":      {"P": (0.00,0), "D": (0.05,0), "C": (0.00,0), "A": (0.05,0)},
    "Hellas Verona": {"P": (0.00,0), "D": (0.05,0), "C": (0.00,0), "A": (0.05,0)},
    "Cremonese":  {"P": (0.00,0), "D": (0.05,0), "C": (0.00,0), "A": (0.05,0)},

    # “Underdogs” e neopromosse: –10/–10/0/–?3
    "Cagliari":   {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (0.00,0)},
    "Lecce":      {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (-0.10,0)},
    "Como":       {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (0.00,0)},
    "Parma":      {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (0.00,0)},
    "Pisa":       {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (0.05,0)},
    "Palermo":    {"P": (-0.10,0), "D": (-0.10,0), "C": (0.00,0), "A": (-0.15,0)},
}



# Se vuoi disabilitare l’override, imposta USE_SA_OVERRIDE = False

USE_SA_OVERRIDE = True
TEAM_ROLE_SCORE = TEAM_ROLE_SCORE_25_26 if USE_SA_OVERRIDE else TEAM_ROLE_SCORE_ORIG

# ======================= FUNZIONI DI SCORE ======================== #
def raw_score(r: pd.Series) -> float:
    """Bonus/malus ‘fantacalcistici’ in valore assoluto."""
    if r.role == "A":
        return 3 * r.gf_pred + 1.5 * r.assist_pred + r.fmv_pred
    if r.role == "C":
        return 3 * r.gf_pred + 2 * r.assist_pred + r.fmv_pred
    if r.role == "D":
        return 1.5 * r.gf_pred + 2 * r.assist_pred + 2.5 * r.fmv_pred + 4 * r.mv_pred
    # Portieri
    return 5 * r.clean_sheet_pred + r.mv_pred

def team_role_coef(team: str, role: str) -> float:
    """Moltiplicatore tattico allenatore×ruolo."""
    adv, dis = TEAM_ROLE_SCORE.get(team, {}).get(role, (0, 0))
    return 1.0 + SCORE_SCALE * (adv - dis)

def availability_coef(starts: float, presenze: float) -> float:
    """
    Coefficiente a scaglioni basato su presenze (e titolarità opzionale).
    """
    # ---- moltiplicatore per le presenze ----
    if presenze <= 8:
        k_pres = 0.60
    elif presenze <= 15:
        k_pres = 0.80
    elif presenze <= 24:
        k_pres = 1.00
    else:
        k_pres = 1.20

    # ---- (opzionale) moltiplicatore per le starts ----
    if starts <= 8:
        k_start = 0.60
    elif starts <= 15:
        k_start = 0.80
    elif starts <= 24:
        k_start = 1.00
    else:
        k_start = 1.10

    # ---- combinazione: prendi la media (o il minimo, se vuoi penalizzare di più) ----
    return (k_pres + k_start) / 2
    # return min(k_pres, k_start)

# ===================== PREZZO PER SINGOLO RUOLO =================== #
def price_for_role(df_in: pd.DataFrame, role: str, teams: int) -> pd.DataFrame:
    df = df_in.copy()
    if df.empty:
        return df

    # 1. Normalizzazione robusta  (percentile 5-95)
    p5, p95 = np.percentile(df.score_adj, [P_LOW, P_HIGH])
    spread  = max(p95 - p5, 1e-6)
    exp     = SHARP_R[role]
    df["score_norm"] = ((df.score_adj - p5).clip(lower=0) / spread) ** exp

    # 2. Pool crediti per il ruolo
    slots_tot   = teams * ROLE_SLOTS[role]
    pool_cred   = slots_tot * AVG_SLOT[role]
    total_score = df.score_norm.sum() or 1e-6
    df["price"] = pool_cred * df.score_norm / total_score
    df.loc[df.price > CAP_ROLE[role], "price"] = CAP_ROLE[role]

    # 3. Ultimi FREE_PERC slot a 1 credito
    df["price"] = df["price"].clip(lower=1)
    df = df.sort_values("price", ascending=False).reset_index(drop=True)
    n_free = min(math.floor(slots_tot * FREE_PERC), len(df) - 1)
    if n_free:
        df.iloc[-n_free:, df.columns.get_loc("price")] = 1

    return df

# ============================ MAIN ================================ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_csv", required=True, help="CSV con *_pred")
    ap.add_argument("--teams", type=int, choices=[8, 10], default=8,
                    help="Numero squadre del Fanta")
    ap.add_argument("--outfile", default="auction_prices.csv",
                    help="CSV di output")
    args = ap.parse_args()

    df = pd.read_csv(args.pred_csv)

    # ---- Punteggio grezzo + coef. contestuali --------------------- #
    df["raw_score"]       = df.apply(raw_score, axis=1)
    df["team_role_coef"]  = df.apply(
        lambda r: team_role_coef(r.team_name_short, r.role), axis=1)
    df["league_coef"]     = (
        df.tournament_name.map(LEAGUE_COEF).fillna(1.0)
        if "tournament_name" in df.columns else 1.0
    )
    df["availability_coef"] = df.apply(
        lambda r: availability_coef(r.starts_pred, r.presenze_pred), axis=1
    )

    df["score_adj"] = (
        df.raw_score *
        df.team_role_coef *
        df.league_coef *
        df.availability_coef
    )

    # ---- Prezzo per ciascun ruolo -------------------------------- #
    price_frames = [
        price_for_role(df[df.role == role], role, teams=args.teams)
        for role in ROLE_SLOTS
    ]
    out = (pd.concat(price_frames)
             .sort_values(["role", "price"], ascending=[True, False])
             .reset_index(drop=True))

    out.price = out.price.round(1)   # arrotondamento finale

    # ---- Mantieni tutte le colonne di input + price --------------- #
    #   (rimuovi colonne temporanee se non ti servono nell’export) raw_score,team_role_coef,league_coef,availability_coef,score_adj,
    cols_to_drop = ["score_norm", "raw_score", "team_role_coef", "league_coef",
                    "availability_coef", "score_adj"]
    out = out.drop(columns=[c for c in cols_to_drop if c in out.columns])

    out.to_csv(args.outfile, index=False)
    print(f"✅  Salvato {Path(args.outfile).resolve()} – {len(out)} righe")

if __name__ == "__main__":
    main()