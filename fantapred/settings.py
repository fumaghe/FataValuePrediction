"""
Global configuration constants for the fantapred package.
"""
from pathlib import Path

# ------------------------------------------------------------------ #
#  Percorsi
# ------------------------------------------------------------------ #
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# ------------------------------------------------------------------ #
#  Feature categoriche
# ------------------------------------------------------------------ #
CAT_COLS = ["team_name_short"]

# ------------------------------------------------------------------ #
#  Target
# ------------------------------------------------------------------ #
TARGETS_ALL  = ["fmv", "mv", "gf", "assist", "clean_sheet"]
TARGETS_CORE = ["fmv", "mv"]

# Colonne che causerebbero leakage se usate come feature
LEAK_COLS = ["fmv", "mv"]

# ------------------------------------------------------------------ #
#  Pesi e trasformazioni base
# ------------------------------------------------------------------ #
RECENCY_WEIGHTS = [0.6, 0.3, 0.1]

AGE_DECAY = {
    "P": {"age_thr": 34, "decay": 0.010},
    "D": {"age_thr": 31, "decay": 0.015},
    "C": {"age_thr": 30, "decay": 0.020},
    "A": {"age_thr": 29, "decay": 0.025},
}

GROWTH_CAP  = {"gf": 0.50, "assist": 0.40, "fmv": 0.20, "mv": 0.20, "clean_sheet": 0.15}
DECLINE_CAP = {"gf": 0.35, "assist": 0.35, "fmv": 0.40, "mv": 0.40, "clean_sheet": 0.40}

MIN_MATCHES_FOR_RATING = 4
DEFAULT_RATING         = 6.0

# ------------------------------------------------------------------ #
#  COEFFICIENTE DI DIFFICOLTÀ DELLA LEGA
#  (Premier > Serie A > La Liga ≈ Bundesliga > tutte le altre)
# ------------------------------------------------------------------ #
LEAGUE_COEF = {
    "1. HNL": 0.65,
    "2. Bundesliga": 0.85,
    "3. Liga": 0.60,
    "A Group": 0.55,
    "A-League": 0.60,
    "Bundesliga": 1.15,
    "Championship": 0.90,
    "Chilean Primera División": 0.60,
    "Conf Premier": 0.30,
    "Czech First League": 0.65,
    "Danish Superliga": 0.70,
    "Eerste Divisie": 0.70,
    "Ekstraklasa": 0.65,
    "Eredivisie": 0.95,
    "First Division A": 0.85,
    "First League": 0.50,
    "HNL": 0.65,
    "Jr.PL2 — Div. 1": 0.20,
    "Jr.PL2 — Div. 2": 0.15,
    "Jr.U17 Bundesliga": 0.10,
    "Jr.U19 Bundesliga": 0.12,
    "La Liga": 1.15,
    "La Liga 2": 0.90,
    "League One": 0.60,
    "League Two": 0.50,
    "Liga Argentina": 0.90,
    "Liga I": 0.65,
    "Liga MX": 0.95,
    "Ligue 1": 1.05,
    "Ligue 2": 0.85,
    "NB I": 0.60,
    "Premier League": 1.20,
    "Premiership": 0.75,
    "Primeira Liga": 1.00,
    "Primera Div": 0.60,
    "Pro League": 0.85,
    "Pro League A": 0.85,
    "Second Division": 0.50,
    "Serie A": 1.10,
    "Serie B": 0.85,
    "Super League": 0.75,
    "Super Lg": 0.80,
    "SuperLiga": 0.70,
    "Süper Lig": 0.80,
    "Uruguayan Primera División": 0.55,
}

DEFAULT_LEAGUE_COEF = 0.80


# ------------------------------------------------------------------ #
#  Punteggi allenatore × ruolo (Serie A 2025-26)
#  0 = neutro, 5 = massimo boost / malus
# ------------------------------------------------------------------ #
TEAM_ROLE_SCORE: dict = {
    #  team : { ruolo : (adv, dis) }
    "ATA": {"D": (4, 0), "C": (3, 0), "A": (0, 1)},
    "BOL": {"A": (4, 0), "C": (3, 0), "D": (2, 2)},
    "CRE": {"C": (4, 0), "D": (4, 0), "A": (0, 2)},
    "CAG": {"D": (5, 0), "C": (3, 0), "A": (0, 1)},
    "COM": {"C": (4, 0), "A": (0, 1)},
    "FIO": {"A": (4, 1), "C": (4, 0), "D": (2, 1)},
    "GEN": {"A": (4, 0), "C": (3, 0), "D": (3, 2)},
    "VER": {"C": (4, 0), "D": (3, 0), "A": (0, 1)},
    "INT": {"A": (5, 1), "C": (2, 0), "D": (3, 0)},
    "JUV": {"C": (4, 0), "D": (4, 0), "A": (0, 1)},
    "LAZ": {"A": (5, 0), "C": (4, 0), "D": (2, 1)},
    "LEC": {"A": (4, 0), "C": (3, 0), "D": (2, 1)},
    "MIL": {"C": (3, 0), "D": (4, 0), "A": (0, 1)},
    "NAP": {"C": (4, 0), "D": (4, 0), "A": (0, 2)},
    "PAR": {"C": (4, 0), "D": (3, 0), "A": (0, 1)},
    "PIS": {"A": (4, 0), "C": (3, 0), "D": (2, 1)},
    "ROM": {"A": (5, 1), "C": (4, 0), "D": (3, 2)},
    "SAS": {"A": (4, 0), "C": (3, 0), "D": (2, 1)},
    "TOR": {"A": (5, 2), "C": (3, 0), "D": (2, 1)},
    "UDI": {"A": (5, 2), "C": (4, 0), "D": (2, 1)},
}
TEAM_DEFAULT_ROLE_COEF = 1.00  # club o ruolo non elencato

# ---------------- coeff finale = 1 + SCALE * (adv - dis) ----------- #
SCORE_SCALE = 0.03  #   delta ±5  →  coeff 0.85 … 1.15

# ------------------------------------------------------------------ #
#  LightGBM – GPU opzionale
# ------------------------------------------------------------------ #
# Se hai build GPU sostituisci con:
# GPU_PARAMS = {"device": "gpu", "gpu_platform_id": 0, "gpu_device_id": 0}
GPU_PARAMS: dict = {}
