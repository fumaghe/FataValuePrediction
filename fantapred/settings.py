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
    "P": {"age_thr": 34, "decay": 0.10},
    "D": {"age_thr": 31, "decay": 0.15},
    "C": {"age_thr": 30, "decay": 0.10},
    "A": {"age_thr": 29, "decay": 0.15},
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
    # --- Elite European Leagues (>= 0.85) ---
    "Premier League": 1.00,         # England
    "La Liga": 0.92,                # Spain
    "Bundesliga": 0.85,             # Germany
    "Serie A": 0.96,                # Italy
    "Ligue 1": 0.82,                # France

    # --- Strong International & European (0.70 - 0.84) ---
    "Primeira Liga": 0.55,          # Portugal
    "Liga MX": 0.50,                # Mexico
    "Eredivisie": 0.61,             # Netherlands
    "Liga Argentina": 0.52,         # Argentina
    "Süper Lig": 0.56,              # Turkey

    # --- Secondary European Leagues (0.55 - 0.69) ---
    "2. Bundesliga": 0.41,          # Germany (2nd tier)
    "Championship": 0.57,           # England (2nd tier)
    "Ligue 2": 0.40,                # France (2nd tier)
    "La Liga 2": 0.42,              # Spain (2nd tier)
    "Premiership": 0.42,            # Scotland

    # --- Mid-Level Professional Leagues (0.45 - 0.54) ---
    "Serie B": 0.45,                # Italy (2nd tier)
    "First Division A": 0.48,       # Belgium
    "Pro League": 0.47,             # Saudi Arabia
    "Pro League A": 0.47,           # Belgium
    "A-League": 0.45,               # Australia
    "Danish Superliga": 0.45,       # Denmark
    "Czech First League": 0.43,     # Czech Republic

    # --- Lower Professional Leagues (0.35 - 0.44) ---
    "Chilean Primera División": 0.40,  # Chile
    "3. Liga": 0.40,                  # Germany (3rd tier)
    "Eerste Divisie": 0.38,           # Netherlands (2nd tier)
    "Primera Div": 0.35,              # Mexico (former top tier)
    "League One": 0.35,               # England (3rd tier)

    # --- Emerging & Domestic Leagues (0.25 - 0.34) ---
    "1. HNL": 0.30,                   # Croatia
    "HNL": 0.30,                      # Croatia
    "League Two": 0.30,               # England (4th tier)
    "Super League": 0.30,             # Switzerland
    "Ekstraklasa": 0.30,              # Poland

    # --- Other Top Divisions (0.20 - 0.24) ---
    "NB I": 0.28,                     # Hungary
    "Liga I": 0.25,                   # Romania
    "Uruguayan Primera División": 0.25,  # Uruguay
    "A Group": 0.20,                  # Bulgaria
    "First League": 0.20,             # Slovakia

    # --- Lower Tiers & Youth Leagues (< 0.20) ---
    "Conf Premier": 0.18,             # England (5th tier)
    "Super Lg": 0.15,                 # China (Super League)
    "SuperLiga": 0.15,                # Serbia
    "Second Division": 0.15,          # Generic 2nd tier
    "Jr.PL2 — Div. 1": 0.10,          # Poland (U19)
    "Jr.PL2 — Div. 2": 0.07,          # Poland (U17)
    "Jr.U19 Bundesliga": 0.06,        # Germany (U19)
    "Jr.U17 Bundesliga": 0.05,        # Germany (U17)
}
DEFAULT_LEAGUE_COEF = 0.20

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
