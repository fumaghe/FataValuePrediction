#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
remaining_columns.py

• Aggiorna players_seasons_stats.csv:
  – override slug+season / slug-only (dati pruned)
  – merge statistiche fantacalcio da:
        • Statistiche_Fantacalcio_Stagione_*.xlsx  (Serie A)
        • Statistiche_Fantacalcio_EuroLeghe_Stagione_*.xlsx  (altre leghe, esclusa Serie A)
  – normalizza i nomi (traslitterati, MAIUSCOLO, con iniziale)
  – gestisce cognomi con caratteri speciali e prefissi
  – doppio pass di merge (con e senza squadra)
"""

from __future__ import annotations
import os, re, glob, unicodedata, numpy as np, pandas as pd

# ---------------------------------------------------------------------------
# FILES & PATTERN
# ---------------------------------------------------------------------------
PLAYERS_FILE  = "players_seasons_stats.csv"
PRUNED_FILE   = "season_stats_pruned_renamed.csv"
OUTPUT_FILE   = "players_seasons_stats_updated.csv"

EXCEL_PATTERNS = [
    "Statistiche_Fantacalcio_Stagione_*.xlsx",            # Serie A
    "Statistiche_Fantacalcio_EuroLeghe_Stagione_*.xlsx",  # altre leghe
]

# mappa Excel → CSV
EXCEL_TO_CSV = {
    "PV":   "presenze",
    "MV":   "mv",
    "FM":   "fmv",    # caso stagioni 24_25
    "MF":   "fmv",    # caso stagioni 21_22
    "GF":   "gf",
    "GS":   "gs",
    "RP":   "rp",
    "RC":   "rc",
    "ASS":  "assist",
    "AMM":  "amm",
    "ESP":  "esp",
    "AU":   "au",
}
# ---------------------------------------------------------------------------
# 1) LOAD BASE DATA & ORIGINAL LOGIC
# ---------------------------------------------------------------------------
df_players = pd.read_csv(PLAYERS_FILE)
df_pruned  = pd.read_csv(PRUNED_FILE)

slug_season_cols = [
    'accurate_crosses','aff_index','au','blocked_shots','clearances',
    'crosses_not_claimed','dispossessed','error_lead_to_goal','error_lead_to_shot',
    'gs','hit_woodwork','injured','possession_won_att_third','punches',
    'shot_from_set_piece','total_chipped_passes','total_opposition_half_passes',
    'total_own_half_passes','mv','fmv'
]
slug_only_cols = [
    'fvm','inf_index','jersey_number','last_active_year','quotazione',
    'r_minus','r_plus','spesa_media_altri','team_id','tit_index'
]
name_col = 'name'
original_names = df_players[name_col].copy()

# 1.1 override slug+season
_pruned_ss = df_pruned.set_index(['slug', 'season'])
for col in slug_season_cols:
    if col in _pruned_ss.columns:
        df_players[col] = (
            df_players.set_index(['slug', 'season']).index.map(_pruned_ss[col])
        )

# 1.2 override slug-only
_pruned_slug = (
    df_pruned.reset_index()
    .drop_duplicates(subset='slug', keep='first')
    .set_index('slug')
)
if name_col in _pruned_slug.columns:
    df_players[name_col] = df_players['slug'].map(_pruned_slug[name_col]).fillna(original_names)
for col in slug_only_cols:
    if col in _pruned_slug.columns:
        df_players[col] = df_players[col].fillna(df_players['slug'].map(_pruned_slug[col]))

# ---------------------------------------------------------------------------
# 2) ENSURE NUMERIC DTYPES FOR FANTACALCIO COLUMNS
# ---------------------------------------------------------------------------
for csv_col in EXCEL_TO_CSV.values():
    if csv_col in df_players.columns:
        df_players[csv_col] = pd.to_numeric(df_players[csv_col], errors='coerce').astype('Float64')

# ---------------------------------------------------------------------------
# 3) HELPER FUNCTIONS
# ---------------------------------------------------------------------------
SPECIAL_MAP = str.maketrans({
    'Ð':'D','ð':'d','Þ':'TH','þ':'th','Đ':'DJ','đ':'dj','Š':'S','š':'s','Ž':'Z','ž':'z',
    'Č':'C','č':'c','Ć':'C','ć':'c','Ł':'L','ł':'l','Æ':'AE','æ':'ae','Ø':'O','ø':'o',
    'Å':'A','å':'a','Ü':'U','ü':'u','Ö':'O','ö':'o','Ä':'A','ä':'a'
})

PREFIXES = {
    "DE","DI","DAL","DA","DEL","DEI","DEGLI","DELLA","D",
    "EL","AL","LA","LE","VAN","VON","MC","MAC","SAINT","SAN"
}

def transliterate(text: str) -> str:
    return text.translate(SPECIAL_MAP)

def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_token(text: str) -> str:
    return transliterate(strip_accents(text.upper().replace('.', '')))

def initial_key(full_name: str) -> str:
    toks = clean_token(full_name).split()
    if len(toks) > 1 and len(toks[-1]) == 1:
        return toks[-1]
    return ''

def surname_csv(full_name: str) -> str:
    toks = clean_token(full_name).split()
    if len(toks) == 1:
        return toks[0]
    last = toks[-1]
    if len(last) == 1 and len(toks) > 1:
        last = toks[-2]
    return last

def load_stat_file(path: str, *, skiprows: int = 1) -> pd.DataFrame:
    """
    Legge un file “Statistiche_Fantacalcio_*” restituendo un DataFrame normalizzato.
    – Standardizza tutti gli header in MAIUSCOLO.
    – Unifica MF e FM in un’unica colonna FM.
    – Rinomina le colonne chiave a Title-case: Nome, Squadra, Nazione.
    – Filtra eventuale colonna «Nazione»: esclude le righe dove vale 'Serie A'.
    """
    # 1) Carica e porta tutti i nomi colonne in uppercase
    df = pd.read_excel(path, skiprows=skiprows, na_values=['[]', '-', ''])
    df.columns = [c.strip().upper() for c in df.columns]

    # 2) Se esiste MF ma non FM, o entrambi, rinomina o unifica in FM
    if 'MF' in df.columns:
        # se FM non esiste, rinomina MF→FM
        if 'FM' not in df.columns:
            df.rename(columns={'MF': 'FM'}, inplace=True)
        # se entrambi esistono, elimina MF lasciando FM
        else:
            df.drop(columns=['MF'], inplace=True)

    # 3) Rinomina solo le colonne che servono al merge (che il codice si aspetta)
    rename_map: dict[str, str] = {}
    if 'NOME' in df.columns:    rename_map['NOME']    = 'Nome'
    if 'SQUADRA' in df.columns: rename_map['SQUADRA'] = 'Squadra'
    if 'NAZIONE' in df.columns: rename_map['NAZIONE'] = 'Nazione'
    if rename_map:
        df.rename(columns=rename_map, inplace=True)

    # 4) Filtra le righe di Serie A (solo se esiste la colonna Nazione)
    if 'Nazione' in df.columns:
        mask = df['Nazione'].str.upper().fillna('') != 'SERIE A'
        df = df[mask]

    return df

def surname_xl(nome: str) -> str:
    toks = clean_token(nome).split()
    if not toks:
        return ''
    last = toks[-1]
    if len(last) == 1 and len(toks) > 1:
        last = toks[-2]
    if toks[0] in PREFIXES and len(toks) > 1:
        last = toks[-1]
    return last

def season_key(fname: str) -> str:
    m = re.search(r'(\d{4})_(\d{2})', fname)
    if not m:
        raise ValueError(f"Stagione non riconosciuta in '{fname}'")
    a = int(m.group(1)) % 100      # 2024 → 24
    b = int(m.group(2))            # 25
    return f"s_{a:02d}_{b:02d}"    # s_24_25

def clean_full_name(n: str) -> str:
    return ' '.join(clean_token(n).split())

# ---------------------------------------------------------------------------
# UPDATE STAT FUNCTIONS (PASS 1 & PASS 2)
# ---------------------------------------------------------------------------
def update_stats_pass1(df_players: pd.DataFrame, df_xl: pd.DataFrame) -> pd.DataFrame:
    for xl_col, csv_col in EXCEL_TO_CSV.items():
        if xl_col not in df_xl.columns:
            continue
        df_xl[xl_col] = pd.to_numeric(df_xl[xl_col], errors="coerce")
        if csv_col not in df_players.columns:
            df_players[csv_col] = pd.Series(dtype="Float64")
        ser_uni  = df_xl[xl_col].groupby(level=[0,1,2]).first()
        new_vals = df_players.index.map(ser_uni)
        mask = pd.notna(new_vals)
        df_players[csv_col] = np.where(mask, new_vals, df_players[csv_col])
    return df_players

def update_stats_pass2(
    df_players: pd.DataFrame,
    df_xl: pd.DataFrame
) -> pd.DataFrame:
    """
    Secondo passaggio senza la chiave team: serve a propagare le *medie voto*
    (mv, fmv) SOLO dove mancano, senza sovrascrivere valori già presenti.
    """
    CROSS_TEAM = {"mv", "fmv"}

    for xl_col, csv_col in EXCEL_TO_CSV.items():
        if csv_col not in CROSS_TEAM or xl_col not in df_xl.columns:
            continue

        df_xl[xl_col] = pd.to_numeric(df_xl[xl_col], errors="coerce")
        if csv_col not in df_players.columns:
            df_players[csv_col] = pd.Series(dtype="Float64")

        # raggruppa per (season, surname_key) e prendi il primo
        ser_uni  = df_xl[xl_col].groupby(level=[0, 1]).first()
        new_vals = df_players.index.map(ser_uni)

        # modifica: riempi SOLO dove mv/fmv è NaN e new_vals non lo è
        mask = df_players[csv_col].isna() & pd.notna(new_vals)
        df_players[csv_col] = np.where(mask, new_vals, df_players[csv_col])

    return df_players

# ---------------------------------------------------------------------------
# MERGE HELPERS (FIRST & SECOND PASS)
# ---------------------------------------------------------------------------
def merge_first_pass(df_players: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[str,str],str]]:
    """
    Aggiorna df_players su (season, team_key, surname_key) e costruisce fullname_map.
    Per il file Serie A 2024_25: legge la colonna 'R' come new_role, la propaga a
    TUTTE le righe con lo stesso slug, quindi rimuove la colonna temporanea.
    """
    # imposta l'indice per il merge numerico
    df_players.set_index(['season','team_key','surname_key'], inplace=True)
    fullname_map: dict[tuple[str,str],str] = {}

    for pattern in EXCEL_PATTERNS:
        for path in glob.glob(pattern):
            skey  = season_key(os.path.basename(path))
            df_xl = load_stat_file(path)

            if not {'Nome','Squadra'}.issubset(df_xl.columns):
                raise ValueError(f"Colonne mancanti in {path}")

            # costruisci le chiavi
            df_xl['surname_key'] = df_xl['Nome'].apply(surname_xl)
            df_xl['initial_key'] = df_xl['Nome'].apply(initial_key)
            df_xl['team_key']    = df_xl['Squadra'].str.upper()
            df_xl['season']      = skey
            df_xl.set_index(['season','team_key','surname_key'], inplace=True)

            # SEZIONE RUOLO per Serie A 24_25
            if skey == "s_24_25" and 'R' in df_xl.columns:
                # drop eventuale leftover di new_role
                if 'new_role' in df_players.columns:
                    df_players.drop(columns=['new_role'], inplace=True)

                # unisci la serie R come colonna new_role
                ser_role = df_xl['R'].rename('new_role')
                df_players = df_players.join(ser_role, how='left')

                # costruisci mapping slug→new_role
                mapping = (
                    df_players.reset_index()[['slug','new_role']]
                              .dropna(subset=['new_role'])
                              .drop_duplicates('slug')
                              .set_index('slug')['new_role']
                              .to_dict()
                )

                # applica il ruolo nuovo a tutte le righe di quel slug
                df_players = df_players.reset_index()
                df_players['role'] = df_players.apply(
                    lambda r: mapping.get(r['slug'], r['role']),
                    axis=1
                )
                # ripristina l'indice per il merge numerico
                df_players.set_index(['season','team_key','surname_key'], inplace=True)

            # fullname_map (stesso di prima)
            fn_ser   = df_xl['Nome'].apply(clean_full_name).groupby(level=[0,1,2]).first()
            init_ser = df_xl['initial_key'].groupby(level=[0,1,2]).first()
            for idx, full_name in fn_ser.items():
                sk = idx[2]; ik = init_ser[idx]
                fullname_map.setdefault((sk, ik), full_name)
                fullname_map.setdefault((sk, ''),  full_name)

            # merge numerico (pass1)
            df_players = update_stats_pass1(df_players, df_xl)

    return df_players, fullname_map



def merge_second_pass(df_players: pd.DataFrame) -> pd.DataFrame:
    df_players.reset_index(inplace=True)
    df_players.set_index(['season','surname_key'], inplace=True)

    for pattern in EXCEL_PATTERNS:
        for path in glob.glob(pattern):
            skey  = season_key(os.path.basename(path))
            df_xl = load_stat_file(path)

            if 'Nome' not in df_xl.columns:
                raise ValueError(f"Colonna 'Nome' mancante in {path}")

            df_xl['surname_key'] = df_xl['Nome'].apply(surname_xl)
            df_xl['season']      = skey
            df_xl.set_index(['season','surname_key'], inplace=True)

            df_players = update_stats_pass2(df_players, df_xl)

    return df_players

# ---------------------------------------------------------------------------
# 4) BUILD KEYS & PREPARE NAME MAP
# ---------------------------------------------------------------------------
df_players['surname_key'] = df_players[name_col].apply(surname_csv)
df_players['initial_key'] = df_players[name_col].apply(initial_key)
df_players['team_key']    = df_players['team_name_short'].str.upper()

# ---------------------------------------------------------------------------
# 5-6) MERGE EXCEL DATA
# ---------------------------------------------------------------------------
df_players, fullname_map = merge_first_pass(df_players)
df_players = merge_second_pass(df_players)

# ---------------------------------------------------------------------------
# 7) REPLACE NAME USING FULLNAME MAP
# ---------------------------------------------------------------------------
df_players.reset_index(inplace=True)

df_players[name_col] = df_players.apply(
    lambda r: fullname_map.get(
        (r['surname_key'], r['initial_key']),
        fullname_map.get((r['surname_key'], ''), r[name_col])
    ),
    axis=1
)

# ---------------------------------------------------------------------------
# 8) CLEANUP & SAVE
# ---------------------------------------------------------------------------
df_players.drop(columns=['surname_key','initial_key','team_key'], errors='ignore', inplace=True)
df_players = df_players.drop_duplicates(subset=['name','team_name_short','season'])

definition_order = [
    'player_id','name','team_name_short','season','tournament_name','tournament_country','role',
    'presenze','mv','fmv','gf','gs','rp','rc','assist','amm','esp','au','r_plus','r_minus',
    'starts_eleven','min_playing_time','injured','total_duels_won_percentage','total_duels_won',
    'possession_lost','total_shots','shots_on_target','goal_conversion_percentage',
    'free_kick_shots','penalty_won','attempt_penalty_miss','attempt_penalty_target','shots_off_target',
    'total_attempt_assist','pass_to_assist','big_chances_created','big_chances_create',
    'expected_goals','scoring_frequency','hit_woodwork',
    'successful_dribbles','successful_dribbles_percentage','touches',
    'key_passes','accurate_passes_percentage','total_passes','accurate_passes',
    'accurate_crosses','total_cross','accurate_final_third_passes',
    'accurate_opposition_half_passes','accurate_chipped_passes',
    'accurate_long_balls','accurate_long_balls_percentage','total_long_balls',
    'total_chipped_passes','total_opposition_half_passes','total_own_half_passes',
    'tackles','tackles_won','tackles_won_percentage','interceptions','clearances',
    'blocked_shots','crosses_not_claimed','punches','duel_lost','aerial_duels_won',
    'aerial_duels_won_percentage','aerial_lost','possession_won_att_third',
    'error_lead_to_goal','error_lead_to_shot','fouls','was_fouled',
    'penalty_conceded','penalty_conversion','offsides',
    'saves','penalty_faced','clean_sheet','successful_runs_out',
    'shot_from_set_piece',
    'quotazione','fvm','inf_index','tit_index','aff_index','spesa_media_altri',
    'last_active_year','date_of_birth','height','jersey_number',
    'preferred_foot','country','team_id',
    'slug','link','image'
]
df_players = df_players[[c for c in definition_order if c in df_players.columns]]

# ---------------------------------------------------------------------------
# 9) Rimuovi mv/fmv = 0.0 (non tutta la riga, solo quei valori)
# ---------------------------------------------------------------------------
for col in ("mv", "fmv"):
    if col in df_players.columns:
        df_players[col] = df_players[col].mask(df_players[col] == 0, pd.NA)

df_players.to_csv(OUTPUT_FILE, index=False)
print(f"Aggiornamento completato ➜ '{OUTPUT_FILE}'")