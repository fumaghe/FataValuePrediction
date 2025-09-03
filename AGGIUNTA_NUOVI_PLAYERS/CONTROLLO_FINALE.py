# allinea_nomi_e_report.py
# -*- coding: utf-8 -*-
#
# Cosa fa:
# 1) Carica:
#    - giocatori_stagioni_updated.csv
#    - auction_prices_updated.csv
#    - Quotazioni_Fantacalcio_Stagione_2025_26.xlsx (sheet "Tutti", esclusi i "Ceduti")
# 2) Aggiorna i "name" in giocatori_stagioni per TUTTE le stagioni,
#    usando il formato del foglio Excel (accenti e puntini esatti).
#    Logica mapping (in ordine):
#       (cognome, iniziale, squadra, ruolo) -> NomeExcel
#       (cognome,        squadra, ruolo)    -> NomeExcel
#       (cognome, iniziale) univoco         -> NomeExcel
# 3) Genera un report con:
#       - Mancanti tra i file per slug (riferimento s_25_26)
#       - Squadre diverse (riferimento s_25_26)
#       - Ruoli diversi (riferimento s_25_26)
#       - Presenti in giocatori_stagioni ma non in Excel (per name, ref s_25_26)
#       - Presenti in Excel ma non in giocatori_stagioni (per name, ref s_25_26)
#
# Output:
#   - giocatori_stagioni_names_fixed.csv
#   - report_confronto_giocatori.xlsx

import pandas as pd
import numpy as np
import unicodedata
import re
from pathlib import Path

# =========================
# CONFIG
# =========================
BASE_DIR = Path(".")
FILE_GIOC  = BASE_DIR / "giocatori_stagioni_updated.csv"
FILE_AUCT  = BASE_DIR / "auction_prices_updated.csv"
FILE_EXCEL = BASE_DIR / "Quotazioni_Fantacalcio_Stagione_2025_26.xlsx"
EXCEL_SHEET_TUTTI  = "Tutti"
EXCEL_SHEET_CEDUTI = "Ceduti"

OUT_GIOC   = BASE_DIR / "giocatori_stagioni_names_fixed.csv"
OUT_REPORT = BASE_DIR / "report_confronto_giocatori.xlsx"

# Stagione di riferimento per i confronti nel report
SEASON_REF = "s_25_26"

# =========================
# UTILS
# =========================
def find_col(df: pd.DataFrame, candidates):
    lowmap = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lowmap:
            return lowmap[c.lower()]
    return None

def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFD", s) if unicodedata.category(ch) != "Mn")

def normalize_text(s: str) -> str:
    """Uppercase, rimuove accenti, parentesi, simboli; lascia solo [A-Z0-9 ] con spazi singoli."""
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"\([^)]*\)", " ", s)  # rimuovi contenuto tra parentesi
    s = s.translate(str.maketrans({
        ".": " ", "'": " ", "’": " ", "´": " ", "`": " ", ",": " ",
        "-": " ", "/": " "
    }))
    s = strip_accents(s).upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    return " ".join(s.split())

def extract_surname_and_initial(norm_name: str):
    """
    Esempi:
      'MARTINEZ L'  -> ('MARTINEZ', 'L')
      'MARTINEZ JO' -> ('MARTINEZ', 'J')  (prima lettera)
      'DODO'        -> ('DODO', '')
    """
    if not norm_name:
        return "", ""
    parts = norm_name.split()
    if len(parts) == 1:
        return parts[0], ""
    surname = parts[0]
    last = parts[-1]
    if len(last) <= 2 and len(last) >= 1:
        return surname, last[0]
    return surname, ""

def load_excel_sheet(path: Path, sheet: str, skip_first_dirty_row=True):
    if skip_first_dirty_row:
        try:
            return pd.read_excel(path, sheet_name=sheet, header=0, skiprows=[0])
        except Exception:
            pass
    return pd.read_excel(path, sheet_name=sheet, header=0)

# =========================
# LOAD
# =========================
print("🔹 Carico i file...")
gioc = pd.read_csv(FILE_GIOC)
auct = pd.read_csv(FILE_AUCT)
tutti = load_excel_sheet(FILE_EXCEL, EXCEL_SHEET_TUTTI, skip_first_dirty_row=True)

try:
    ceduti = load_excel_sheet(FILE_EXCEL, EXCEL_SHEET_CEDUTI, skip_first_dirty_row=True)
except Exception:
    ceduti = pd.DataFrame(columns=["Nome"])

# =========================
# COLONNE CHIAVE
# =========================
g_name   = find_col(gioc, ["name", "nome", "player", "giocatore"]) or "name"
g_slug   = find_col(gioc, ["slug"]) or "slug"
g_team   = find_col(gioc, ["team_name_short", "team", "squadra", "club", "team_name"]) or "team_name_short"
g_role   = find_col(gioc, ["role", "ruolo", "r"]) or "role"
g_season = find_col(gioc, ["season", "stagione"]) or "season"

a_slug = find_col(auct, ["slug"]) or "slug"
a_team = find_col(auct, ["team_name_short", "team", "squadra", "club"]) or "team_name_short"
a_role = find_col(auct, ["role", "ruolo", "r"]) or "role"

t_nome = find_col(tutti, ["Nome", "name", "NOME"]) or "Nome"
t_role = find_col(tutti, ["R", "role", "ruolo"]) or "R"
t_team = find_col(tutti, ["Squadra", "team", "club"]) or "Squadra"

c_nome = find_col(ceduti, ["Nome", "name", "NOME"]) or "Nome"

# =========================
# ESCLUDI CEDUTI DA "Tutti"
# =========================
print("🔹 Escludo i 'Ceduti' dal foglio 'Tutti'...")
tutti = tutti.copy()
ceduti = ceduti.copy()

tutti["_nome_norm"]  = tutti[t_nome].astype(str).map(normalize_text)
ceduti["_nome_norm"] = ceduti[c_nome].astype(str).map(normalize_text)

ceduti_set = set(ceduti["_nome_norm"].dropna().unique())
prima_len = len(tutti)
tutti = tutti[~tutti["_nome_norm"].isin(ceduti_set)].drop(columns=["_nome_norm"])
print(f"✅ Rimossi {prima_len - len(tutti)} record (Ceduti).")

# =========================
# COSTRUISCI INDICI DAI NOMI EXCEL
# =========================
print("🔹 Costruisco indici nomi da Excel...")
excel_index_full = {}     # (surname, initial, team, role) -> NomeExcel
excel_index_fb_tr = {}    # (surname, team, role)          -> NomeExcel
multi_map_si = {}         # (surname, initial) -> set(NomeExcel)

for _, row in tutti.iterrows():
    nome_disp = str(row.get(t_nome, "")).strip()
    ruolo     = str(row.get(t_role, "")).strip()
    squadra   = str(row.get(t_team, "")).strip()

    nome_norm = normalize_text(nome_disp)
    surname, initial = extract_surname_and_initial(nome_norm)

    key_full = (surname, initial, normalize_text(squadra), normalize_text(ruolo))
    key_tr   = (surname,              normalize_text(squadra), normalize_text(ruolo))
    key_si   = (surname, initial)

    if surname:
        excel_index_full.setdefault(key_full, nome_disp)
        excel_index_fb_tr.setdefault(key_tr,   nome_disp)
        multi_map_si.setdefault(key_si, set()).add(nome_disp)

# mappa univoca per (surname, initial)
excel_index_si_unique = {k: list(v)[0] for k, v in multi_map_si.items() if len(v) == 1}

print(f"✅ Chiavi: full={len(excel_index_full)}, team/role={len(excel_index_fb_tr)}, si_unique={len(excel_index_si_unique)}")

# =========================
# FUNZIONE DI MAPPING NOME
# =========================
def map_name(curr_name: str, squadra: str, ruolo: str) -> str:
    norm = normalize_text(curr_name)
    surname, initial = extract_surname_and_initial(norm)
    if not surname:
        return curr_name

    k_full = (surname, initial, normalize_text(squadra), normalize_text(ruolo))
    if k_full in excel_index_full:
        return excel_index_full[k_full]

    k_tr = (surname, normalize_text(squadra), normalize_text(ruolo))
    if k_tr in excel_index_fb_tr:
        return excel_index_fb_tr[k_tr]

    if initial:  # fallback su (surname, initial) se univoco
        k_si = (surname, initial)
        if k_si in excel_index_si_unique:
            return excel_index_si_unique[k_si]

    return curr_name

# =========================
# AGGIORNA "name" SU TUTTE LE STAGIONI
# =========================
print("🔹 Aggiorno i 'name' su TUTTE le stagioni (non solo s_25_26)...")
gioc_all = gioc.copy()
changes = []

def update_row(row):
    curr_name = row[g_name]
    squadra   = row[g_team]
    ruolo     = row[g_role]
    new_name  = map_name(curr_name, squadra, ruolo)
    if isinstance(new_name, str) and new_name != curr_name:
        changes.append((row.get("player_id", np.nan), row.get(g_season, ""), curr_name, new_name, squadra, ruolo))
    return new_name

gioc_all[g_name] = gioc_all.apply(update_row, axis=1)
gioc_all.to_csv(OUT_GIOC, index=False, encoding="utf-8")
print(f"✅ Salvato: {OUT_GIOC}")

changes_df = pd.DataFrame(
    changes, columns=["player_id", "season", "old_name", "new_name", "team", "role"]
).sort_values(by=["season", "old_name", "new_name"])

# =========================
# REPORT (riferimento s_25_26)
# =========================
print("🔹 Creo report (riferimento season", SEASON_REF, ")...")
gioc_ref = gioc_all[gioc_all[g_season] == SEASON_REF].copy()

# Mancanti per slug
gioc_slugs = set(gioc_ref[g_slug].dropna().unique())
auct_slugs = set(auct[a_slug].dropna().unique())

missing_in_auction_by_slug = gioc_ref[~gioc_ref[g_slug].isin(auct_slugs)][[g_slug, g_name, g_team, g_role]].drop_duplicates()
missing_in_gioc_by_slug = auct[~auct[a_slug].isin(gioc_slugs)][[a_slug, a_team, a_role]].drop_duplicates().rename(
    columns={a_slug:"slug", a_team:"team_name_short", a_role:"role"}
)

# Mismatch team/role su slug
merged_by_slug = gioc_ref.merge(auct, left_on=g_slug, right_on=a_slug, how="inner", suffixes=("_gioc","_auct"))

team_mismatch = merged_by_slug[merged_by_slug[f"{g_team}_gioc"] != merged_by_slug[f"{a_team}_auct"]][
    [g_slug, g_name, f"{g_team}_gioc", f"{a_team}_auct", f"{g_role}_gioc", f"{a_role}_auct"]
].rename(columns={
    g_slug:"slug", g_name:"name",
    f"{g_team}_gioc":"team_gioc", f"{a_team}_auct":"team_auct",
    f"{g_role}_gioc":"role_gioc", f"{a_role}_auct":"role_auct"
})

role_mismatch = merged_by_slug[merged_by_slug[f"{g_role}_gioc"] != merged_by_slug[f"{a_role}_auct"]][
    [g_slug, g_name, f"{g_role}_gioc", f"{a_role}_auct", f"{g_team}_gioc", f"{a_team}_auct"]
].rename(columns={
    g_slug:"slug", g_name:"name",
    f"{g_role}_gioc":"role_gioc", f"{a_role}_auct":"role_auct",
    f"{g_team}_gioc":"team_gioc", f"{a_team}_auct":"team_auct"
})

# Presenza per name vs Excel (post-fix) sul riferimento
gioc_ref_names = gioc_ref[[g_name, g_team, g_role]].copy()
gioc_ref_names["name_norm"] = gioc_ref_names[g_name].apply(normalize_text)

tutti_names = tutti[[t_nome, t_team, t_role]].copy()
tutti_names["name_norm"] = tutti_names[t_nome].apply(normalize_text)

excel_name_set = set(tutti_names["name_norm"].unique())
gioc_name_set  = set(gioc_ref_names["name_norm"].unique())

missing_in_excel_by_name = gioc_ref_names[~gioc_ref_names["name_norm"].isin(excel_name_set)][[g_name, g_team, g_role]].drop_duplicates()
extra_in_excel_by_name    = tutti_names[~tutti_names["name_norm"].isin(gioc_name_set)][[t_nome, t_team, t_role]].drop_duplicates().rename(
    columns={t_nome:"Nome", t_team:"Squadra", t_role:"R"}
)

# =========================
# SALVA REPORT
# =========================
with pd.ExcelWriter(OUT_REPORT, engine="xlsxwriter") as writer:
    changes_df.to_excel(writer, sheet_name="01_name_changes (ALL seasons)", index=False)
    missing_in_auction_by_slug.to_excel(writer, sheet_name="02_missing_in_auction_by_slug", index=False)
    missing_in_gioc_by_slug.to_excel(writer, sheet_name="03_missing_in_gioc_by_slug", index=False)
    team_mismatch.to_excel(writer, sheet_name="04_team_mismatch_by_slug", index=False)
    role_mismatch.to_excel(writer, sheet_name="05_role_mismatch_by_slug", index=False)
    missing_in_excel_by_name.to_excel(writer, sheet_name="06_missing_in_excel_by_name", index=False)
    extra_in_excel_by_name.to_excel(writer, sheet_name="07_extra_in_excel_by_name", index=False)

print(f"✅ Report creato: {OUT_REPORT}")

# =========================
# LOG
# =========================
print("\n===== RIEPILOGO =====")
print(f"Name aggiornato su {len(changes_df)} righe (TUTTE le stagioni).")
print(f"[Ref {SEASON_REF}] Missing in auction by slug: {len(missing_in_auction_by_slug)}")
print(f"[Ref {SEASON_REF}] Missing in gioc by slug:    {len(missing_in_gioc_by_slug)}")
print(f"[Ref {SEASON_REF}] Team mismatch:              {len(team_mismatch)}")
print(f"[Ref {SEASON_REF}] Role mismatch:              {len(role_mismatch)}")
print(f"[Ref {SEASON_REF}] Missing in Excel by name:   {len(missing_in_excel_by_name)}")
print(f"[Ref {SEASON_REF}] Extra in Excel by name:     {len(extra_in_excel_by_name)}")
