# -*- coding: utf-8 -*-

import pandas as pd
import re
from pathlib import Path

# ---- CONFIG "one-click" ----
BASE_DIR = Path.cwd()
DATA_DIR = BASE_DIR / "PREZZOxFVM"

EXCEL_PATH = DATA_DIR / "Quotazioni_Fantacalcio_Stagione_2025_26.xlsx"
GS_CSV_PATH = DATA_DIR / "giocatori_stagioni.csv"
AUCTION_CSV_PATH = DATA_DIR / "auction_prices.csv"

SEASON = "s_25_26"

# Fattori per ruolo (puoi modificarli qui)
ROLE_FACTORS = {
    "P": 0.95,  # Portieri
    "D": 0.90,  # Difensori
    "C": 0.64,  # Centrocampisti
    "A": 0.95,  # Attaccanti
}
DEFAULT_FACTOR = 0.60  # usato se ruolo mancante/sconosciuto

OUT_UNMATCHED = DATA_DIR / "unmatched_s_25_26.csv"
OUT_AUCTION_UPDATED = DATA_DIR / "auction_prices_updated.csv"
# ----------------------------

ACCENT_MAP = {
    "à":"A'", "è":"E'", "é":"E'", "ì":"I'", "í":"I'", "ò":"O'", "ó":"O'", "ù":"U'",
    "À":"A'", "È":"E'", "É":"E'", "Ì":"I'", "Í":"I'", "Ò":"O'", "Ó":"O'", "Ù":"U'",
    "ä":"A", "ë":"E", "ï":"I", "ö":"O", "ü":"U", "Ä":"A", "Ë":"E", "Ï":"I", "Ö":"O", "Ü":"U",
    "ç":"C", "Ç":"C", "ñ":"N", "Ñ":"N",
    "’":"'", "ʻ":"'", "ˈ":"'", "ʹ":"'", "′":"'", "᾿":"'", "ʼ":"'", "＇":"'"
}

def normalize_name(s: str) -> str:
    """Normalizza nome per il match:
    - 'Rossi F.' -> 'Rossi F' (rimozione del punto negli iniziali)
    - Accenti stile Fanta: 'Konè' -> 'KONE''
    - Apostrofi tipografici -> ASCII '
    - Uppercase + spazi normalizzati
    """
    if pd.isna(s):
        return ""
    s = str(s).strip()
    # rimuove punto negli iniziali singoli, es. "Rossi F." -> "Rossi F"
    s = re.sub(r"\b([A-Za-zÀ-ÖØ-öø-ÿ])\.", r"\1", s)
    s = "".join(ACCENT_MAP.get(ch, ch) for ch in s)  # accenti/apostrofi
    s = s.upper()
    s = re.sub(r"\s+", " ", s)
    return s

def detect_relevant_sheet(xls: pd.ExcelFile) -> str:
    """Sceglie il foglio più probabile (preferisce nomi con 'quot', 'rosa', 'list', 'tutti')."""
    sheet_names = xls.sheet_names
    candidates = []
    for sn in sheet_names:
        try:
            df_tmp = pd.read_excel(xls, sheet_name=sn, nrows=20)
        except Exception:
            continue
        cols_lower = [str(c).strip().lower() for c in df_tmp.columns]
        if any(c in cols_lower for c in ["nome", "giocatore", "calciatore", "player", "nome_giocatore", "nome giocatore"]):
            candidates.append(sn)
    if candidates:
        for sn in candidates:
            low = sn.lower()
            if any(k in low for k in ["quot", "rosa", "list", "tutti"]):
                return sn
        return candidates[0]
    return sheet_names[0]

def read_excel_with_header_detection(excel_path: Path, sheet_name: str) -> pd.DataFrame:
    """Rileva l'header cercando una riga che contenga 'Calciatore'/'Nome' e 'Ruolo'."""
    df_raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
    header_row_idx = None
    for i in range(min(20, len(df_raw))):
        row_vals_lower = df_raw.iloc[i].astype(str).str.strip().str.lower().tolist()
        if ("ruolo" in row_vals_lower) and ("calciatore" in row_vals_lower or "nome" in row_vals_lower):
            header_row_idx = i
            break
    if header_row_idx is None:
        header_row_idx = 1
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=header_row_idx)
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def pick_name_col(df: pd.DataFrame) -> str:
    for cand in ["Nome", "Calciatore", "Giocatore", "Player", "Nome_giocatore", "Nome giocatore"]:
        if cand in df.columns:
            return cand
    str_cols = [c for c in df.columns if df[c].dtype == object]
    if str_cols:
        picked = None
        for c in str_cols:
            try:
                if df[c].nunique(dropna=True) > 50:
                    picked = c; break
            except Exception:
                continue
        return picked or str_cols[0]
    return df.columns[0]

def pick_fvm_col(df: pd.DataFrame) -> str:
    for cand in ["FVM", "FVM M", "Fvm", "fvm"]:
        if cand in df.columns:
            return cand
    raise ValueError("Colonna 'FVM' non trovata nell'Excel delle quotazioni.")

def pick_role_col(df: pd.DataFrame) -> str:
    # spesso 'R' = ruolo breve (P/D/C/A), 'RM' = ruolo esteso
    for cand in ["R", "RM", "Ruolo", "Role"]:
        if cand in df.columns:
            return cand
    return None

def pick_price_col(df: pd.DataFrame) -> str:
    for cand in ["price", "Price", "prezzo", "auction_price"]:
        if cand in df.columns:
            return cand
    # fallback: prima colonna numerica ragionevole
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                return c
        except Exception:
            continue
    raise ValueError("Colonna 'price' non trovata in auction_prices.csv")

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Excel quotazioni
    xls = pd.ExcelFile(EXCEL_PATH)
    sheet = detect_relevant_sheet(xls)
    excel_df = read_excel_with_header_detection(EXCEL_PATH, sheet)
    name_col = pick_name_col(excel_df)
    fvm_col = pick_fvm_col(excel_df)
    role_col = pick_role_col(excel_df)

    # Prepara nome Excel: rimuove i punti e normalizza
    excel_df["_nome_raw"] = excel_df[name_col].astype(str).str.replace(".", "", regex=False)
    excel_df["_nome_norm"] = excel_df["_nome_raw"].apply(normalize_name)

    # 2) giocatori_stagioni filtrato per season
    gs = pd.read_csv(GS_CSV_PATH)
    gs.columns = [str(c).strip() for c in gs.columns]
    required = {"season", "name", "slug"}
    if not required.issubset(gs.columns):
        raise ValueError(f"Mancano colonne in giocatori_stagioni.csv (richieste: {sorted(required)}) - Trovate: {gs.columns.tolist()}")
    gs_season = gs[gs["season"].astype(str).str.strip() == SEASON].copy()
    gs_season["_name_raw"] = gs_season["name"]
    gs_season["_name_norm"] = gs_season["_name_raw"].apply(normalize_name)

    # 3) Merge per ottenere slug + FVM + RUOLO (se disponibile)
    left_cols = ["_nome_norm", name_col, fvm_col]
    if role_col and role_col in excel_df.columns:
        left_cols.append(role_col)
    merged = pd.merge(
        excel_df[left_cols].rename(columns={name_col: "nome", fvm_col: "FVM", (role_col or "RUOLO"): "RUOLO"}),
        gs_season[["_name_norm", "name", "slug"]],
        left_on="_nome_norm",
        right_on="_name_norm",
        how="left"
    )

    matched = merged[merged["slug"].notna()].copy()
    unmatched = merged[merged["slug"].isna()].copy()

    # 4) Salva SOLO gli unmatched
    unmatched_out = unmatched[["nome", "name", "slug"]].copy()
    unmatched_out["name"] = ""
    unmatched_out["slug"] = ""
    unmatched_out["nome_norm"] = unmatched["_name_norm"]
    unmatched_out["name_norm"] = ""
    unmatched_out.to_csv(OUT_UNMATCHED, index=False)

    # 5) Aggiorna auction_prices.csv con i matched + factor per ruolo
    auction = pd.read_csv(AUCTION_CSV_PATH)
    auction.columns = [str(c).strip() for c in auction.columns]
    price_col = pick_price_col(auction)

    # Merge su slug se possibile, altrimenti su nome normalizzato
    if "slug" in auction.columns:
        to_update = matched[["slug", "FVM", "RUOLO"]].copy()
        auction = auction.merge(to_update, how="left", on="slug")
    else:
        # crea chiave nome normalizzato in auction
        name_col_auction = None
        for cand in ["name", "player_name", "Nome", "Calciatore"]:
            if cand in auction.columns:
                name_col_auction = cand
                break
        if name_col_auction is None:
            raise ValueError("Impossibile allineare auction_prices.csv: mancano sia 'slug' che 'name'.")
        auction["_name_norm"] = auction[name_col_auction].apply(normalize_name)
        to_update = matched.rename(columns={"_name_norm": "_name_norm"})[["_name_norm", "FVM", "RUOLO"]]
        auction = auction.merge(to_update, how="left", on="_name_norm")

    # Cast numerici e calcolo factor per ruolo
    auction["FVM"] = pd.to_numeric(auction["FVM"], errors="coerce")
    auction[price_col] = pd.to_numeric(auction[price_col], errors="coerce")

    # normalizza RUOLO (prendiamo solo la prima lettera maiuscola tra P/D/C/A se presente)
    if "RUOLO" in auction.columns:
        auction["RUOLO"] = auction["RUOLO"].astype(str).str.strip().str.upper().str[0]
    else:
        auction["RUOLO"] = ""

    def role_to_factor(r: str) -> float:
        return ROLE_FACTORS.get(r, DEFAULT_FACTOR)

    auction["_factor"] = auction["RUOLO"].map(role_to_factor)
    WEIGHT_PRICE = 0.40  # peso del prezzo attuale
    WEIGHT_FVM   = 0.60  # peso della parte FVM corretta

    mask = auction["FVM"].notna() & auction[price_col].notna()
    auction.loc[mask, price_col] = (
        auction.loc[mask, price_col] * WEIGHT_PRICE
        + (auction.loc[mask, "FVM"] / 2.0 * auction.loc[mask, "_factor"]) * WEIGHT_FVM
    )
    # Pulisci colonne temporanee
    drop_cols = [c for c in ["FVM", "_name_norm", "_factor", "RUOLO"] if c in auction.columns]
    if drop_cols:
        auction = auction.drop(columns=drop_cols)

    # 6) Salva CSV aggiornato
    auction.to_csv(OUT_AUCTION_UPDATED, index=False)

    # Report finale
    print("=== COMPLETATO ===")
    print(f"Foglio Excel usato: {sheet}")
    print(f"Colonne -> Nome: {name_col} | FVM: {fvm_col} | Ruolo: {role_col or 'N/D'}")
    print(f"Season filtrata: {SEASON}")
    print(f"Unmatched salvati -> {OUT_UNMATCHED} (righe: {len(unmatched_out)})")
    print(f"Auction aggiornato -> {OUT_AUCTION_UPDATED}")

if __name__ == "__main__":
    main()
