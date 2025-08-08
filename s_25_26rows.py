#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script standalone: crea le righe s_25_26 partendo da:
- data/giocatori_stagioni.csv
- data_retriever_fbref/Statistiche_Fantacalcio_Stagione_2025_26.xlsx (foglio 'Tutti')

Salva:
- giocatori_stagioni_25_26.csv

Regole:
- Per ogni player_id presente in s_24_25 crea una riga s_25_26.
- Merge con Excel per NOME normalizzato:
    * se match "sicuro" (nome non ambiguo né nel CSV né nell'Excel): aggiorna role (da colonna R) e team_name_short (mappando "Squadra"), setta tournament_name/ country a Serie A/Italy.
    * se nome ambiguo (duplicato nel CSV o nell'Excel): NON aggiorna da Excel, tiene squadra/ruolo della s_24_25. Report in nomi_ambigui.csv.
- Azzeramento colonne numeriche di statistica, preservando ID e anagrafiche.
"""

from pathlib import Path
import re
import unicodedata
from typing import Dict, List, Set

import numpy as np
import pandas as pd


# ---------- Utility: normalizza nomi per il match ---------- #
def norm_name(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s)
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\bjr\b|\bjunior\b|\bsr\b|\bsenior\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def main() -> None:
    # ---- Percorsi fissi come richiesto ---- #
    in_csv = Path("data") / "giocatori_stagioni.csv"
    in_xlsx = Path("data_retriever_fbref") / "Statistiche_Fantacalcio_Stagione_2025_26.xlsx"
    out_csv = Path("giocatori_stagioni_25_26.csv")

    train_until = "s_24_25"
    predict = "s_25_26"
    excel_sheet = "Tutti"

    if not in_csv.exists():
        raise FileNotFoundError(f"CSV non trovato: {in_csv.resolve()}")
    if not in_xlsx.exists():
        raise FileNotFoundError(f"Excel non trovato: {in_xlsx.resolve()}")

    # ---- 1) Carica storico ---- #
    df = pd.read_csv(in_csv)
    required_cols = {"season", "name", "player_id", "team_name_short", "role"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV: mancano colonne richieste: {sorted(missing)}")

    base = (
        df[df["season"] == train_until]
        .sort_values(["player_id"])
        .drop_duplicates(subset=["player_id"], keep="last")
        .copy()
    )
    if base.empty:
        raise ValueError(f"Nessuna riga trovata per {train_until} in {in_csv.name}")

    base["norm_name"] = base["name"].apply(norm_name)
    # Flag nomi duplicati nel CSV (left)
    base["dup_in_csv"] = base["norm_name"].duplicated(keep=False)

    # Evita duplicati se esistono già righe s_25_26
    already_ids = set(df.loc[df["season"] == predict, "player_id"].unique())
    if already_ids:
        base = base[~base["player_id"].isin(already_ids)].copy()

    print(f"Storico totale: {len(df):,} righe | Base {train_until}: {len(base):,} giocatori "
          f"({len(already_ids)} già presenti in {predict} verranno saltati)")

    # ---- 2) Carica Excel ---- #
    # Molti file Fantacalcio hanno header utile alla riga 2 -> header=1
    x = pd.read_excel(in_xlsx, sheet_name=excel_sheet, header=1)
    expected_excel = {"Nome", "R", "Squadra"}
    if not expected_excel.issubset(x.columns):
        # fallback: prova header=0
        x = pd.read_excel(in_xlsx, sheet_name=excel_sheet, header=0)
        if not expected_excel.issubset(x.columns):
            missing = expected_excel - set(x.columns)
            raise ValueError(f"Excel: mancano colonne {sorted(missing)} nel foglio '{excel_sheet}'")

    x = x.rename(columns={"Nome": "name_excel", "R": "role_excel", "Squadra": "team_full"})
    x["norm_name"] = x["name_excel"].apply(norm_name)
    # Flag nomi duplicati nell'Excel (right)
    dup_in_excel_map = x["norm_name"].value_counts().gt(1)

    # ---- 3) Mappa 'team_full' → 'team_name_short' (adattare se necessario) ---- #
    TEAM_MAP: Dict[str, str] = {
        "atalanta": "ATA", "bologna": "BOL", "cagliari": "CAG", "como": "COM", "cremonese": "CRE",
        "empoli": "EMP", "fiorentina": "FIO", "genoa": "GEN", "hellas verona": "VER", "verona": "VER",
        "inter": "INT", "juventus": "JUV", "lazio": "LAZ", "lecce": "LEC", "milan": "MIL",
        "monza": "MON", "napoli": "NAP", "parma": "PAR", "pisa": "PIS", "roma": "ROM",
        "torino": "TOR", "udinese": "UDI", "venezia": "VEN",
        # compatibilità storica (potrebbero non essere in A nel 25/26, ma li lasciamo):
        "sassuolo": "SAS", "salernitana": "SAL", "frosinone": "FRO", "spezia": "SPE",
    }
    x["team_name_short_new"] = x["team_full"].astype(str).str.lower().map(TEAM_MAP)

    # ---- 4) Join per nome normalizzato (many_to_one) ---- #
    merged = base.merge(
        x[["norm_name", "role_excel", "team_full", "team_name_short_new"]],
        on="norm_name",
        how="left",
        # niente validate="one_to_one" per gestire nomi duplicati nel CSV
    )

    # ---- 5) Costruisci righe future ---- #
    fut = merged.copy()
    fut["season"] = predict

    # Flag nomi duplicati nell'Excel (mappato per riga)
    fut["dup_in_excel"] = fut["norm_name"].map(dup_in_excel_map.to_dict()).fillna(False)

    # match "sicuro" = non duplicato nel CSV, non duplicato in Excel, e c'è almeno un dato dall'Excel
    fut["has_excel_data"] = fut["role_excel"].notna() | fut["team_name_short_new"].notna()
    fut["safe_match"] = (~fut["dup_in_csv"]) & (~fut["dup_in_excel"]) & (fut["has_excel_data"])

    # Aggiorna role/team SOLO dove il match è sicuro
    fut.loc[fut["safe_match"] & fut["role_excel"].notna(), "role"] = fut.loc[
        fut["safe_match"] & fut["role_excel"].notna(), "role_excel"
    ]
    fut.loc[fut["safe_match"] & fut["team_name_short_new"].notna(), "team_name_short"] = fut.loc[
        fut["safe_match"] & fut["team_name_short_new"].notna(), "team_name_short_new"
    ]

    # Se la squadra è stata aggiornata dall'Excel (match sicuro), forza campionato/paese
    fut["tournament_name"] = np.where(
        fut["safe_match"] & fut["team_name_short_new"].notna(),
        "Serie A",
        fut["tournament_name"] if "tournament_name" in fut.columns else "Serie A",
    )
    fut["tournament_country"] = np.where(
        fut["safe_match"] & fut["team_name_short_new"].notna(),
        "Italy",
        fut["tournament_country"] if "tournament_country" in fut.columns else "Italy",
    )

    # ---- 6) Azzerare colonne numeriche di statistica, preservando ID/anagrafiche ---- #
    # Regole: NON azzerare campi ID, year, age, height, jersey_number, birth_year, last_active_year
    numeric_cols: List[str] = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    explicit_keep: Set[str] = {
        "player_id", "team_id", "height", "jersey_number", "age", "birth_year", "last_active_year"
    }

    def should_zero(col: str) -> bool:
        c = col.lower()
        if c in explicit_keep:
            return False
        # escludi solo le colonne che finiscono davvero in "_id" o si chiamano esattamente "id"
        if c.endswith("_id") or c == "id":
            return False
        if "year" in c:
            return False
        return True

    stat_cols_zero = [c for c in numeric_cols if should_zero(c)]
    for c in stat_cols_zero:
        if c in fut.columns:
            # assicuriamoci che la colonna gestisca NaN convertendola a float
            fut[c] = fut[c].astype(float)
            fut[c] = np.nan

    # ---- 7) Pulisci helper e ordina colonne come input ---- #
    fut = fut.drop(columns=[
        "norm_name", "dup_in_csv", "dup_in_excel", "has_excel_data",
        "role_excel", "team_full", "team_name_short_new"
    ], errors="ignore")
    fut = fut[df.columns.tolist()]  # stesso ordine colonne dell'input

    # ---- 8) Unisci allo storico e salva ---- #
    out = pd.concat([df, fut], ignore_index=True)
    out.to_csv(out_csv, index=False)

    # ---- 9) Report rapidi ---- #
    # Nomi ambigui = duplicati nel CSV o nell’Excel
    ambiguous_names = merged.loc[
        merged["dup_in_csv"] | merged["norm_name"].map(dup_in_excel_map.to_dict()).fillna(False),
        "name"
    ].drop_duplicates().sort_values()
    if len(ambiguous_names) > 0:
        ambiguous_names.to_csv("nomi_ambigui.csv", index=False)

    not_mapped_teams = (
        x.loc[x["team_full"].notna() & x["team_name_short_new"].isna(), "team_full"]
        .drop_duplicates().sort_values()
    )
    if len(not_mapped_teams) > 0:
        not_mapped_teams.to_csv("teams_non_mappati.csv", index=False)

    updated_count = int((fut["season"] == predict).sum())
    print(f"✅ Create {updated_count:,} righe {predict}. Totale output: {len(out):,}")
    print(f"💾 Salvato in: {out_csv.resolve()}")
    if len(ambiguous_names) > 0:
        print(f"⚠️ Nomi ambigui (non aggiornati da Excel): {len(ambiguous_names)} – vedi nomi_ambigui.csv")
    if len(not_mapped_teams) > 0:
        print(f"⚠️ Team Excel non mappati: {len(not_mapped_teams)} – vedi teams_non_mappati.csv")


if __name__ == "__main__":
    main()
