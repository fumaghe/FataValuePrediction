#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FantaValue – Roster Sync & Cleanup Tool
=======================================
Aggiorna i dataset (giocatori_stagioni, future_predictions, auction_prices) 
partendo dall'Excel "Quotazioni_Fantacalcio_Stagione_2025_26.xlsx" → foglio "Tutti"
(escludendo i "Ceduti"), applicando:
- Cambio squadra per chi ha cambiato team:
  - future_predictions_s_25_26.csv e auction_prices.csv → sempre
  - giocatori_stagioni.csv → solo righe con season == "s_25_26"
- Rimozione dei giocatori non più presenti in Excel "Tutti" (post-exclude Ceduti)
  - Rimozione da TUTTI e 3 i dataset (usando slug)
- Esporta i "nuovi" (presenti in Excel ma non in giocatori_stagioni) in giocatori_nuovi.csv
  con colonne: Nome, Squadra, R, tournament_name="Serie A"

Uso
----
python fv_sync_roster.py \
  --gs /path/giocatori_stagioni.csv \
  --fp /path/future_predictions_s_25_26.csv \
  --ap /path/auction_prices.csv \
  --excel /path/Quotazioni_Fantacalcio_Stagione_2025_26.xlsx \
  --players /path/serie_a_players.csv \
  --outdir /path/output \
  [--alias /path/alias_nome_slug.csv] \
  [--dry-run]

Note
----
- Matching principale per rimozioni: per nome normalizzato dal foglio "Tutti" (post-ceduti).
- Mapping slug per aggiornare team ed eliminazioni: usa join su slug quando disponibile,
  altrimenti su name_norm (con disambiguazione su team quando possibile).
- Il file alias_nome_slug.csv è opzionale: colonne attese ["Nome","slug"] per fix manuali.
"""

from __future__ import annotations
import argparse
import sys
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import pandas as pd
import numpy as np
import unicodedata
from pathlib import Path

SEASON_TARGET = "s_25_26"

# --- Hardcoded alias dictionaries (EDIT HERE) ---
# Mappa i nomi come compaiono nell'Excel "Tutti" -> nome desiderato (come nel CSV)
# Esempio: "Leao" in Excel ma "Rafael Leao" nei CSV
HARDCODED_NAME_ALIAS = {
    "Leao": "Rafael Leao",
    # "Donnarumma An": "Gianluigi Donnarumma",
}

# (Opzionale) Mappa diretta Nome Excel -> slug (se vuoi bypassare ogni matching)
# Se presente, ha PRIORITÀ ASSOLUTA. Usa lo slug del tuo sistema (FBref).
HARDCODED_NAME_TO_SLUG = {
    # "Leao": "rafael-leao",
}
# --- end alias dictionaries ---


# ------------------------- Utilities -------------------------

def normalize_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join([c for c in s if not unicodedata.combining(c)])
    repl = {
        ".":"",
        "'":" ",
        "’":" ",
        "`":" ",
        "´":" ",
        "-":" ",
        ",":" ",
        "(": " ",
        ")": " ",
        "/": " ",
    }
    for k,v in repl.items():
        s = s.replace(k, v)
    s = " ".join(s.split())
    return s.lower()


TEAM_SYNONYMS = {
    # Common normalizations
    "internazionale": "inter",
    "fc internazionale": "inter",
    "inter milano": "inter",
    "inter milan": "inter",
    "milan": "milan",
    "ac milan": "milan",
    "juventus": "juventus",
    "napoli": "napoli",
    "ssc napoli": "napoli",
    "lazio": "lazio",
    "roma": "roma",
    "as roma": "roma",
    "atalanta": "atalanta",
    "bologna": "bologna",
    "fiorentina": "fiorentina",
    "torino": "torino",
    "genoa": "genoa",
    "monza": "monza",
    "lecce": "lecce",
    "udinese": "udinese",
    "cagliari": "cagliari",
    "empoli": "empoli",
    "verona": "verona",
    "hellas verona": "verona",
    "venezia": "venezia",
    "como": "como",
    "parma": "parma",
    "pisa": "pisa",
    # Add more if needed
}
def normalize_team(s: str) -> str:
    s_norm = normalize_text(s)
    return TEAM_SYNONYMS.get(s_norm, s_norm)


def ensure_columns(df: pd.DataFrame, required: List[str], label: str) -> None:
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"[{label}] Missing required columns: {miss}")


def load_excel_tutti_and_ceduti(path_xlsx: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # prefer header in row 1; fallback to row 0
    try:
        tutti = pd.read_excel(path_xlsx, sheet_name="Tutti", header=1)
    except Exception:
        tutti = pd.read_excel(path_xlsx, sheet_name="Tutti", header=0)
    try:
        ceduti = pd.read_excel(path_xlsx, sheet_name="Ceduti", header=1)
    except Exception:
        ceduti = pd.read_excel(path_xlsx, sheet_name="Ceduti", header=0)

    # Standardize column names by trimming
    tutti.rename(columns={c: c.strip() for c in tutti.columns}, inplace=True)
    ceduti.rename(columns={c: c.strip() for c in ceduti.columns}, inplace=True)

    # Keep relevant columns if present
    keep_tutti = [c for c in ["Id","R","RM","Nome","Squadra","Qt.I","Qt.A","Diff.","FVM"] if c in tutti.columns]
    if keep_tutti:
        tutti = tutti[keep_tutti].copy()

    keep_ced = [c for c in ["Nome","Squadra"] if c in ceduti.columns]
    if keep_ced:
        ceduti = ceduti[keep_ced].copy()

    ensure_columns(tutti, ["Nome","Squadra"], "Excel::Tutti")
    ensure_columns(ceduti, ["Nome"], "Excel::Ceduti")

    # Apply hardcoded name alias BEFORE normalization
    if HARDCODED_NAME_ALIAS:
        tutti["Nome"] = tutti["Nome"].map(lambda x: HARDCODED_NAME_ALIAS.get(str(x), x))
        ceduti["Nome"] = ceduti["Nome"].map(lambda x: HARDCODED_NAME_ALIAS.get(str(x), x))

    # Normalized names
    tutti["name_norm"] = tutti["Nome"].map(normalize_text)
    ceduti["name_norm"] = ceduti["Nome"].map(normalize_text)

    # Exclude Ceduti by name
    ceduti_set = set(ceduti["name_norm"])
    tutti = tutti[~tutti["name_norm"].isin(ceduti_set)].copy()

    # Normalized team for disambiguation
    tutti["team_norm_excel"] = tutti["Squadra"].map(normalize_team)

    return tutti, ceduti


@dataclass
class IOConfig:
    gs_path: Path
    fp_path: Path
    ap_path: Path
    excel_path: Path
    players_path: Path
    out_dir: Path
    alias_path: Optional[Path] = None
    dry_run: bool = False


# ------------------------- Core logic -------------------------

def build_name_slug_index(gs: pd.DataFrame, players: pd.DataFrame, alias: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Restituisce una tabella di mapping name_norm -> slug (potenzialmente multi-mappa, 1:N)
    Priorità:
      0) HARDCODED_NAME_TO_SLUG (Nome -> slug, priorità assoluta)
      1) alias manuali CSV (Nome -> slug)
      2) giocatori_stagioni season s_25_26 (più recente)
      3) serie_a_players (elenco anagrafica)
      4) fallback: altri season in gs (meno affidabili)
    """
    rows = []

    # 0) Hardcoded direct mapping Nome Excel -> slug (highest priority)
    if HARDCODED_NAME_TO_SLUG:
        h = pd.DataFrame([
            {"Nome": k, "slug": v} for k, v in HARDCODED_NAME_TO_SLUG.items()
        ])
        h["name_norm"] = h["Nome"].map(normalize_text)
        h["source"] = "alias_hard_slug"
        rows.append(h[["name_norm","slug","source"]])

    if alias is not None and not alias.empty:
        a = alias.rename(columns={"Nome":"Nome"})
        a["name_norm"] = a["Nome"].map(normalize_text)
        ensure_columns(a, ["name_norm","slug"], "Alias")
        a["source"] = "alias"
        rows.append(a[["name_norm","slug","source"]])

    gs["name_norm"] = gs["name"].map(normalize_text)
    gs["season_norm"] = gs["season"].astype(str).str.lower()
    gs_s = gs[gs["season_norm"].eq(SEASON_TARGET)].dropna(subset=["slug"])
    if not gs_s.empty:
        tmp = gs_s[["name_norm","slug"]].drop_duplicates().copy()
        tmp["source"] = "gs_current"
        rows.append(tmp)

    players["name_norm"] = players["name"].map(normalize_text)
    tmp = players[["name_norm","slug"]].drop_duplicates().copy()
    tmp["source"] = "players"
    rows.append(tmp)

    gs_others = gs[~gs["season_norm"].eq(SEASON_TARGET)].dropna(subset=["slug"])
    if not gs_others.empty:
        tmp = gs_others[["name_norm","slug"]].drop_duplicates().copy()
        tmp["source"] = "gs_past"
        rows.append(tmp)

    mapping = pd.concat(rows, ignore_index=True).drop_duplicates()
    return mapping


def map_excel_to_slugs(excel_tutti: pd.DataFrame, name_slug_map: pd.DataFrame, gs_current: pd.DataFrame) -> pd.DataFrame:
    """
    Produce excel_mapped con colonne: [Nome, Squadra, R, name_norm, team_norm_excel, slug?, team_name_short?]
    - Primo match: name_norm -> slug usando priorità del mapping
    - Disambiguazione se più slug per lo stesso name_norm:
        usa team attuale in gs_current (season s_25_26) confrontato con team_norm_excel.
    """
    df = excel_tutti.copy()
    df = df.drop_duplicates(subset=["name_norm"])

    # join to candidates
    candidates = df.merge(name_slug_map, on="name_norm", how="left", suffixes=("",""))
    # Attach team from gs_current to disambiguate
    gs_cur = gs_current.copy()
    gs_cur["team_norm_gs"] = gs_cur["team_name_short"].map(normalize_team)
    gs_cur = gs_cur[["slug","team_norm_gs"]].drop_duplicates()

    candidates = candidates.merge(gs_cur, on="slug", how="left")

    # For name_norm with multiple slug candidates, pick the one where team_norm_excel == team_norm_gs if possible
    chosen_rows = []
    for name, group in candidates.groupby("name_norm", dropna=False):
        if group["slug"].notna().sum() == 0:
            chosen_rows.append(group.iloc[[0]][df.columns.tolist() + ["slug"]])  # keep without slug (unmapped)
            continue
        # Prefer exact team match
        matches = group[group["team_norm_gs"] == group["team_norm_excel"]]
        if not matches.empty:
            chosen_rows.append(matches.iloc[[0]][df.columns.tolist() + ["slug"]])
        else:
            # fallback to first by priority order (alias_hard_slug > alias > gs_current > players > gs_past)
            priority = {"alias_hard_slug": 0, "alias": 1, "gs_current": 2, "players": 3, "gs_past": 4}
            group["prio"] = group["source"].map(lambda s: priority.get(s, 99))
            group = group.sort_values(["prio"]).copy()
            chosen_rows.append(group.iloc[[0]][df.columns.tolist() + ["slug"]])
    mapped = pd.concat(chosen_rows, ignore_index=True)
    return mapped


def update_team_in_df(df: pd.DataFrame, slug_to_team: Dict[str,str], slug_col="slug", team_col_candidates=("team_name_short","team")) -> Tuple[pd.DataFrame, int]:
    df = df.copy()
    team_col = None
    for c in team_col_candidates:
        if c in df.columns:
            team_col = c
            break
    if team_col is None:
        return df, 0

    changed = 0
    for i, row in df.iterrows():
        slug = row.get(slug_col, np.nan)
        if pd.isna(slug):
            continue
        s = str(slug)
        if s in slug_to_team:
            new_team = slug_to_team[s]
            old_team = row.get(team_col, None)
            if pd.isna(old_team) or str(old_team) != str(new_team):
                df.at[i, team_col] = new_team
                changed += 1
    return df, changed


def run_pipeline(cfg: IOConfig) -> Dict[str, object]:
    # Load inputs
    gs = pd.read_csv(cfg.gs_path)
    fp = pd.read_csv(cfg.fp_path)
    ap = pd.read_csv(cfg.ap_path)
    players = pd.read_csv(cfg.players_path)

    alias = None
    if cfg.alias_path and cfg.alias_path.exists():
        alias = pd.read_csv(cfg.alias_path)

    # Excel
    tutti, ceduti = load_excel_tutti_and_ceduti(cfg.excel_path)

    # Prepare name normalization for gs & players
    for df in (gs, players):
        if "name" not in df.columns:
            raise ValueError("[Input] Missing 'name' column")
        df["name_norm"] = df["name"].map(normalize_text)

    # Restrict current season rows (for disambiguation & team snapshots)
    if "season" not in gs.columns:
        raise ValueError("[giocatori_stagioni] Missing 'season' column")
    gs_current = gs[gs["season"].astype(str).str.lower().eq(SEASON_TARGET)].copy()
    keep_cols = ["slug","name","name_norm","team_name_short","season"]
    for c in keep_cols:
        if c not in gs_current.columns:
            # tolerate missing team_name_short by creating it if needed
            if c == "team_name_short":
                gs_current["team_name_short"] = np.nan
            else:
                raise ValueError(f"[giocatori_stagioni] Missing required column '{c}'")
    gs_current = gs_current[keep_cols].drop_duplicates()

    # Build mapping name -> slug
    name_slug_map = build_name_slug_index(gs, players, alias)

    # Map Excel rows to slugs (+ disambig by team when possible)
    excel_mapped = map_excel_to_slugs(tutti, name_slug_map, gs_current)

    # -------- Identify changes & compute slug sets --------
    # Slugs present in Excel after mapping
    slugs_in_excel: Set[str] = set(excel_mapped["slug"].dropna().astype(str))

    # For deletions: use NAMES as ground truth to avoid false negatives from unmapped slugs
    names_in_excel: Set[str] = set(excel_mapped["name_norm"])

    # Names present in giocatori_stagioni (any season)
    names_in_gs: Set[str] = set(gs["name_norm"])

    # Names to remove = in gs but NOT in excel (post-ceduti)
    names_to_remove = sorted(names_in_gs - names_in_excel)

    # Derive slugs to remove by collecting all slugs with those names
    slugs_to_remove = set(gs.loc[gs["name_norm"].isin(names_to_remove), "slug"].dropna().astype(str))

    # For team updates, build slug -> new team from Excel mapped (where slug is known)
    slug_team_pairs = excel_mapped.dropna(subset=["slug"])[["slug","Squadra"]].drop_duplicates()
    slug_to_team = dict(zip(slug_team_pairs["slug"].astype(str), slug_team_pairs["Squadra"].astype(str)))

    # -------- Apply TEAM updates --------
    fp_updated, fp_changes = update_team_in_df(fp, slug_to_team, slug_col="slug", team_col_candidates=("team_name_short","team"))
    ap_updated, ap_changes = update_team_in_df(ap, slug_to_team, slug_col="slug", team_col_candidates=("team_name_short","team"))

    # giocatori_stagioni: only SEASON_TARGET rows
    gs_updated = gs.copy()
    mask_s = gs_updated["season"].astype(str).str.lower().eq(SEASON_TARGET)
    sub = gs_updated.loc[mask_s].copy()
    sub_updated, gs_changes = update_team_in_df(sub, slug_to_team, slug_col="slug", team_col_candidates=("team_name_short","team"))
    gs_updated.loc[mask_s, :] = sub_updated

    # -------- Delete players not in Excel --------
    if slugs_to_remove:
        gs_updated = gs_updated[~gs_updated["slug"].astype(str).isin(slugs_to_remove)].copy()
        fp_updated = fp_updated[~fp_updated["slug"].astype(str).isin(slugs_to_remove)].copy()
        ap_updated = ap_updated[~ap_updated["slug"].astype(str).isin(slugs_to_remove)].copy()

    # -------- New players file (hardcoded-aware) --------
    # Considera l'alias HARDCODED_NAME_ALIAS (già applicato a 'tutti') e il mapping diretto HARDCODED_NAME_TO_SLUG
    # Se Excel mappa a uno slug esistente in gs, NON è un giocatore nuovo anche se il nome differisce.
    slugs_in_gs = set(gs["slug"].dropna().astype(str))
    if "slug" in excel_mapped.columns:
        mask_nuovi = excel_mapped["slug"].isna() | (~excel_mapped["slug"].astype(str).isin(slugs_in_gs))
    else:
        mask_nuovi = ~excel_mapped["name_norm"].isin(names_in_gs)
    base_cols = [c for c in ["Nome","Squadra","R"] if c in excel_mapped.columns]
    if not base_cols:
        base_cols = ["Nome","Squadra"]
    nuovi = excel_mapped.loc[mask_nuovi, base_cols].copy()
    nuovi["tournament_name"] = "Serie A"

    # -------- Reporting --------
    # Unmapped Excel rows (no slug)
    unmapped = excel_mapped[excel_mapped["slug"].isna()][["Nome","Squadra","R","name_norm"]].copy() if "R" in excel_mapped.columns else excel_mapped[excel_mapped["slug"].isna()][["Nome","Squadra","name_norm"]].copy()

    report = {
        "fp_team_changes": int(fp_changes),
        "ap_team_changes": int(ap_changes),
        "gs_team_changes_s_25_26": int(gs_changes),
        "names_removed_count": int(len(names_to_remove)),
        "slugs_removed_count": int(len(slugs_to_remove)),
        "new_players_count": int(len(nuovi)),
        "unmapped_excel_rows": int(len(unmapped)),
    }

    if cfg.dry_run:
        return {
            "report": report,
            "gs_updated": gs_updated.head(3),
            "fp_updated": fp_updated.head(3),
            "ap_updated": ap_updated.head(3),
            "nuovi": nuovi.head(10),
            "unmapped": unmapped.head(20),
        }

    # -------- Write outputs --------
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_gs = cfg.out_dir / "giocatori_stagioni_updated.csv"
    out_fp = cfg.out_dir / "future_predictions_s_25_26_updated.csv"
    out_ap = cfg.out_dir / "auction_prices_updated.csv"
    out_nuovi = cfg.out_dir / "giocatori_nuovi.csv"
    out_unmapped = cfg.out_dir / "excel_unmapped_rows.csv"
    out_log = cfg.out_dir / "aggiornamento_log.json"

    gs_updated.to_csv(out_gs, index=False)
    fp_updated.to_csv(out_fp, index=False)
    ap_updated.to_csv(out_ap, index=False)
    nuovi.to_csv(out_nuovi, index=False)
    unmapped.to_csv(out_unmapped, index=False)
    with open(out_log, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    return {
        "report": report,
        "paths": {
            "giocatori_stagioni_updated": str(out_gs),
            "future_predictions_s_25_26_updated": str(out_fp),
            "auction_prices_updated": str(out_ap),
            "giocatori_nuovi": str(out_nuovi),
            "excel_unmapped_rows": str(out_unmapped),
            "aggiornamento_log": str(out_log),
        }
    }


# ------------------------- CLI -------------------------

def parse_args(argv: Optional[List[str]] = None) -> IOConfig:
    p = argparse.ArgumentParser(description="FantaValue – Roster Sync & Cleanup Tool")
    p.add_argument("--gs", dest="gs_path", required=True, type=Path, help="Path a giocatori_stagioni.csv")
    p.add_argument("--fp", dest="fp_path", required=True, type=Path, help="Path a future_predictions_s_25_26.csv")
    p.add_argument("--ap", dest="ap_path", required=True, type=Path, help="Path a auction_prices.csv")
    p.add_argument("--excel", dest="excel_path", required=True, type=Path, help="Path all'Excel Quotazioni")
    p.add_argument("--players", dest="players_path", required=True, type=Path, help="Path a serie_a_players.csv")
    p.add_argument("--outdir", dest="out_dir", required=True, type=Path, help="Cartella di output")
    p.add_argument("--alias", dest="alias_path", type=Path, default=None, help="(Opzionale) CSV alias Nome->slug")
    p.add_argument("--dry-run", dest="dry_run", action="store_true", help="Esegue senza scrivere output, stampa un riassunto")

    # If user clicked "Run" in VS Code, argv may be None or empty -> use smart defaults
    if argv is None:
        argv = []

    # If no args provided, assume files are in the same folder as the script
    if len(argv) == 0:
        here = Path(__file__).resolve().parent
        default_out = here / "out"
        default_args = [
            "--gs", str(here / "giocatori_stagioni.csv"),
            "--fp", str(here / "future_predictions_s_25_26.csv"),
            "--ap", str(here / "auction_prices.csv"),
            "--excel", str(here / "Quotazioni_Fantacalcio_Stagione_2025_26.xlsx"),
            "--players", str(here / "serie_a_players.csv"),
            "--outdir", str(default_out),
        ]
        args = p.parse_args(default_args)
    else:
        args = p.parse_args(argv)

    return IOConfig(
        gs_path=args.gs_path,
        fp_path=args.fp_path,
        ap_path=args.ap_path,
        excel_path=args.excel_path,
        players_path=args.players_path,
        out_dir=args.out_dir,
        alias_path=args.alias_path,
        dry_run=args.dry_run
    )


def main(argv: Optional[List[str]] = None) -> int:
    cfg = parse_args(argv)
    try:
        res = run_pipeline(cfg)
    except Exception as e:
        print(f"[ERRORE] {e}", file=sys.stderr)
        return 1

    if cfg.dry_run:
        print(json.dumps(res["report"], ensure_ascii=False, indent=2))
    else:
        print("Output generati:")
        for k, v in res["paths"].items():
            print(f"- {k}: {v}")
        print("Report:")
        print(json.dumps(res["report"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
