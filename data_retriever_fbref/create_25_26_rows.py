#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MINIMAL – Usa SOLO due file e merge per SLUG, niente ordini colonne per evitare KeyError.

Input:
  1) data_retriever_fbref/players_seasons_stats_updated.csv  (BASE: contiene s_24_25)
  2) serie_a_players.csv                                     (OVERRIDE: role/team per slug)

Output:
  data_retriever_fbref/players_seasons_stats_updated_with_25_26.csv

Logica:
  - Prende dalla BASE solo le righe con season == "s_24_25" (1 per slug).
  - Crea copie con season == "s_25_26".
  - Merge per 'slug' con Serie A: se presenti, aggiorna 'role' e 'team_name_short'.
  - Azzera TUTTE le colonne numeriche di statistica tranne ID/anagrafiche.
  - NON riordina le colonne; mantiene l'ordine della BASE per evitare errori.
"""

from pathlib import Path
from typing import List, Set, Optional
import numpy as np
import pandas as pd


def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None


def main():
    players_path = Path("data_retriever_fbref") / "players_seasons_stats_updated.csv"
    seriea_path  = Path("serie_a_players.csv")
    out_path     = players_path.parent / "players_seasons_stats_updated_with_25_26.csv"

    season_src = "s_24_25"
    season_new = "s_25_26"

    # --- Carica BASE ---
    if not players_path.exists():
        raise FileNotFoundError(players_path.resolve())
    base_all = pd.read_csv(players_path)

    name_col   = pick_col(base_all, ["name","Nome","player_name","full_name"])
    slug_col   = pick_col(base_all, ["slug","player_slug","slug_fbref","id_slug"])
    role_col   = pick_col(base_all, ["role","R","pos","position"])
    team_col   = pick_col(base_all, ["team_name_short","team","Squadra","squadra","club","team_full"])
    season_col = pick_col(base_all, ["season","stagione"])

    needed = {"name":name_col, "slug":slug_col, "role":role_col, "team":team_col, "season":season_col}
    missing = [k for k,v in needed.items() if v is None]
    if missing:
        raise ValueError(f"Mancano colonne nella BASE: {missing}")

    # Filtra le righe s_24_25 e 1 per slug
    base = (
        base_all.loc[base_all[season_col] == season_src]
        .sort_values([slug_col])
        .drop_duplicates(subset=[slug_col], keep="last")
        .copy()
    )
    if base.empty:
        raise ValueError(f"Nessuna riga '{season_src}' in {players_path.name}")

    # Evita duplicati se in BASE ci sono già s_25_26 per alcuni slug
    existing_new = set(base_all.loc[base_all[season_col] == season_new, slug_col].astype(str).unique())
    if existing_new:
        base = base[~base[slug_col].astype(str).isin(existing_new)].copy()

    # --- Carica SERIE A ---
    if not seriea_path.exists():
        raise FileNotFoundError(seriea_path.resolve())
    seriea = pd.read_csv(seriea_path)

    slug_sa = pick_col(seriea, ["slug","player_slug","slug_fbref","id_slug"])
    role_sa = pick_col(seriea, ["role","R","pos","position"])
    team_sa = pick_col(seriea, ["team_name_short","team","Squadra","squadra","club","team_full"])
    if not slug_sa:
        raise ValueError("serie_a_players.csv deve avere 'slug'")

    # Prepara chiave 'slug' nella BASE: garantisco colonna 'slug' standard per il merge
    if slug_col != "slug":
        base = base.copy()
        base["slug"] = base[slug_col]
    else:
        # già 'slug'
        pass

    # Prepara df con override da Serie A
    sa = seriea[[slug_sa]].copy()
    sa.columns = ["slug"]
    sa["role_sa"] = seriea[role_sa] if role_sa else np.nan
    sa["team_sa"] = seriea[team_sa] if team_sa else np.nan

    # --- MERGE per SLUG ---
    fut = base.merge(sa, on="slug", how="left")

    # Imposta nuova stagione
    fut[season_col] = season_new

    # Aggiorna role/team con priorità Serie A
    fut[role_col] = np.where(fut["role_sa"].notna(), fut["role_sa"], fut[role_col])
    fut[team_col] = np.where(fut["team_sa"].notna(), fut["team_sa"], fut[team_col])

    # Azzera colonne numeriche di statistica (tranne ID/anagrafiche)
    numeric_cols: List[str] = [c for c in base_all.columns if pd.api.types.is_numeric_dtype(base_all[c])]
    explicit_keep: Set[str] = {"player_id","team_id","height","jersey_number","age","birth_year","last_active_year"}
    def should_zero(col: str) -> bool:
        cl = col.lower()
        if cl in explicit_keep: return False
        if cl.endswith("_id") or cl == "id": return False
        if "year" in cl: return False
        return True
    # azzera solo colonne presenti in fut
    for c in [c for c in numeric_cols if should_zero(c) and c in fut.columns]:
        fut[c] = fut[c].astype(float)
        fut[c] = np.nan

    # Rimuovi helper, ma NON togliere 'slug'
    fut = fut.drop(columns=["role_sa","team_sa"], errors="ignore")

    # Concat all'originale e salva
    out = pd.concat([base_all, fut], ignore_index=True)
    out.to_csv(out_path, index=False)

    # Log
    print(f"✅ Create {len(fut):,} righe {season_new}. Totale output: {len(out):,}")
    print(f"💾 Salvato in: {out_path.resolve()}")
    upd_role = int((fut[role_col].values != base[role_col].reindex(fut.index, fill_value=np.nan).values).sum())
    upd_team = int((fut[team_col].values != base[team_col].reindex(fut.index, fill_value=np.nan).values).sum())
    print(f"🔁 Ruoli aggiornati da Serie A: {upd_role}")
    print(f"🔁 Squadre aggiornate da Serie A: {upd_team}")

if __name__ == "__main__":
    main()
