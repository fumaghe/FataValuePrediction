#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update FantaValue CSVs by merging in new players and updating role/team_name_short.

- Adds missing players (by slug) from the "new" files.
- Updates only `role` and `team_name_short` for existing players.
  * giocatori_stagioni: restrict updates/additions to season s_25_26 (configurable)
  * auction_prices and future_predictions: update globally (no season filter)

Usage (defaults match filenames you attached):
    python update_fantavalue_files.py \
        --giocatori giocatori_stagioni.csv \
        --auction   auction_prices.csv \
        --future    future_predictions_s_25_26.csv \
        --new_giocatori new_giocatori.csv \
        --new_auction   new_auction.csv \
        --new_future    new_future.csv \
        --out_giocatori giocatori_stagioni_updated.csv \
        --out_auction   auction_prices_updated.csv \
        --out_future    future_predictions_s_25_26_updated.csv \
        --season s_25_26

Author: KLB (Kenny Luigi Boateng)
"""
from __future__ import annotations
import argparse
import pandas as pd
from typing import Dict, Iterable, Tuple

def safe_read_csv(path: str) -> pd.DataFrame:
    """
    Read CSV trying to auto-detect the separator. Falls back to semicolon.
    """
    try:
        return pd.read_csv(path, sep=None, engine="python")
    except Exception:
        return pd.read_csv(path, sep=";")

def merge_update_specific(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    key: str = "slug",
    cols_to_update: Iterable[str] = ("role", "team_name_short"),
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Left-merge df_new (deduped by key) onto df_base and update only given columns.
    Returns updated df and counts per updated column.
    """
    df_new_dedup = df_new.drop_duplicates(subset=[key], keep="last")
    merged = df_base.merge(
        df_new_dedup[[key] + list(cols_to_update)],
        on=key, how="left", suffixes=("", "_new")
    )
    updated_counts: Dict[str, int] = {}
    for c in cols_to_update:
        c_new = f"{c}_new"
        if c not in merged.columns or c_new not in merged.columns:
            updated_counts[c] = 0
            continue
        mask = merged[c_new].notna() & (merged[c] != merged[c_new])
        updated_counts[c] = int(mask.sum())
        merged.loc[mask, c] = merged.loc[mask, c_new]
        merged.drop(columns=[c_new], inplace=True)
    # drop any leftover *_new columns just in case
    leftover_new = [c for c in merged.columns if c.endswith("_new")]
    if leftover_new:
        merged = merged.drop(columns=leftover_new)
    return merged, updated_counts

def add_missing_by_key(
    df_base: pd.DataFrame,
    df_new: pd.DataFrame,
    key: str = "slug",
) -> Tuple[pd.DataFrame, int]:
    """
    Add rows from df_new that are not present in df_base (by `key`).
    Keeps base columns (and appends any extras at the end if present).
    Returns combined df and number of added rows.
    """
    base_keys = set(df_base[key].astype(str)) if key in df_base.columns else set()
    if key not in df_new.columns:
        # nothing to add (no key to match)
        return df_base.copy(), 0
    df_new_notin = df_new[~df_new[key].astype(str).isin(base_keys)].copy()

    # Ensure base columns exist in df_new_notin
    for col in df_base.columns:
        if col not in df_new_notin.columns:
            df_new_notin[col] = pd.NA

    # Order columns
    extras = [c for c in df_new_notin.columns if c not in df_base.columns]
    df_new_notin = df_new_notin[df_base.columns.tolist() + extras]

    combined = pd.concat([df_base, df_new_notin], ignore_index=True)
    return combined, int(len(df_new_notin))

def process_giocatori_stagioni(
    path_base: str,
    path_new: str,
    out_path: str,
    season_filter: str = "s_25_26",
) -> Dict[str, int | str]:
    df_base = safe_read_csv(path_base)
    df_new = safe_read_csv(path_new)

    # 1) ADD: aggiungi TUTTE le stagioni per gli slug NUOVI
    base_slugs = set(df_base["slug"].astype(str)) if "slug" in df_base.columns else set()
    new_slugs_all = set(df_new["slug"].astype(str)) if "slug" in df_new.columns else set()
    slugs_to_add = new_slugs_all - base_slugs
    df_new_for_add = df_new[df_new["slug"].astype(str).isin(slugs_to_add)].copy()
    df_added, added = add_missing_by_key(df_base, df_new_for_add, key="slug")

    # 2) UPDATE: solo s_25_26 per role/team_name_short
    if "season" in df_new.columns:
        df_new_for_update = df_new[df_new["season"] == season_filter].copy()
    else:
        df_new_for_update = df_new.copy()

    if "season" in df_added.columns:
        mask_2526 = df_added["season"] == season_filter
        df_target = df_added.loc[mask_2526].copy()
        df_target_updated, upd_counts = merge_update_specific(
            df_target, df_new_for_update, key="slug", cols_to_update=("role", "team_name_short")
        )
        df_final = df_added.copy()
        df_final.loc[mask_2526, ["role", "team_name_short"]] = df_target_updated[["role", "team_name_short"]].values
    else:
        df_final, upd_counts = merge_update_specific(
            df_added, df_new_for_update, key="slug", cols_to_update=("role", "team_name_short")
        )

    df_final.to_csv(out_path, index=False)
    return {
        "added_rows": added,
        "updated_role": int(upd_counts.get("role", 0)),
        "updated_team_name_short": int(upd_counts.get("team_name_short", 0)),
        "output": out_path,
    }


def process_generic(
    path_base: str,
    path_new: str,
    out_path: str,
) -> Dict[str, int | str]:
    df_base = safe_read_csv(path_base)
    df_new = safe_read_csv(path_new)

    # Add missing (by slug), then update globally (no season filter)
    df_added, added = add_missing_by_key(df_base, df_new, key="slug")
    df_final, upd_counts = merge_update_specific(
        df_added, df_new, key="slug", cols_to_update=("role", "team_name_short")
    )
    df_final.to_csv(out_path, index=False)
    return {
        "added_rows": added,
        "updated_role": int(upd_counts.get("role", 0)),
        "updated_team_name_short": int(upd_counts.get("team_name_short", 0)),
        "output": out_path,
    }

def main():
    p = argparse.ArgumentParser(description="Merge 'new' CSVs into existing FantaValue datasets.")
    p.add_argument("--giocatori", default="giocatori_stagioni.csv")
    p.add_argument("--auction",   default="auction_prices.csv")
    p.add_argument("--future",    default="future_predictions_s_25_26.csv")
    p.add_argument("--new_giocatori", default="new_giocatori.csv")
    p.add_argument("--new_auction",   default="new_auction.csv")
    p.add_argument("--new_future",    default="new_future.csv")
    p.add_argument("--out_giocatori", default="giocatori_stagioni_updated.csv")
    p.add_argument("--out_auction",   default="auction_prices_updated.csv")
    p.add_argument("--out_future",    default="future_predictions_s_25_26_updated.csv")
    p.add_argument("--season", default="s_25_26", help="Season to filter for giocatori_stagioni updates/additions.")
    args = p.parse_args()

    report_g = process_giocatori_stagioni(args.giocatori, args.new_giocatori, args.out_giocatori, season_filter=args.season)
    report_a = process_generic(args.auction, args.new_auction, args.out_auction)
    report_f = process_generic(args.future, args.new_future, args.out_future)

    print("== Report ==")
    print("giocatori_stagioni:", report_g)
    print("auction_prices:", report_a)
    print("future_predictions_s_25_26:", report_f)

if __name__ == "__main__":
    main()
