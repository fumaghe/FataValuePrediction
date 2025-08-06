#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .settings import DATA_DIR, TARGETS_ALL, TARGETS_CORE
from .utils.logging import setup as setup_logging
from .data_processing import hierarchical_impute, aggregate_midseason_rows
from .feature_engineering import build_features, build_future_dataframe
from .modeling.minutes import minutes_regressor
from .modeling.lgb_optuna import fit_predict_by_role
from .modeling.postprocessing import postprocess


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fantapred",
        description="Pipeline completa di addestramento e previsione stagione futura.",
    )
    parser.add_argument("--csv", default=str(DATA_DIR / "giocatori_stagioni.csv"),
                        help="Path al CSV storico")
    parser.add_argument("--train_until", default="s_24_25",
                        help="Ultima stagione inclusa nel training")
    parser.add_argument("--predict_season", default="s_25_26",
                        help="Stagione futura da proiettare")
    parser.add_argument("--targets", choices=["all", "core"], default="all",
                        help="Target da predire")
    parser.add_argument("--bonus_mode", choices=["curve", "linear", "off"],
                        default="curve",
                        help="Logica bonus gol/assist per fmv (default: curve)")
    parser.add_argument("--round_stats", action="store_true",
                        help="Arrotonda voti/contatori a 2 decimali")
    parser.add_argument("--log_file", default=None, help="Log file (opzionale)")
    args = parser.parse_args()

    setup_logging(log_file=args.log_file)

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV non trovato: {csv_path}")

    # 1) LOAD + aggrega trasferimenti
    df_raw = pd.read_csv(csv_path)
    df_raw = aggregate_midseason_rows(df_raw)

    # 2) FEATURE ENGINEERING
    df_prep = build_features(hierarchical_impute(df_raw))

    # 3) COSTRUISCI STAGIONE FUTURA
    fut_df = build_future_dataframe(df_prep, args.train_until, args.predict_season)

    # 4) MINUTI
    combined = pd.concat([df_prep, fut_df], ignore_index=True)
    combined = build_features(hierarchical_impute(combined))
    combined = minutes_regressor(combined, args.train_until)

    # 5) PREDIZIONI
    targets    = TARGETS_ALL if args.targets == "all" else TARGETS_CORE
    models_dir = Path("models")
    for tgt in tqdm(targets, desc="Target", leave=False):
        combined = combined.join(
            fit_predict_by_role(combined, tgt, args.train_until, models_dir),
            how="left",
        )

    # 6) POST-PROCESS & EXPORT
    out_df = combined[combined.season == args.predict_season].copy()
    out_df = postprocess(out_df,
                         bonus_mode=args.bonus_mode,
                         round_stats=args.round_stats)

    pred_cols = [c for c in out_df.columns if c.endswith("_pred")]
    cols      = ["slug", "role", "team_name_short", "presenze_pred", *pred_cols]
    outfile   = Path(f"future_predictions_{args.predict_season}.csv")
    out_df[cols].to_csv(outfile, index=False)
    print(f"✅  Predizioni salvate in {outfile.resolve()}  –  {len(out_df)} righe")


if __name__ == "__main__":
    main()
