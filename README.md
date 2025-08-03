FataValuePrediction
A complete pipeline for scraping, cleaning and merging Fantacalcio (Italian fantasy football) statistics, training LightGBM models to forecast next-season performance, and converting those forecasts into recommended auction prices.

Table of Contents
Project Overview

Features

Directory Structure

Installation

Usage

Data Retrieval

Data Preparation & Budget Analysis

Training & Prediction

Auction Pricing

Key Modules

data_retriever_fbref

fantapred

models

utilities & scripts

Outputs

Contributing

License

Project Overview
This repository automates the end-to-end workflow for Fantacalcio value prediction:

Scrape detailed per-season player stats from FBref.

Enrich with additional Fantacalcio data from Excel sheets.

Clean & Impute missing values using hierarchical strategies.

Engineer Features (per-90 metrics, age decay, team strength).

Train Models (LightGBM + Optuna tuning) to predict goals, assists, ratings, clean sheets, and monetary value.

Post-process predictions and convert them into auction-style price recommendations.

Features
Robust Scraping of FBref with Cloudflare-aware retries

Two-pass Merging of Excel-based Fantacalcio stats

Hierarchical Imputation → player, team-role, role levels + iterative imputer

Feature Engineering: age curves, league/coach coefficients, attack strength

Optuna-tuned LightGBM per role & target

Auction Pricing Tool that respects role-credit pools, difficulty coefficients, and free-slot minimums

Directory Structure
bash
Copia
Modifica
FataValuePrediction/
├── data/                             # Raw & processed CSVs
├── data_retriever_fbref/             # FBref scraper & merger scripts
├── fantapred/                        # Core pipeline: CLI, processing, features, modeling
│   ├── cli.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling/
│   │   ├── minutes.py
│   │   ├── lgb_optuna.py
│   │   └── postprocessing.py
│   └── tools/auction_pricing.py
├── models/                           # Serialized LightGBM models (.pkl)
├── SUDDDIVISIONE_BUDGET.md           # Analysis of budget allocation by role
├── LINK_BEFORE_STATS_COMES_OUT.py    # Utility to fetch upcoming-season FBref links
├── s_25_26rows.py                    # Generates blank rows for season s_25_26
└── various *.csv                     # Input data & final outputs
Installation
Clone the repo

bash
Copia
Modifica
git clone https://github.com/fumaghe/FataValuePrediction.git
cd FataValuePrediction
Create & activate a venv (optional but recommended)

bash
Copia
Modifica
python3 -m venv venv
source venv/bin/activate
Install dependencies

bash
Copia
Modifica
pip install -r requirements.txt
Usage
Data Retrieval
bash
Copia
Modifica
python data_retriever_fbref/stats_retriever_fbref.py
python data_retriever_fbref/remaining_columns.py
stats_retriever_fbref.py: Scrapes FBref Serie A pages, merges standard/shooting/passing stats.

remaining_columns.py: Merges Excel-sourced Fantacalcio metrics into the master CSV.

Data Preparation & Budget Analysis
data/ folder contains raw and interim CSV files.

SUDDDIVISIONE_BUDGET.md details optimal budget split by role (Portieri, Difensori, Centrocampisti, Attaccanti).

Training & Prediction
Run the full pipeline via the CLI:

bash
Copia
Modifica
python -m fantapred.cli \
  --csv data/giocatori_stagioni.csv \
  --train_until s_24_25 \
  --predict_season s_25_26 \
  [--targets all|core] \
  [--bonus_mode curve|linear|off] \
  [--round_stats] \
  [--log_file path/to/log]
Stages:

Aggregate mid-season transfers

Hierarchical imputation of missing stats

Feature building & future-season template

Minutes regression

Role-based LightGBM predictions

Post-processing & export → future_predictions_s_25_26.csv

Auction Pricing
Convert predictions into auction credits:

bash
Copia
Modifica
python -m fantapred.tools.auction_pricing \
  --pred_csv future_predictions_s_25_26.csv \
  --teams 10 \
  [--outfile auction_prices.csv]
Applies role credit pools, league difficulty, and availability adjustments.

Outputs a CSV of recommended bids per player.

Key Modules
data_retriever_fbref
stats_retriever_fbref.py: FBref scraper using cloudscraper + BeautifulSoup4.

remaining_columns.py: Two-pass merge with Excel inputs; handles transliteration & slug matching.

fantapred
cli.py: Orchestrates full pipeline.

data_processing.py:

aggregate_midseason_rows()

hierarchical_impute()

feature_engineering.py:

build_features()

build_future_dataframe()

modeling/:

minutes.py (predict minutes/starts/appearances)

lgb_optuna.py (Optuna-tuned LightGBM per role/target)

postprocessing.py (clipping, bonus adjustments)

models
Stores trained .pkl LightGBM models for each role and target.

Utilities & Scripts
LINK_BEFORE_STATS_COMES_OUT.py: Pre-season FBref link extractor.

s_25_26rows.py: Generates blank season-start rows for s_25_26.

Outputs
Merged Historical Data:

players_seasons_stats.csv → players_seasons_stats_updated.csv

giocatori_stagioni.csv, giocatori_stagioni_25_26.csv

Predictions:

future_predictions_s_<season>.csv

Auction Prices:

auction_prices.csv