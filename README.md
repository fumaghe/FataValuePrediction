# FataValuePrediction

A complete pipeline for scraping, cleaning and merging **Fantacalcio** (Italian fantasy football) statistics, training LightGBM models to forecast next-season performance, and converting those forecasts into recommended auction prices.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Features](#features)  
- [Directory Structure](#directory-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Data Retrieval](#data-retrieval)  
  - [Data Preparation & Budget Analysis](#data-preparation--budget-analysis)  
  - [Training & Prediction](#training--prediction)  
  - [Auction Pricing](#auction-pricing)  
- [Key Modules](#key-modules)  
  - [data_retriever_fbref](#data_retriever_fbref)  
  - [fantapred](#fantapred)  
  - [models](#models)  
  - [Utilities & Scripts](#utilities--scripts)  
- [Outputs](#outputs)  
- [Contributing](#contributing)  
- [License](#license)  

---

## Project Overview

This repository automates the end-to-end workflow for Fantacalcio value prediction:

1. **Scrape** detailed per-season player stats from FBref.  
2. **Enrich** with additional Fantacalcio data from Excel sheets.  
3. **Clean & Impute** missing values using hierarchical strategies.  
4. **Engineer Features** (per-90 metrics, age decay, team strength).  
5. **Train Models** (LightGBM + Optuna tuning) to predict goals, assists, ratings, clean sheets, and monetary value.  
6. **Post-process** predictions and **convert** them into auction-style price recommendations.

---

## Features

- Robust scraping of FBref with Cloudflare-aware retries  
- Two-pass merging of Excel-based Fantacalcio stats  
- Hierarchical imputation (player, team-role, role levels + iterative imputer)  
- Feature engineering: age curves, league/coaching coefficients, attack strength  
- Optuna-tuned LightGBM models per role & target  
- Auction pricing tool with role credit pools, difficulty adjustments, free-slot minimums  

---

## Directory Structure

```
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
├── SUDDDIVISIONE_BUDGET.md           # Budget allocation analysis by role
├── LINK_BEFORE_STATS_COMES_OUT.py    # Upcoming-season FBref link extractor
├── s_25_26rows.py                    # Generates blank rows for season s_25_26
└── various *.csv                     # Input data & final outputs
```

---

## Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/fumaghe/FataValuePrediction.git
   cd FataValuePrediction
   ```

2. (Optional) Create and activate a virtual environment  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### Data Retrieval

```bash
python data_retriever_fbref/stats_retriever_fbref.py
python data_retriever_fbref/remaining_columns.py
```

- **stats_retriever_fbref.py**: scrapes FBref Serie A pages and merges standard, shooting, and passing stats.  
- **remaining_columns.py**: merges Fantacalcio metrics from Excel files into the master CSV.

### Data Preparation & Budget Analysis

- The `data/` folder contains raw and interim CSV files.  
- `SUDDDIVISIONE_BUDGET.md` provides an analysis of optimal budget allocation by role (Goalkeepers, Defenders, Midfielders, Forwards).

### Training & Prediction

```bash
python -m fantapred.cli \
  --csv data/giocatori_stagioni.csv \
  --train_until s_24_25 \
  --predict_season s_25_26 \
  [--targets all|core] \
  [--bonus_mode curve|linear|off] \
  [--round_stats] \
  [--log_file path/to/log]
```

Pipeline stages:

1. Aggregate mid-season transfers  
2. Hierarchical imputation of missing stats  
3. Feature construction & future-season template  
4. Minutes regression  
5. Role-based LightGBM predictions  
6. Post-processing & export → `future_predictions_s_25_26.csv`

### Auction Pricing

```bash
python -m fantapred.tools.auction_pricing \
  --pred_csv future_predictions_s_25_26.csv \
  --teams 10 \
  [--outfile auction_prices.csv]
```

- Applies role credit pools, league difficulty, and availability adjustments.  
- Outputs recommended bid credits per player.

---

## Key Modules

### data_retriever_fbref

- `stats_retriever_fbref.py`: uses `cloudscraper` and `BeautifulSoup4` to fetch and merge FBref stats.  
- `remaining_columns.py`: two-pass merge of Excel data, handles transliteration, accent stripping, and slug matching.

### fantapred

- `cli.py`: main entry point for the full pipeline.  
- `data_processing.py`:  
  - `aggregate_midseason_rows()`  
  - `hierarchical_impute()`  
- `feature_engineering.py`:  
  - `build_features()`  
  - `build_future_dataframe()`  
- `modeling/`:  
  - `minutes.py` – predict minutes, starts, appearances  
  - `lgb_optuna.py` – Optuna-tuned LightGBM per role/target  
  - `postprocessing.py` – clipping, bonus adjustments, default ratings

### models

- Stores trained LightGBM `.pkl` models for reuse.

### Utilities & Scripts

- `LINK_BEFORE_STATS_COMES_OUT.py`: pre-season FBref link extractor.  
- `s_25_26rows.py`: generates blank season-start rows for `s_25_26`.

---

## Outputs

- **Merged Historical Data**:  
  - `players_seasons_stats.csv` → `players_seasons_stats_updated.csv`  
  - `giocatori_stagioni.csv`, `giocatori_stagioni_25_26.csv`  
- **Predictions**:  
  - `future_predictions_s_<season>.csv`  
- **Auction Prices**:  
  - `auction_prices.csv`  

---

## Contributing

1. Fork the repository  
2. Create a feature branch  
   ```bash
   git checkout -b feature/YourFeature
   ```  
3. Commit your changes  
   ```bash
   git commit -m "Add feature"
   ```  
4. Push and open a Pull Request

Please follow PEP 8 for Python code and include tests for new functionality.

---

## License

This project is provided “as-is.” Refer to repository headers or contact the author for licensing details.
