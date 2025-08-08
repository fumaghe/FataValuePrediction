import os
import json
import pandas as pd

# === CONFIG ===
FUTURE_PATH = 'future_predictions_s_25_26.csv'
GIOCATORI_PATH = 'data/giocatori_stagioni.csv'
JSON_PATH = 'fantapred/tools/custom_adjustments.json'  # <-- richiesto

# Tornei invariati per i default (percentuali)
unchanged = ['Serie A', 'Premier League', 'La Liga', 'Bundesliga', 'Ligue 1']

# Default adjustments (percentuali, es. -0.40 = -40%)
default_adjustments = {
    'mv_pred': -0.02,
    'fmv_pred': -0.02,
    'gf_pred': -0.40,
    'assist_pred': -0.40
}

# Colonne che potremmo toccare
maybe_numeric = [
    'presenze_pred', 'starts_pred', 'mv_pred', 'fmv_pred',
    'gf_pred', 'assist_pred', 'clean_sheet_pred'
]

# === LOAD DATA ===
future = pd.read_csv(FUTURE_PATH)
giocatori = pd.read_csv(GIOCATORI_PATH)

giocatori_24_25 = giocatori.loc[
    giocatori['season'] == 's_24_25',
    ['slug', 'tournament_name']
]

df = future.merge(giocatori_24_25, on='slug', how='left')
df.drop_duplicates(subset='slug', keep='first', inplace=True)

# Cast numerico sulle colonne che tocchiamo
for col in maybe_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# === LOAD CUSTOM JSON (tutto percentuale) ===
if not os.path.exists(JSON_PATH):
    raise FileNotFoundError(f"File JSON non trovato: {JSON_PATH}")

with open(JSON_PATH, 'r', encoding='utf-8') as f:
    custom_adjustments = json.load(f)

# Normalizza valori JSON a float
for slug, stats in list(custom_adjustments.items()):
    for k, v in list(stats.items()):
        try:
            stats[k] = float(v)
        except Exception:
            del stats[k]

json_slugs = set(custom_adjustments.keys())
df_slugs = set(df['slug'].astype(str))
missing_from_df = sorted(list(json_slugs - df_slugs))
if missing_from_df:
    print(f"⚠️ {len(missing_from_df)} slug nel JSON non trovati nel CSV (prime 10): {missing_from_df[:10]}")

# Salva copia originale per diff
orig = df.copy()

# === DEFAULT ADJUSTMENTS (percentuali) dove NON c'è custom e torneo NON è in 'unchanged' ===
has_custom_for_slug = df['slug'].isin(list(custom_adjustments.keys()))
mask_default = (~df['tournament_name'].isin(unchanged)) & (~has_custom_for_slug)

for stat, pct in default_adjustments.items():
    if stat in df.columns:
        df.loc[mask_default, stat] = df.loc[mask_default, stat] * (1 + pct)

# === CUSTOM ADJUSTMENTS (sempre) — TUTTO in percentuale ===
applied_rows = set()
for slug, stats in custom_adjustments.items():
    mask = (df['slug'] == slug)
    if not mask.any():
        continue
    applied_rows.add(slug)
    for stat, pct in stats.items():
        if stat in df.columns and pd.api.types.is_numeric_dtype(df[stat]):
            df.loc[mask, stat] = df.loc[mask, stat] * (1 + pct)

# Arrotonda e clip per presenze/starts
for col in ['presenze_pred', 'starts_pred']:
    if col in df.columns:
        df[col] = df[col].round().clip(lower=0).astype('Int64')

# Rimuovi la colonna torneo prima di salvare
if 'tournament_name' in df.columns:
    df.drop(columns=['tournament_name'], inplace=True)

if 'presenze_pred' in df.columns:
    df['presenze_pred.1'] = df['presenze_pred'].astype('Int64')
# === Salva file finale ===
out_path = 'future_predictions_s_25_26_adjusted.csv'
df.to_csv(out_path, index=False)

print(f"✅ Completato. Slug con custom applicato: {len(applied_rows)} | Output: '{out_path}'")