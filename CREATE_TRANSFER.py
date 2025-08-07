import pandas as pd
import unicodedata
from pathlib import Path

def normalize_name(s: str) -> str:
    s = str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    for ch in ["'", "’", "-", ".", ","]:
        s = s.replace(ch, " ")
    return " ".join(s.split())

# Input
players = pd.read_csv("serie_a_players.csv")
mapping = pd.read_csv("data/giocatori_stagioni.csv")
stats = pd.read_excel("data_retriever_fbref/Statistiche_Fantacalcio_Stagione_2025_26.xlsx", header=1)

# Role da Excel (colonna R)
stats_small = stats.loc[:, ["Nome", "R"]].rename(columns={"Nome": "name", "R": "role_new"})
mapping["name_key"] = mapping["name"].astype(str).map(normalize_name)
stats_small["name_key"] = stats_small["name"].astype(str).map(normalize_name)

role_by_slug = (
    stats_small
    .merge(mapping[["name", "slug", "name_key"]], on="name_key", how="left")
    .loc[:, ["slug", "role_new"]]
    .drop_duplicates()
)

# Dati “nuovi” (partendo da players) + filtro: SOLO slug presenti in giocatori_stagioni
new_df = players.merge(role_by_slug, on="slug", how="left")
new_df = new_df[new_df["slug"].isin(mapping["slug"].unique())].copy()

new_df["team_name_short_new"] = new_df["team"]
new_df["minutes_hint"] = ""
new_df["starter_hint"] = ""

# Ultima riga per slug nella stagione s_24_25
last_s_24_25 = (
    mapping[mapping["season"] == "s_24_25"]
    .groupby("slug", as_index=False)
    .tail(1)
)
old = last_s_24_25[["slug", "team_name_short", "role"]].rename(columns={
    "team_name_short": "team_name_short_old",
    "role": "role_old"
})

# Confronto SOLO per chi ha baseline s_24_25: inner merge
cur = new_df.merge(old, on="slug", how="inner")

def norm_or_none(x):
    if pd.isna(x):
        return None
    return str(x).strip()

cur["team_change"] = cur.apply(
    lambda r: norm_or_none(r["team_name_short_new"]) != norm_or_none(r["team_name_short_old"]),
    axis=1
)
cur["role_change"] = cur.apply(
    lambda r: (
        (r["role_new"] is not None and not pd.isna(r["role_new"]))
        and (norm_or_none(r["role_new"]) != norm_or_none(r["role_old"]))
    ),
    axis=1
)

changed = cur[(cur["team_change"]) | (cur["role_change"])].copy()

final_cols = [
    "player_id", "slug", "team_name_short_new", "tournament_name_new",
    "role_new", "minutes_hint", "starter_hint"
]
out_df = changed[final_cols].copy()

out_path = Path("TRANSFERS.csv")
out_df.to_csv(out_path, index=False)
print(f"Creato: {out_path.resolve()}, righe: {len(out_df)}")
