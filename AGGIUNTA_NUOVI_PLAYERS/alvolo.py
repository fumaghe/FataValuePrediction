import pandas as pd

# file di input/output
path = "giocatori_stagioni_updated.csv"
out_path = "giocatori_stagioni_updated.csv"

# leggi csv
df = pd.read_csv(path)

# lista delle squadre di Serie A (puoi ampliarla se ne mancano)
serie_a_teams = [
    "Atalanta", "Bologna", "Cagliari", "Como", "Empoli", "Fiorentina",
    "Genoa", "Inter", "Juventus", "Lazio", "Lecce", "Milan",
    "Monza", "Napoli", "Parma", "Roma", "Torino", "Udinese", "Venezia", "Verona"
]

# applica aggiornamento solo per la stagione s_25_26
mask = (df["season"] == "s_25_26") & (df["team_name_short"].isin(serie_a_teams))
df.loc[mask, "tournament_name"] = "Serie A"
df.loc[mask, "tournament_country"] = "ITA"

# salva
df.to_csv(out_path, index=False)

print(f"Aggiornati {mask.sum()} giocatori a Serie A -> {out_path}")
