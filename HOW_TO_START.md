Prima cosa da fare far partire il file "data_retriever_fbref\stats_retriever_fbref.py"

Seconda cosa far partire il file "data_retriever_fbref\remaining_columns.py"

Copiare il contenuto di "data_retriever_fbref\players_seasons_stats_updated.csv" in "data\giocatori_stagioni.csv"

## Cosi avremo recuperato tutte le statistiche e siamo pronti per la prediction


Per fare partire la prediction nella bash runnare
python -m fantapred.cli --csv data/giocatori_stagioni.csv --train_until s_23_24 --predict_season s_24_25
python -m fantapred.cli --csv data/giocatori_stagioni.csv --train_until s_24_25 --predict_season s_25_26 

# Ricordare di eliminare i modelli

Le predictions verranno salvate in future_predictions_s_25_26.csv

Per fare partire il calcolo dei valori runnare 

python -m fantapred.tools.auction_pricing --pred_csv future_predictions_s_25_26.csv --teams 10

I valori verranno salvati in auction_prices.csv
