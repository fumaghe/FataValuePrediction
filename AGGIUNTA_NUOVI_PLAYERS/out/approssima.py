import pandas as pd

# Lista colonne da arrotondare
cols_to_round = ["fmv_pred", "mv_pred", "gf_pred", "assist_pred", "clean_sheet_pred"]

# Funzione per arrotondare le colonne se esistono
def round_columns_in_csv(file_path):
    df = pd.read_csv(file_path)

    for col in cols_to_round:
        if col in df.columns:
            df[col] = df[col].round(2)

    df.to_csv(file_path, index=False)
    print(f"✅ File aggiornato: {file_path}")

if __name__ == "__main__":
    round_columns_in_csv("future_predictions_s_25_26_updated.csv")
    round_columns_in_csv("auction_prices_updated.csv")
