
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matchday cross-club links + birthdays (Rich TUI) — versione con variabili interne

Esegui semplicemente:  python matchday_links_and_birthdays_config.py

Dipendenze: pandas, rich, python-dateutil
pip install pandas rich python-dateutil
"""
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Set, Optional

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.padding import Padding
from rich.text import Text
from rich import box

# =====================
# CONFIGURAZIONE QUI
# =====================
CSV_PATH = Path("PREZZOxFVM/giocatori_stagioni.csv")  # Percorso al CSV
SEASON = "s_25_26"                                   # Stagione
FIXTURES = [                                         # Partite (usa i valori di team_name_short)
    "Genoa-Lecce",
    "Sassuolo-Napoli",
    "Milan-Cremonese",
    "Roma-Bologna",
    "Cagliari-Fiorentina",
    "Como-Lazio",
    "Atalanta-Pisa",
    "Juventus-Parma",
    "Udinese-Verona",
    "Inter-Torino",
]
BDAY_RANGE = "08-20:08-26"                           # Intervallo compleanni MM-DD:MM-DD (o None per disattivare)
ALL_BIRTHDAYS = False                                # True: compleanni su tutte le squadre della season

# Se i nomi colonna differiscono dai default, modifica qui:
TEAM_COL = "team_name_short"
PLAYER_COL = "name"
SEASON_COL = "season"
DOB_COL = "date_of_birth"

console = Console()

# =====================
# CODICE
# =====================
def load_data(csv_path: Path,
              team_col: str,
              player_col: str,
              season_col: str,
              dob_col: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        console.print(f"[red]Errore nel leggere il CSV: {e}[/red]")
        raise

    missing = [c for c in [team_col, player_col, season_col] if c not in df.columns]
    if missing:
        console.print(f"[red]Colonne mancanti nel CSV: {missing}[/red]")
        raise RuntimeError(f"Colonne mancanti: {missing}")

    # Normalize strings
    df[player_col] = df[player_col].astype(str).str.strip()
    df[team_col] = df[team_col].astype(str).str.strip()
    df[season_col] = df[season_col].astype(str).str.strip()

    # Parse birth dates if present
    if dob_col in df.columns:
        def _parse_dob(x):
            x = str(x).strip()
            for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%Y/%m/%d"):
                try:
                    return datetime.strptime(x, fmt).date()
                except Exception:
                    continue
            return pd.NaT
        df[dob_col] = df[dob_col].apply(_parse_dob)
    else:
        df[dob_col] = pd.NaT

    return df

def parse_fixture(s: str) -> Tuple[str, str]:
    sep = "-" if "-" in s else " "
    parts = [p.strip() for p in s.replace(" vs ", sep).replace(" VS ", sep).split(sep)]
    if len(parts) != 2:
        raise ValueError(f'Fixture "{s}" non valida. Usa "TeamA-TeamB".')
    return parts[0], parts[1]

def seasons_played(df: pd.DataFrame,
                   player: str,
                   team: str,
                   player_col: str,
                   team_col: str,
                   season_col: str,
                   exclude_season: Optional[str] = None) -> Set[str]:
    q = (df[player_col] == player) & (df[team_col] == team)
    if exclude_season is not None:
        q = q & (df[season_col] != exclude_season)
    return set(df.loc[q, season_col].dropna().unique())

def print_cross_links(df: pd.DataFrame,
                      season: str,
                      fixtures: List[Tuple[str, str]],
                      team_col: str,
                      player_col: str,
                      season_col: str):
    header = Text(f"Cross-Club Links — Stagione {season}", style="bold white on blue")
    console.print(Padding(Panel(header, expand=False), (1,0,1,0)))

    df_season = df[df[season_col] == season].copy()
    if df_season.empty:
        console.print(f"[red]Nessun dato per la stagione {season}[/red]")
        return

    for a, b in fixtures:
        title = Text(f"{a}  ↔  {b}", style="bold yellow")
        console.print(Panel(title, style="bright_black", box=box.ROUNDED))

        roster_a = sorted(set(df_season.loc[df_season[team_col] == a, player_col]))
        roster_b = sorted(set(df_season.loc[df_season[team_col] == b, player_col]))

        rows_ab = []
        for p in roster_a:
            seas = seasons_played(df, p, b, player_col, team_col, season_col, exclude_season=season)
            if seas:
                rows_ab.append((p, len(seas), ", ".join(sorted(seas))))
        rows_ba = []
        for p in roster_b:
            seas = seasons_played(df, p, a, player_col, team_col, season_col, exclude_season=season)
            if seas:
                rows_ba.append((p, len(seas), ", ".join(sorted(seas))))

        def render_table(caption: str, rows: list):
            t = Table(title=caption, box=box.SIMPLE_HEAVY, show_lines=False, header_style="bold cyan")
            t.add_column("Giocatore", style="white")
            t.add_column("Stagioni", justify="right")
            t.add_column("Dettaglio stagioni", overflow="fold")
            if rows:
                for r in sorted(rows, key=lambda x: (-x[1], x[0])):
                    t.add_row(r[0], str(r[1]), r[2])
            else:
                t.add_row("[dim]- Nessuno -[/dim]", "", "")
            console.print(t)

        render_table(f"[{a}] hanno giocato in passato nel [{b}]", rows_ab)
        render_table(f"[{b}] hanno giocato in passato nel [{a}]", rows_ba)

def month_day_tuple(md: str) -> Tuple[int, int]:
    mm, dd = md.split("-")
    return int(mm), int(dd)

def is_in_range(d: datetime.date, start_mmdd: Tuple[int,int], end_mmdd: Tuple[int,int]) -> bool:
    if pd.isna(d):
        return False
    m, day = d.month, d.day
    s_m, s_d = start_mmdd
    e_m, e_d = end_mmdd
    if s_m == e_m:
        return (m == s_m) and (s_d <= day <= e_d)
    if (s_m < e_m):
        return (m > s_m or (m == s_m and day >= s_d)) and (m < e_m or (m == e_m and day <= e_d))
    return (m > s_m or (m == s_m and day >= s_d)) or (m < e_m or (m == e_m and day <= e_d))

def print_birthdays(df: pd.DataFrame,
                    season: str,
                    fixtures: List[Tuple[str,str]],
                    team_col: str,
                    player_col: str,
                    season_col: str,
                    dob_col: str,
                    bday_range: Optional[str],
                    all_teams: bool):

    header = Text("Compleanni nel periodo", style="bold white on magenta")
    console.print(Padding(Panel(header, expand=False), (1,0,0,0)))

    df_season = df[df[season_col] == season].copy()
    if df_season.empty:
        console.print(f"[red]Nessun dato per la stagione {season}[/red]")
        return

    if bday_range is None:
        console.print("[yellow]Range compleanni non specificato. Imposta BDAY_RANGE = 'MM-DD:MM-DD'[/yellow]")
        return

    try:
        start, end = bday_range.split(":")
        start_mmdd = month_day_tuple(start)
        end_mmdd = month_day_tuple(end)
    except Exception:
        console.print(f"[red]Formato range non valido: {bday_range}. Usa MM-DD:MM-DD[/red]")
        return

    if all_teams:
        teams = sorted(set(df_season[team_col]))
    else:
        teams = sorted(set([t for pair in fixtures for t in pair]))

    df_season = df_season[df_season[team_col].isin(teams)].copy()

    bday_rows = []
    for _, row in df_season.iterrows():
        dob = row.get(dob_col, pd.NaT)
        if pd.isna(dob):
            continue
        if is_in_range(dob, start_mmdd, end_mmdd):
            bday_rows.append((
                row[player_col],
                row[team_col],
                f"{dob.month:02d}-{dob.day:02d}"
            ))

    seen = set()
    unique_rows = []
    for r in bday_rows:
        key = (r[0], r[1], r[2])
        if key not in seen:
            seen.add(key)
            unique_rows.append(r)

    t = Table(title=f"Compleanni tra {start} e {end}" + (" (tutte le squadre)" if all_teams else " (solo squadre dei match)"),
              box=box.SIMPLE_HEAVY, header_style="bold cyan")
    t.add_column("Giocatore", style="white")
    t.add_column("Squadra", style="green")
    t.add_column("Giorno", justify="center")

    if unique_rows:
        for name, team, mmdd in sorted(unique_rows, key=lambda x: (x[2], x[1], x[0])):
            t.add_row(name, team, mmdd)
    else:
        t.add_row("[dim]- Nessuno -[/dim]", "", "")

    console.print(t)

def prepare_fixtures_list(fixtures_raw: List[str]) -> List[Tuple[str, str]]:
    fixtures: List[Tuple[str,str]] = []
    for f in fixtures_raw:
        a, b = parse_fixture(f)
        fixtures.append((a, b))
    return fixtures

def main():
    df = load_data(CSV_PATH, TEAM_COL, PLAYER_COL, SEASON_COL, DOB_COL)
    fixtures = prepare_fixtures_list(FIXTURES)
    print_cross_links(df, SEASON, fixtures, TEAM_COL, PLAYER_COL, SEASON_COL)
    if BDAY_RANGE:
        print_birthdays(df, SEASON, fixtures, TEAM_COL, PLAYER_COL, SEASON_COL, DOB_COL, BDAY_RANGE, ALL_BIRTHDAYS)

if __name__ == "__main__":
    main()
