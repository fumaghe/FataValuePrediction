import re
import unicodedata
import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup, Comment
from typing import Optional, List, Dict

# =========================================
# Config
# =========================================
URL_WAGES_SERIE_A = "https://fbref.com/en/comps/11/2025-2026/wages/2025-2026-Serie-A-Wages"
OUTPUT_CSV = "serie_a_players.csv"
BIG5_LINKS_CSV = "big5links.csv"  # deve essere presente nella stessa cartella (se non c'è, i link mancanti restano vuoti)

# =========================================
# Utils
# =========================================
def slugify(name: str) -> str:
    """
    Normalizza una stringa rimuovendo accenti e caratteri diacritici,
    converte in minuscolo, sostituisce gli spazi con trattini e
    rimuove caratteri non alfanumerici/dash.
    """
    name = name.strip()
    normalized = unicodedata.normalize('NFKD', name)
    no_diacritics = ''.join(c for c in normalized if not unicodedata.combining(c))
    lowered = no_diacritics.lower()
    hyphenated = re.sub(r"\s+", "-", lowered)
    cleaned = re.sub(r"[^a-z0-9-]", "", hyphenated)
    cleaned = re.sub(r"-{2,}", "-", cleaned).strip("-")
    return cleaned


def extract_player_id_from_url(url: str) -> str:
    """
    Estrae il player_id dall'URL FBref.
    Esempio: https://fbref.com/en/players/29812be3/Gabriele-Artistico -> 29812be3
    """
    if not url:
        return ""
    m = re.search(r"/players/([^/]+)/", url)
    return m.group(1) if m else ""


def _first_table_with_player_cells(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """Restituisce la prima tabella che contiene celle con data-stat='player'."""
    for table in soup.find_all("table"):
        if table.find("td", {"data-stat": "player"}):
            return table
    return None


def _extract_table_from_comments(node: BeautifulSoup, desired_id: Optional[str] = None) -> Optional[BeautifulSoup]:
    """
    Cerca tabelle dentro commenti HTML (pattern comune su FBref).
    Se desired_id è dato, preferisce la tabella con quell'id; altrimenti prende la prima
    che contenga celle data-stat='player'.
    """
    for comment in node.find_all(string=lambda text: isinstance(text, Comment)):
        if "table" not in comment:
            continue
        comment_soup = BeautifulSoup(comment, "html.parser")
        table = None
        if desired_id:
            table = comment_soup.find("table", id=desired_id)
        if table is None:
            table = _first_table_with_player_cells(comment_soup)
        if table is not None:
            return table
    return None


def extract_player_wages_table(soup: BeautifulSoup) -> Optional[BeautifulSoup]:
    """
    Estrae la tabella dei 'Player Wages' da FBref.
    Prova nell'ordine:
      1) table#player_wages
      2) tabella dentro #all_player_wages (commentata o meno)
      3) qualsiasi tabella con celle data-stat='player' nell'intera pagina (in chiaro)
      4) qualsiasi tabella con celle data-stat='player' nei commenti dell'intera pagina
    """
    table = soup.find("table", id="player_wages")
    if table:
        return table

    container = soup.find(id="all_player_wages")
    if container:
        table = container.find("table", id="player_wages") or _first_table_with_player_cells(container)
        if table:
            return table
        table = _extract_table_from_comments(container, desired_id="player_wages")
        if table:
            return table

    table = _first_table_with_player_cells(soup)
    if table:
        return table

    table = _extract_table_from_comments(soup, desired_id="player_wages")
    if table:
        return table

    return None


def load_slug_to_link_map(csv_path: str) -> Dict[str, str]:
    """
    Carica il mapping slug -> link_player_page da un CSV del tipo:
      name,link_player_page,slug
    Restituisce un dizionario {slug: link_player_page}.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        return {}

    if "slug" not in df.columns and "name" in df.columns:
        df["slug"] = df["name"].astype(str).map(slugify)

    if "slug" not in df.columns or "link_player_page" not in df.columns:
        return {}

    df = df.dropna(subset=["slug"]).copy()
    df["slug"] = df["slug"].astype(str)
    mapping = (
        df[["slug", "link_player_page"]]
        .dropna(subset=["link_player_page"])
        .drop_duplicates(subset=["slug"])
        .set_index("slug")["link_player_page"]
        .to_dict()
    )
    return mapping

# =========================================
# Core
# =========================================
def fetch_wages_players(url: str) -> List[dict]:
    """
    Scarica la pagina 'Player Wages' e ritorna una lista di dict:
      { 'name': ..., 'link_player_page': (str|''), 'slug': ..., 'player_id': (str|'') }
    """
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    table = extract_player_wages_table(soup)
    if table is None:
        raise ValueError("Tabella 'Player Wages' non trovata nella pagina.")

    players = []
    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        if row.get("class") and ("thead" in row.get("class") or "over_header" in row.get("class")):
            continue

        cell = row.find("td", {"data-stat": "player"})
        if not cell:
            continue

        link = cell.find("a")
        if link and link.text.strip():
            name = link.text.strip()
            href = link.get("href", "").strip()
            link_player_page = ("https://fbref.com" + href) if href.startswith("/") else href
        else:
            name = cell.get_text(strip=True)
            link_player_page = ""

        if not name:
            continue

        player_id = extract_player_id_from_url(link_player_page)
        players.append(
            {
                "name": name,
                "link_player_page": link_player_page,
                "slug": slugify(name),
                "player_id": player_id,
            }
        )

    # rimuovi eventuali duplicati per slug mantenendo il primo
    seen = set()
    unique_players = []
    for p in players:
        if p["slug"] in seen:
            continue
        seen.add(p["slug"])
        unique_players.append(p)

    return unique_players


def enrich_missing_links_and_ids(players: List[dict], slug_to_link: Dict[str, str]) -> List[dict]:
    """
    Per i giocatori senza link, prova a recuperarli usando il mapping slug->link.
    Compila anche player_id quando possibile.
    """
    for p in players:
        if not p.get("link_player_page"):
            link = slug_to_link.get(p["slug"])
            if link:
                p["link_player_page"] = link
        # in ogni caso, se manca player_id ma c'è link, prova a estrarlo
        if not p.get("player_id"):
            p["player_id"] = extract_player_id_from_url(p.get("link_player_page", ""))
    return players


def main():
    try:
        players = fetch_wages_players(URL_WAGES_SERIE_A)
        slug_to_link = load_slug_to_link_map(BIG5_LINKS_CSV)
        players = enrich_missing_links_and_ids(players, slug_to_link)

        df = pd.DataFrame(players, columns=["name", "link_player_page", "slug", "player_id"])
        df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"Salvati {len(df)} giocatori in '{OUTPUT_CSV}'.")
        missing_links = (df["link_player_page"] == "").sum()
        missing_ids = (df["player_id"] == "").sum()
        if missing_links:
            print(f"Nota: {missing_links} giocatori senza link_player_page.")
        if missing_ids:
            print(f"Nota: {missing_ids} giocatori senza player_id.")
    except Exception as e:
        print(f"Errore: {e}")


if __name__ == "__main__":
    main()
