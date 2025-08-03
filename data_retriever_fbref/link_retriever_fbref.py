import pandas as pd
import unicodedata
import cloudscraper
from bs4 import BeautifulSoup, Comment

# URL della pagina delle statistiche di Serie A su FBref
URL_SERIE_A = 'https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats'


def slugify(name: str) -> str:
    """
    Normalizza una stringa rimuovendo accenti e caratteri diacritici,
    converte in minuscolo e sostituisce spazi con trattini.
    """
    normalized = unicodedata.normalize('NFKD', name)
    no_diacritics = ''.join(c for c in normalized if not unicodedata.combining(c))
    return no_diacritics.lower().replace(' ', '-')


def extract_stats_standard_table(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Estrae la tabella 'stats_standard' anche se contenuta in commenti HTML.
    """
    # Prova a trovare direttamente
    table = soup.find('table', id='stats_standard')
    if table:
        return table

    # Altrimenti cerca nei commenti
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        if 'table' in comment and 'stats_standard' in comment:
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', id='stats_standard')
            if table:
                return table
    return None


def fetch_serie_a_players() -> pd.DataFrame:
    """
    Utilizza cloudscraper per aggirare protezioni Cloudflare e ottenere
    la lista di giocatori con name, link_player_page e slug.
    """
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(URL_SERIE_A)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, 'html.parser')
    table = extract_stats_standard_table(soup)
    if table is None:
        raise ValueError('Tabella "stats_standard" non trovata.')

    players = []
    for row in table.find('tbody').find_all('tr'):
        cell = row.find('td', {'data-stat': 'player'})
        if not cell:
            continue
        link = cell.find('a')
        if not link:
            continue

        name = link.text.strip()
        href = link['href']
        link_player_page = 'https://fbref.com' + href
        slug = slugify(name)

        players.append({
            'name': name,
            'link_player_page': link_player_page,
            'slug': slug
        })

    return pd.DataFrame(players)


def main():
    try:
        df = fetch_serie_a_players()
        df.to_csv('serie_a_players.csv', index=False, encoding='utf-8')
        print(f"Salvati {len(df)} giocatori in 'serie_a_players.csv'.")
    except Exception as e:
        print(f"Errore: {e}")


if __name__ == '__main__':
    main()
