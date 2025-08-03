# coding: utf-8
"""
FBref → CSV stagioni giocatori (layout Fantacalcio)
Tabelle lette:
   ▸ stats_standard_dom_lg
   ▸ stats_shooting_dom_lg
   ▸ stats_passing_dom_lg
Output: players_seasons_stats.csv
"""

import re, time, random, unicodedata
from pathlib import Path
from typing  import Dict, Tuple, List

import pandas as pd
import cloudscraper
from bs4 import BeautifulSoup, Comment
from requests.exceptions import HTTPError


# ───────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────
BASE_URL   = "https://fbref.com"
COMP_URL   = f"{BASE_URL}/en/comps/11/stats/Serie-A-Stats"
OUT_FILE   = "players_seasons_stats.csv"
PLAYERS_CSV= "serie_a_players.csv"

USER_AGENTS = [
    # 20 real UA (Chrome, FF, Safari desktop/mobile)
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 8 Pro) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/126.0.0.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPad; CPU OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0",
    # backup UA
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 13; SM-S928B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_7_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Safari/605.1.15",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.4 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (X11; Fedora; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Windows NT 10.0; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/125.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Linux; Android 14; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 11_7_10) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (iPad; CPU OS 16_7 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/124.0.0.0 Mobile/15E148 Safari/604.1",
    "Mozilla/5.0 (Windows NT 6.3; WOW64; rv:109.0) Gecko/20100101 Firefox/109.0",
]

# colonne d’esempio + slug + link + image
BASE_COLS = """player_id,name,team_name_short,season,tournament_name,tournament_country,role,
presenze,assist,mv,fmv,gf,gs,rp,rc,r_plus,r_minus,amm,esp,au,starts_eleven,
min_playing_time,injured,total_duels_won_percentage,total_duels_won,
possession_lost,total_shots,shots_on_target,goal_conversion_percentage,free_kick_shots,
penalty_won,successful_dribbles,successful_dribbles_percentage,
touches,key_passes,accurate_passes_percentage,total_passes,accurate_passes,accurate_crosses,total_cross,
accurate_final_third_passes,accurate_opposition_half_passes,tackles,
tackles_won_percentage,tackles_won,interceptions,fouls,error_lead_to_goal,
penalty_conceded,clean_sheet,saves,
penalty_faced,big_chances_created,
pass_to_assist,accurate_chipped_passes,accurate_long_balls,
accurate_long_balls_percentage,aerial_duels_won,
aerial_duels_won_percentage,aerial_lost,attempt_penalty_miss,
attempt_penalty_target,big_chances_create,blocked_shots,
clearances,crosses_not_claimed,dispossessed,duel_lost,error_lead_to_shot,expected_goals,
hit_woodwork,
inaccurate_passes,offsides,penalty_conversion,
possession_won_att_third,punches,scoring_frequency,shot_from_set_piece,shots_off_target,
successful_runs_out,total_attempt_assist,total_chipped_passes,
total_long_balls,total_opposition_half_passes,total_own_half_passes,
totw_appearances,was_fouled,last_active_year,team_id,aff_index,inf_index,
tit_index,spesa_media_altri,fvm,quotazione,date_of_birth,height,jersey_number,
preferred_foot,country,slug""".replace("\n", "").split(",")

FULL_COLS = BASE_COLS + ["link", "image"]


# ───────────────────────────────────────────────────
# UTILS
# ───────────────────────────────────────────────────
# Istanza unica di scraper, con UA fisso e cookie persistenti
UA = random.choice(USER_AGENTS)
SCRAPER = cloudscraper.create_scraper(
    browser={"custom": UA},
    interpreter="nodejs"
)
SCRAPER.headers.update({
    "User-Agent": UA,
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE_URL
})

def new_scraper() -> cloudscraper.CloudScraper:
    ua = random.choice(USER_AGENTS)
    s = cloudscraper.create_scraper(browser={"custom": ua}, interpreter="nodejs")
    s.headers.update({
        "User-Agent": ua,
        "Accept-Language": random.choice(["en-US,en;q=0.9", "it-IT,it;q=0.9,en-US;q=0.8"]),
        "Referer": BASE_URL,
        "Cache-Control": "no-cache",
    })
    return s

SCRAPER = new_scraper()

_CF_BLOCK_MARKERS = (
    "Just a moment", "Attention Required", "/cdn-cgi/challenge-platform", "cf-browser-verification"
)

def _looks_blocked(html: str) -> bool:
    if not html:
        return False
    h = html[:5000]
    return any(m in h for m in _CF_BLOCK_MARKERS)

def fetch(url, timeout=25, retries=12, backoff=2.0, jitter=0.35):
    """
    Fetch resiliente:
    - retry su 429/403/5xx e su errori di rete
    - rispetta Retry-After
    - ruota sessione/UA ogni 3 tentativi o se rileva pagina di blocco
    - backoff esponenziale + jitter
    """
    global SCRAPER
    delay = 1.5
    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            r = SCRAPER.get(url, timeout=timeout)
            status = r.status_code

            # OK e non pagina di challenge
            if status == 200 and not _looks_blocked(r.text):
                return r.text

            # Rate limit o challenge
            if status in (429, 403, 503) or _looks_blocked(r.text):
                ra = r.headers.get("Retry-After")
                wait = float(ra) if ra and ra.isdigit() else delay
                wait += random.uniform(0, wait * jitter)

                # ruota sessione/UA ogni 3 tentativi o se bloccato
                if attempt % 3 == 0 or _looks_blocked(r.text):
                    SCRAPER = new_scraper()

                time.sleep(wait)
                delay *= backoff
                continue

            # 5xx generici
            if 500 <= status < 600:
                time.sleep(delay + random.uniform(0, delay * jitter))
                delay *= backoff
                continue

            # altri status → errore immediato (404 ecc.)
            r.raise_for_status()

        except Exception as e:
            # errori di rete, timeout, SSL, chunked ecc.
            last_exc = e
            if attempt % 3 == 0:
                SCRAPER = new_scraper()
            time.sleep(delay + random.uniform(0, delay * jitter))
            delay *= backoff
            continue

    raise HTTPError(f"Unable to fetch {url} after {retries} retries; last error: {last_exc}")


def slugify(txt: str) -> str:
    # mantiene i trattini tra i token (→ evita “juventusfc”)
    norm = unicodedata.normalize("NFKD", txt)
    norm = re.sub(r"\s+", "-", norm.strip())
    norm = re.sub(r"[^\w\-]", "", norm, flags=re.UNICODE)
    norm = re.sub(r"-{2,}", "-", norm)
    return norm.lower()


def clean_num(txt):
    return txt.replace(",", "").strip() if txt else ""


def extract_table(soup, table_id):
    t = soup.find("table", id=table_id)
    if t:
        return t
    # FBref spesso mette le tabelle in commento HTML
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in c:
            return BeautifulSoup(c, "html.parser").find("table", id=table_id)
    return None


def season_code(yr):
    y1, y2 = yr.split("-")
    return f"s_{y1[-2:]}_{y2[-2:]}"


def parse_country(raw):
    return raw.strip()[-3:].upper() if raw else ""


# ───────────────────────────────────────────────────
# LISTA GIOCATORI
# ───────────────────────────────────────────────────
def get_players(url):
    soup = BeautifulSoup(fetch(url), "html.parser")
    table = extract_table(soup, "stats_standard")
    players = []
    for tr in table.select("tbody tr"):
        cell = tr.find("td", {"data-stat": "player"})
        if cell and cell.a:
            href = cell.a["href"]
            players.append(
                {
                    "player_id": href.split("/")[3],
                    "name": cell.get_text(strip=True),
                    "slug": slugify(cell.get_text(strip=True)),
                    "link": BASE_URL + href,
                }
            )
    pd.DataFrame(players).to_csv(PLAYERS_CSV, index=False)
    return players


# ───────────────────────────────────────────────────
# BIO
# ───────────────────────────────────────────────────
def parse_bio(soup):
    bio = {}

    # ── Immagine
    img_tag = soup.select_one("div.media-item img")
    if img_tag and img_tag.has_attr("src"):
        bio["image"] = img_tag["src"]
    else:
        og = soup.find("meta", property="og:image")
        bio["image"] = og["content"] if og and og.has_attr("content") else ""

    # ── Position → role e Footed → preferred_foot
    pos_tag = soup.find("strong", string=re.compile(r"Position:"))
    role = ""
    foot = ""
    if pos_tag:
        p_text = pos_tag.parent.get_text(" ", strip=True)
        # estraggo la parte dopo "Position:" e prima di "▪" o "("
        m_role = re.search(r"Position:\s*([^▪\(]+)", p_text)
        if m_role:
            raw = m_role.group(1).strip()         # es. "FW-MF" o "DF "
            mapping = {"G": "P", "D": "D", "M": "C", "F": "A"}
            for ch in raw:
                if ch.upper() in mapping:
                    role = mapping[ch.upper()]
                    break
        # cerco left/right
        m_foot = re.search(r"Footed:\s*(Left|Right)", p_text, re.I)
        if m_foot:
            foot = m_foot.group(1).lower()
        else:
            # fallback: dentro parentesi
            m_fb = re.search(r"\b(left|right)\b", p_text, re.I)
            if m_fb:
                foot = m_fb.group(1).lower()

    bio["role"] = role
    bio["preferred_foot"] = foot

    # ── Altezza
    h = soup.find(string=re.compile(r"\d{2,3}cm"))
    if h:
        m = re.search(r"(\d{2,3})cm", h)
        bio["height"] = m.group(1) if m else ""
    else:
        bio["height"] = ""

    # ── Data di nascita
    birth = soup.find("span", id="necro-birth")
    bio["date_of_birth"] = birth["data-birth"] if birth and birth.has_attr("data-birth") else ""

    # ── Nazione
    nt = soup.select_one('p:has(strong:contains("National Team")) a')
    bio["country"] = nt.get_text(strip=True).lower() if nt else ""

    return bio


# ───────────────────────────────────────────────────
# PARSER TABELLE (chiave = season, team)
# ───────────────────────────────────────────────────
Key = Tuple[str, str]            # (season_code, team_short)


# ---------- 1. STANDARD
def parse_standard(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_standard_dom_lg")
    data = {}
    if not tbl:
        return data

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc   = season_code(yr)
        team = tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key  = (sc, team)

        rec = {c: "" for c in FULL_COLS}
        rec.update({"season": sc, "team_name_short": team})
        comp_cell = tr.find("td", {"data-stat": "comp_level"}) or tr.find("td", {"data-stat": "comp"})
        rec["tournament_name"]    = re.sub(r"^\d+\.", "", comp_cell.get_text(strip=True))
        rec["tournament_country"] = parse_country(tr.find("td", {"data-stat": "country"}).get_text())

        m = {
            "presenze":         "games",
            "starts_eleven":    "games_starts",
            "assist":           "assists",
            "mv":               "minutes",
            "fmv":              "minutes_90s",
            "gf":               "goals",
            "amm":              "cards_yellow",
            "esp":              "cards_red",
            "min_playing_time": "minutes",
        }
        for out_col, stat in m.items():
            td = tr.find("td", {"data-stat": stat})
            rec[out_col] = clean_num(td.get_text()) if td else ""

        # totw_appearances ← MP
        td_mp = tr.find("td", {"data-stat": "games"})
        rec["totw_appearances"] = clean_num(td_mp.get_text()) if td_mp else ""

        # rc ← rigori calciati = pens_made
        td_rc = tr.find("td", {"data-stat": "pens_made"})
        rec["rc"] = clean_num(td_rc.get_text()) if td_rc else ""

        data[key] = rec

    return data


# ---------- 2. SHOOTING
def parse_shooting(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_shooting_dom_lg")
    out = {}
    if not tbl:
        return out

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        g = lambda k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""
        sh             = g("shots")
        sot            = g("shots_on_target")
        gls            = g("goals")
        conv          = f"{100*float(gls)/float(sh):.2f}" if sh and sh != "0" else ""

        # calcola shots_off_target = total_shots - shots_on_target
        try:
            off_target = str(int(sh) - int(sot))
        except:
            off_target = ""

        out[key] = {
            "total_shots":                 sh,
            "shots_on_target":             sot,
            "shots_off_target":            off_target,
            "goal_conversion_percentage":  conv,
            "free_kick_shots":             g("shots_free_kicks"),
            "attempt_penalty_target":      g("pens_att"),
            "penalty_conversion":          g("pens_made"),
            "expected_goals":              g("xg"),
            # nuove metriche
            "attempt_penalty_miss":        g("pens_missed"),
            "hit_woodwork":                g("woodwork"),
            "shot_from_set_piece":         g("shots_set_pieces"),
            "scoring_frequency":           g("goals_per_shot"),
        }

    return out

# ---------- 3. PASSING
def parse_passing(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_passing_dom_lg")
    out = {}
    if not tbl:
        return out

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        g = lambda k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""
        total = g("passes")
        comp  = g("passes_completed")

        # inaccurate_passes = total - comp
        try:
            inacc = str(int(total) - int(comp))
        except:
            inacc = ""

        out[key] = {
            "accurate_passes":                comp,
            "total_passes":                   total,
            "accurate_passes_percentage":     g("passes_pct"),
            "key_passes":                     g("assisted_shots"),
            "accurate_long_balls":            g("passes_completed_long"),
            "total_long_balls":               g("passes_long"),
            "accurate_long_balls_percentage": g("passes_pct_long"),
            "accurate_final_third_passes":    g("passes_into_final_third"),
            "accurate_opposition_half_passes":g("passes_into_penalty_area"),
            "accurate_chipped_passes":        g("crosses_into_penalty_area"),
            "assist":                         g("assists") or "",
            "total_own_half_passes":          g("passes_own_half"),
            "total_opposition_half_passes":   g("passes_opposition_half"),
            "inaccurate_passes":              inacc,
        }

    return out


# ---------- 4a. PASS TYPES
def parse_pass_types(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_passing_types_dom_lg")
    out = {}
    if not tbl:
        return out

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        g = lambda k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""

        out[key] = {
            # già esistenti
            "total_cross":            g("crosses"),
            "offsides":               g("passes_offsides"),
            # nuove metriche
            "total_chipped_passes":   g("passes_chip"),
            "accurate_crosses":       g("crosses_into_penalty_area"),
        }
    return out



# ---------- 4b. GCA / SCA
def parse_gca(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_gca_dom_lg")
    out = {}
    if not tbl:
        return out

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        g = lambda k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""

        # p2a = pass to assist (live + dead)
        p2a = ""
        live = g("gca_passes_live") or "0"
        dead = g("gca_passes_dead") or "0"
        if live or dead:
            p2a = str(int(live) + int(dead))

        out[key] = {
            # già esistenti
            "big_chances_created":    g("gca"),
            "total_attempt_assist":   g("sca"),
            "pass_to_assist":         p2a,
            # nuova metrica
            "big_chances_create":     g("sca"),
        }
    return out


# ---------- 5a. DEFENSIVE (rivisto secondo le nuove definizioni di duels)
def parse_defensive(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_defense_dom_lg")
    out = {}
    if not tbl:
        return out

    # helper per leggere e pulire
    get = lambda tr, k: clean_num(tr.find("td", {"data-stat": k}).get_text()) \
                       if tr.find("td", {"data-stat": k}) else ""

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue

        sc   = season_code(yr)
        team = tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key  = (sc, team)

        # estraggo le metriche raw
        tkl          = get(tr, "tackles")               # Tkl
        tkl_won      = get(tr, "tackles_won")           # TklW
        drib_chall   = get(tr, "challenges")            # Dribbles Challenged
        drib_tackles = get(tr, "challenge_tackles")     # Dribblers Tackled
        challenges_lost = get(tr, "challenges_lost")    # Lost

        # converto in interi
        to_int = lambda x: int(x) if x and x.isdigit() else 0
        i_tkl          = to_int(tkl)
        i_won          = to_int(tkl_won)
        i_chall        = to_int(drib_chall)
        i_tackles_drib = to_int(drib_tackles)
        i_lost         = to_int(challenges_lost)

        # calcolo duels secondo le nuove regole
        total_duels           = i_tkl + i_chall
        total_duels_won       = i_won + i_tackles_drib
        duel_lost             = i_lost + (i_tkl - i_won)
        total_duels_won_pct   = (
            f"{100 * total_duels_won / total_duels:.2f}"
            if total_duels > 0 else ""
        )

        # popolo il dizionario di output
        out[key] = {
            # metriche esistenti
            "tackles":               tkl,
            "tackles_won":           tkl_won,
            "tackles_won_percentage": f"{100*i_won/i_tkl:.1f}" if i_tkl > 0 else "",
            "interceptions":         get(tr, "interceptions"),
            "blocked_shots":         get(tr, "blocked_shots"),
            "clearances":            get(tr, "clearances"),
            "error_lead_to_shot":    get(tr, "errors"),

            # nuove metriche duels
            "total_duels":               str(total_duels),
            "total_duels_won":           str(total_duels_won),
            "duel_lost":                 str(duel_lost),
            "total_duels_won_percentage": total_duels_won_pct,
        }

    return out

# ---------- 5b. POSSESSION
def parse_possession(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_possession_dom_lg")
    out = {}
    if not tbl:
        return out

    get = lambda tr, k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        mis   = get(tr, "miscontrols") or "0"
        dis   = get(tr, "dispossessed") or "0"
        pos_l = str(int(mis) + int(dis))

        out[key] = {
            # esistenti
            "touches":                         get(tr, "touches"),
            "successful_dribbles":             get(tr, "take_ons_won"),
            "successful_dribbles_percentage":  get(tr, "take_ons_won_pct"),
            "dispossessed":                    get(tr, "dispossessed"),
            "possession_lost":                 pos_l,
            "possession_won_att_third":        get(tr, "possession_won_att_third"),
        }
    return out


# ---------- 5c. MISCELLANEOUS
def parse_misc(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_misc_dom_lg")
    out = {}
    if not tbl:
        return out

    get = lambda tr, k: clean_num(tr.find("td", {"data-stat": k}).get_text()) if tr.find("td", {"data-stat": k}) else ""

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc, team = season_code(yr), tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key      = (sc, team)

        out[key] = {
            # esistenti
            "amm":                        get(tr, "cards_yellow"),
            "esp":                        get(tr, "cards_red"),
            "fouls":                      get(tr, "fouls"),
            "was_fouled":                 get(tr, "fouled"),
            "offsides":                   get(tr, "offsides"),
            "crosses":                    get(tr, "crosses"),
            "penalty_won":                get(tr, "pens_won"),
            "penalty_conceded":           get(tr, "pens_conceded"),
            "aerial_duels_won":           get(tr, "aerials_won"),
            "aerial_lost":                get(tr, "aerials_lost"),
            "aerial_duels_won_percentage": get(tr, "aerials_won_pct"),
            "blocked_shots":              get(tr, "blocked_shots"),
            "clearances":                 get(tr, "clearances"),
            "dispossessed":               get(tr, "dispossessed"),
            "error_lead_to_shot":         get(tr, "errors"),
            "ball_recoveries":            get(tr, "ball_recoveries"),
            # nuova metrica
            "error_lead_to_goal":         get(tr, "errors_lead_to_goal"),
        }
    return out


# ---------- 6. GOALKEEPING
def parse_keeper(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_keeper_dom_lg")
    out = {}
    if not tbl:
        return out

    get = lambda tr, k: clean_num(tr.find("td", {"data-stat": k}).get_text()) \
                       if tr.find("td", {"data-stat": k}) else ""

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc   = season_code(yr)
        team = tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key  = (sc, team)

        out[key] = {
            "clean_sheet":         get(tr, "gk_clean_sheets"),
            "saves":               get(tr, "gk_saves"),
            "penalty_faced":       get(tr, "gk_pens_allowed"),
            "attempt_penalty_miss":get(tr, "gk_pens_missed"),
            "rp":                  get(tr, "gk_pens_saved"),
            "punches":             get(tr, "punches"),
            "crosses_not_claimed": get(tr, "gk_crosses_stopped_pct"),
        }
    return out

# ---------- 7. ADVANCED GOALKEEPING
def parse_keeper_adv(soup) -> Dict[Key, Dict]:
    tbl = extract_table(soup, "stats_keeper_adv_dom_lg")
    out = {}
    if not tbl:
        return out

    get = lambda tr, k: clean_num(tr.find("td", {"data-stat": k}).get_text()) \
                       if tr.find("td", {"data-stat": k}) else ""

    for tr in tbl.select("tbody tr"):
        yr = tr.th.get_text(strip=True)
        if "-" not in yr:
            continue
        sc   = season_code(yr)
        team = tr.find("td", {"data-stat": "team"}).get_text(strip=True)
        key  = (sc, team)

        out[key] = {
            "successful_runs_out":             get(tr, "gk_def_actions_outside_pen_area"),  # #OPA
        }
    return out


# ───────────────────────────────────────────────────
# COMPILA GIOCATORE  (merge di TUTTI i parsers, inclusi GK)
# ───────────────────────────────────────────────────
def compile_player(pl):
    soup = BeautifulSoup(fetch(pl["link"]), "html.parser")
    bio  = parse_bio(soup)

    std  = parse_standard(soup)
    sh   = parse_shooting(soup)
    pas  = parse_passing(soup)
    ptyp = parse_pass_types(soup)
    gca  = parse_gca(soup)
    dfn  = parse_defensive(soup)
    pos  = parse_possession(soup)
    misc = parse_misc(soup)

    # se è portiere, parse anche keeper e advanced
    keeper     = parse_keeper(soup) if bio.get("role") == "P" else {}
    keeper_adv = parse_keeper_adv(soup) if bio.get("role") == "P" else {}

    # merge di tutti i dizionari per stagione
    seasons = std
    for dct in (sh, pas, ptyp, gca, dfn, pos, misc, keeper, keeper_adv):
        for key, vals in dct.items():
            seasons.setdefault(key, {c: "" for c in FULL_COLS}).update(vals)

    # compongo le righe finali
    rows = []
    for rec in seasons.values():
        rec.update(
            player_id           = pl["player_id"],
            name                = pl["name"].upper(),
            slug                = pl["slug"],
            link                = pl["link"],
            image               = bio.get("image", ""),
            role                = bio.get("role", ""),
            preferred_foot      = bio.get("preferred_foot", ""),
            country             = bio.get("country", ""),
            date_of_birth       = bio.get("date_of_birth", ""),
            height              = bio.get("height", ""),
        )
        rows.append(rec)

    return rows


# ───────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────
def main():
    # ─── Inizializza file di output ───
    if not Path(OUT_FILE).exists():
        pd.DataFrame(columns=FULL_COLS).to_csv(OUT_FILE, index=False)

    # ─── Carica lista giocatori ───
    if Path(PLAYERS_CSV).exists():
        players = pd.read_csv(PLAYERS_CSV).to_dict("records")
    else:
        players = get_players(COMP_URL)

    # ─── Resume: salta player già presenti nell'output ───
    processed_ids = set()
    try:
        if Path(OUT_FILE).stat().st_size > 0:
            processed_ids = set(pd.read_csv(OUT_FILE, usecols=["player_id"])["player_id"].astype(str).unique())
    except Exception:
        pass

    todo = [p for p in players if str(p["player_id"]) not in processed_ids]
    total = len(todo)
    print(f"Da processare: {total} giocatori (saltati {len(players) - total} già presenti)")

    failed: List[Dict] = []
    BATCH = 30  # pausa lunga ogni 30 player
    for i, p in enumerate(todo, 1):
        try:
            rows = compile_player(p)
            if rows:
                pd.DataFrame(rows, columns=FULL_COLS).to_csv(OUT_FILE, mode="a", header=False, index=False)
            print(f"[{i}/{total}] {p['name']} → {len(rows)} righe")
        except Exception as e:
            failed.append(p)
            print(f"[!] {p['name']} errore: {e}")

        # pausa “gentile” tra player
        time.sleep(random.uniform(4.0, 8.0))

        # cool-down di batch per evitare rate-limit prolungati
        if i % BATCH == 0 and i < total:
            cool = random.uniform(60, 120)
            print(f"Cool-down di batch: {int(cool)}s")
            time.sleep(cool)

    # ─── Secondo pass sui falliti con cool-down più aggressivo ───
    if failed:
        print(f"\nSecondo tentativo su {len(failed)} giocatori falliti…")
        time.sleep(random.uniform(90, 150))
        retried_failed = []
        for j, p in enumerate(failed, 1):
            try:
                # ruota subito la sessione per il retry
                global SCRAPER
                SCRAPER = new_scraper()
                rows = compile_player(p)
                if rows:
                    pd.DataFrame(rows, columns=FULL_COLS).to_csv(OUT_FILE, mode="a", header=False, index=False)
                print(f"[retry {j}/{len(failed)}] {p['name']} → {len(rows)} righe")
            except Exception as e:
                retried_failed.append((p, str(e)))
                print(f"[retry KO] {p['name']} errore: {e}")
            time.sleep(random.uniform(6.0, 12.0))

        if retried_failed:
            print("\n⚠️ Ancora falliti:")
            for p, msg in retried_failed:
                print(f"   - {p['name']}: {msg}")

    print("✅ Output →", OUT_FILE)


if __name__ == "__main__":
    main()
