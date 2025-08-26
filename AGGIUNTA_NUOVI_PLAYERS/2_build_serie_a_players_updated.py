#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FBref click-first builder (fix SERP “vuote”)
- 1) DuckDuckGo HTML (no-JS) → stabile in headless
- 2) DuckDuckGo classico (SPA)
- 3) Bing
Clicca/legge il primo link valido a /en/players/<id>/..., estrae name/id/slug.
"""

from __future__ import annotations
import csv, logging, random, re, subprocess, sys, time, unicodedata
from dataclasses import dataclass
from typing import List, Optional, Tuple
from urllib.parse import urlparse, unquote, quote_plus, urlsplit, parse_qs
from pathlib import Path

# ---------------------- Config ----------------------
HEADLESS_DEFAULT = False
DELAY_DEFAULT = 0.7
JITTER_DEFAULT = 0.5
INPUT_DEFAULT = "./out/giocatori_nuovi.csv"
OUTDIR_DEFAULT = "./out"
LOGLEVEL_DEFAULT = "INFO"
MAX_RESULTS = 12
WAIT_SELECTOR_TIMEOUT = 15000
NAV_TIMEOUT = 20000
RETRIES_PER_ENGINE = 2
# ---------------------------------------------------

def _pip_install(pkg: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg],
                          stdout=sys.stdout, stderr=sys.stderr)

def ensure_deps() -> None:
    try:
        import pandas as _pd  # noqa
    except Exception:
        print("[setup] Installing pandas ..."); _pip_install("pandas")
    try:
        import playwright  # noqa
    except Exception:
        print("[setup] Installing playwright ..."); _pip_install("playwright")
    try:
        from playwright.sync_api import sync_playwright  # noqa
        try:
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True); b.close()
        except Exception:
            print("[setup] Installing Playwright browsers (chromium) ...")
            subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                                  stdout=sys.stdout, stderr=sys.stderr)
    except Exception:
        print("[setup] Re-install playwright and browsers ...")
        _pip_install("playwright")
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"],
                              stdout=sys.stdout, stderr=sys.stderr)

ensure_deps()

import pandas as pd
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeoutError

logger = logging.getLogger("fbref_clickfirst_builder")
FBREF_PLAYER_PATH_RE = re.compile(r"^/en/players/([A-Za-z0-9]+)/([^/?#]+)")

def setup_logging(level: str = "INFO"):
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    ch = logging.StreamHandler(); ch.setFormatter(fmt)
    logger.handlers.clear(); logger.addHandler(ch)

def slugify(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("’", "'").replace("`", "'")
    s = re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-").lower()
    return s

def extract_from_fbref_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    try:
        pu = urlparse(url)
        m = FBREF_PLAYER_PATH_RE.match(pu.path)
        if not m: return None, None
        player_id = m.group(1)
        pretty = unquote(pu.path.rstrip("/").split("/")[-1])
        return player_id, slugify(pretty)
    except Exception:
        return None, None

def _ddg_decode_redirect(url: str) -> str:
    try:
        parts = urlsplit(url)
        if "duckduckgo.com" in parts.netloc and parts.path.startswith("/l"):
            q = parse_qs(parts.query)
            if "uddg" in q and q["uddg"]:
                return unquote(q["uddg"][0])
    except Exception:
        pass
    return url

def _maybe_accept_cookies(page, engine: str) -> None:
    selectors_by_engine = {
        "ddg": [
            "button[aria-label='Accept all']","button:has-text('Accept')","button:has-text('Accetta')",
            "button:has-text('OK')","button:has-text('I agree')","#consent-banner button",
        ],
        "bing": [
            "#bnp_btn_accept","button:has-text('Accept')","button:has-text('Accetta')",
            "button:has-text('I agree')","button#bnp_close_button",
        ],
    }
    try:
        for sel in selectors_by_engine.get(engine, []):
            loc = page.locator(sel)
            if loc.count() > 0:
                loc.first.click(); break
    except Exception:
        pass

def _stabilize_serp(page):
    """Aiuta le SERP “vuote”: aspetta networkidle, scrolla giù e su, piccola pausa."""
    try:
        page.wait_for_load_state("networkidle", timeout=8000)
    except Exception:
        pass
    try:
        page.mouse.wheel(0, 2000); time.sleep(0.25); page.mouse.wheel(0, -2000)
    except Exception:
        pass
    time.sleep(0.2)

def _force_enter_if_empty(page, qbox_selector: str, results_selector: str):
    """Se non ci sono risultati visibili, invia Enter sulla searchbox."""
    try:
        if page.locator(results_selector).count() == 0:
            box = page.locator(qbox_selector)
            if box.count() > 0:
                box.first.press("Enter")
                _stabilize_serp(page)
    except Exception:
        pass

def _click_and_capture_url(page, locator) -> Optional[str]:
    try:
        with page.expect_navigation(wait_until="domcontentloaded", timeout=NAV_TIMEOUT):
            locator.click()
        return page.url
    except Exception:
        try:
            with page.expect_popup() as popinfo:
                locator.click()
            popup = popinfo.value
            popup.wait_for_load_state("domcontentloaded", timeout=NAV_TIMEOUT)
            url = popup.url; popup.close()
            return url
        except Exception:
            return None

def _is_valid_fbref_player_url(url: str) -> bool:
    try:
        pu = urlparse(url)
        return "fbref.com" in pu.netloc and FBREF_PLAYER_PATH_RE.match(pu.path) is not None
    except Exception:
        return False

# --------- Engine 1: DuckDuckGo HTML (no JS) ----------
def ddg_html_click_first_fbref(page, query: str) -> Optional[str]:
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    page.goto(url, wait_until="domcontentloaded")
    _stabilize_serp(page)
    # SERP statica: risultati in .result__a oppure a[href*='fbref.com']
    selectors = ["a.result__a", "div.results_links a[href*='fbref.com']", "a[href*='fbref.com']"]
    for sel in selectors:
        anchors = page.locator(sel)
        n = min(anchors.count(), MAX_RESULTS)
        for i in range(n):
            a = anchors.nth(i)
            href = (a.get_attribute("href") or "").strip()
            if not href: continue
            href = _ddg_decode_redirect(href)
            if _is_valid_fbref_player_url(href):
                return href
    return None

# --------- Engine 2: DuckDuckGo classico ----------
def ddg_click_first_fbref(page, query: str) -> Optional[str]:
    url = f"https://duckduckgo.com/?q={quote_plus(query)}&kl=it-it"
    page.goto(url, wait_until="domcontentloaded")
    _maybe_accept_cookies(page, engine="ddg")
    _stabilize_serp(page)
    _force_enter_if_empty(page, qbox_selector="input#search_form_input", results_selector="#links, article")
    try:
        page.wait_for_selector("a[href]", timeout=WAIT_SELECTOR_TIMEOUT)
    except Exception:
        pass
    selectors = [
        "a[data-testid='result-title-a']",
        "#links .result__a",
        "article a.result__a",
        "a[href*='fbref.com']",
    ]
    for sel in selectors:
        anchors = page.locator(sel)
        n = min(anchors.count(), MAX_RESULTS)
        for i in range(n):
            a = anchors.nth(i)
            href = (a.get_attribute("href") or "").strip()
            if href:
                href = _ddg_decode_redirect(href)
            if "fbref.com" not in (href or ""):
                txt = (a.text_content() or "")
                if "fbref.com" not in txt:
                    continue
            target_url = _click_and_capture_url(page, a)
            if target_url and _is_valid_fbref_player_url(target_url):
                return target_url
            else:
                try: page.go_back(wait_until="domcontentloaded")
                except Exception: pass
    return None

# --------- Engine 3: Bing ----------
def bing_click_first_fbref(page, query: str) -> Optional[str]:
    url = f"https://www.bing.com/search?q={quote_plus(query)}&setlang=it-IT&cc=IT"
    page.goto(url, wait_until="domcontentloaded")
    _maybe_accept_cookies(page, engine="bing")
    _stabilize_serp(page)
    _force_enter_if_empty(page, qbox_selector="input[name='q']", results_selector="ol#b_results")
    try:
        page.wait_for_selector("ol#b_results a[href]", timeout=WAIT_SELECTOR_TIMEOUT)
    except Exception:
        pass
    selectors = ["ol#b_results li.b_algo h2 a", "ol#b_results h2 a", "ol#b_results a[href]"]
    for sel in selectors:
        anchors = page.locator(sel)
        n = min(anchors.count(), MAX_RESULTS)
        for i in range(n):
            a = anchors.nth(i)
            href = (a.get_attribute("href") or "").strip()
            if "fbref.com" not in (href or ""):
                continue
            target_url = _click_and_capture_url(page, a)
            if target_url and _is_valid_fbref_player_url(target_url):
                return target_url
            else:
                try: page.go_back(wait_until="domcontentloaded")
                except Exception: pass
    return None

@dataclass
class NewRow:
    nome: str
    squadra: str
    ruolo: Optional[str]
    torneo: str

def read_giocatori_nuovi(path: str) -> List[NewRow]:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    nome_c = cols.get("nome") or "Nome"
    squadra_c = cols.get("squadra") or "Squadra"
    ruolo_c = cols.get("r") or ("R" if "R" in df.columns else None)
    torneo_c = cols.get("tournament_name") or "tournament_name"
    if torneo_c not in df.columns:
        df["tournament_name"] = "Serie A"; torneo_c = "tournament_name"
    out: List[NewRow] = []
    for _, r in df.iterrows():
        out.append(NewRow(
            nome=str(r[nome_c]).strip(),
            squadra=str(r[squadra_c]).strip(),
            ruolo=(str(r[ruolo_c]).strip() if ruolo_c and ruolo_c in df.columns else None),
            torneo=(str(r[torneo_c]).strip() if torneo_c in df.columns and pd.notna(r[torneo_c]) else "Serie A"),
        ))
    return out

def _best_effort_player_name(page, fallback_url: str) -> str:
    try:
        nm = page.locator("h1[itemprop='name'], h1").first.text_content()
        nm = (nm or "").strip()
        if nm: return nm
    except Exception: pass
    try:
        tit = (page.title() or "").strip()
        if tit: return re.sub(r"\s*\|\s*FBref.*$", "", tit).strip()
    except Exception: pass
    try:
        last = unquote(urlparse(fallback_url).path.rstrip("/").split("/")[-1])
        return last.replace("-", " ").strip().title()
    except Exception:
        return "Unknown"

def process_rows(rows: List[NewRow], outdir: str, headless: bool, delay: float, jitter: float) -> None:
    found_rows, miss_rows, logs = [], [], []
    outdir_path = Path(outdir); screens = outdir_path / "screens"
    screens.mkdir(parents=True, exist_ok=True)

    seen_queries = set()
    query_templates = [
        '{name} {team} site:fbref.com/en/players',
        '{name} site:fbref.com/en/players',
        '{name} fbref',
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless, args=["--disable-blink-features=AutomationControlled","--no-sandbox"])
        context = browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/127 Safari/537.36"),
            locale="it-IT",
            timezone_id="Europe/Rome",
        )
        page = context.new_page()
        page.set_default_navigation_timeout(NAV_TIMEOUT)

        engines = [
            ("ddg_html", ddg_html_click_first_fbref),
            ("ddg", ddg_click_first_fbref),
            ("bing", bing_click_first_fbref),
        ]

        total = len(rows)
        for idx, r in enumerate(rows, 1):
            key = (r.nome.lower().strip(), r.squadra.lower().strip())
            if key in seen_queries:
                logger.info(f"[{idx}/{total}] SKIP duplicato: {r.nome} ({r.squadra})")
                time.sleep(0.2 + random.uniform(0, 0.2)); continue
            seen_queries.add(key)

            logger.info(f"[{idx}/{total}] {r.nome} ({r.squadra}) → primo link FBref")
            variants = [t.format(name=r.nome, team=r.squadra) for t in query_templates]

            chosen_url = None; used_engine = ""; used_query = ""
            for engine_name, engine_fn in engines:
                if chosen_url: break
                for q in variants:
                    ok = False
                    for attempt in range(1, RETRIES_PER_ENGINE + 1):
                        try:
                            qry = q + " fbref"
                            chosen_url = engine_fn(page, qry)
                            if chosen_url:
                                used_engine, used_query = engine_name, qry
                                ok = True
                            break
                        except PWTimeoutError:
                            logger.warning(f"{engine_name.upper()} timeout ({attempt}/{RETRIES_PER_ENGINE})")
                        except Exception as e:
                            logger.debug(f"{engine_name.upper()} errore ({attempt}/{RETRIES_PER_ENGINE}): {e}")
                        time.sleep(0.3 + random.uniform(0, 0.5))
                    if ok: break
                    # screenshot SERP/variante fallita
                    try:
                        safeq = re.sub(r"[^a-z0-9]+", "_", (q.lower())[:80])
                        page.screenshot(path=str(screens / f"failed_{idx}_{engine_name}_{safeq}.png"))
                    except Exception: pass

            if chosen_url:
                try: page.goto(chosen_url, wait_until="domcontentloaded")
                except Exception: pass
                pid, slug = extract_from_fbref_url(chosen_url)
                name = _best_effort_player_name(page, chosen_url)
                found_rows.append([name, chosen_url, slug or "", pid or "", r.squadra, r.torneo])
                logs.append([idx, r.nome, r.squadra, used_engine, used_query, chosen_url, "found", pid or "", slug or ""])
                logger.info(f"→ {name} | {chosen_url}")
            else:
                try:
                    safeq = re.sub(r"[^a-z0-9]+", "_", f"{r.nome}_{r.squadra}".lower())[:80]
                    page.screenshot(path=str(screens / f"failed_{idx}_final_{safeq}.png"))
                except Exception: pass
                miss_rows.append([r.nome, r.squadra, r.torneo, f"{r.nome} {r.squadra} fbref", "no_fbref_link"])
                logs.append([idx, r.nome, r.squadra, "", f"{r.nome} {r.squadra} fbref", "", "missing", "", ""])
                logger.warning(f"× Nessun link FBref per: {r.nome} ({r.squadra}) — screenshot salvati.")

            time.sleep(max(0.2, delay + random.uniform(0, jitter)))

        browser.close()

    Path(outdir).mkdir(parents=True, exist_ok=True)
    with open(Path(outdir) / "serie_a_players_updated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["name","link","slug","player_id","team","tournament_name_new"]); w.writerows(found_rows)
    with open(Path(outdir) / "serie_a_players_not_found.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["query_name","team","tournament","query","reason"]); w.writerows(miss_rows)
    with open(Path(outdir) / "serie_a_players_search_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["idx","name_in","team_in","engine","query","chosen_url","status","player_id","slug"]); w.writerows(logs)

    logger.info(f"Salvati {len(found_rows)} giocatori → {Path(outdir) / 'serie_a_players_updated.csv'}")
    logger.info(f"Non trovati: {len(miss_rows)} → {Path(outdir) / 'serie_a_players_not_found.csv'}")

def main():
    setup_logging(LOGLEVEL_DEFAULT)
    here = Path(__file__).resolve().parent
    input_path = str((here / INPUT_DEFAULT).resolve())
    outdir = str((here / OUTDIR_DEFAULT).resolve())
    logger.info(f"Input: {input_path}")
    logger.info(f"Outdir: {outdir}")
    logger.info(f"Defaults → headless={HEADLESS_DEFAULT}, delay={DELAY_DEFAULT}, jitter={JITTER_DEFAULT}")
    rows = read_giocatori_nuovi(input_path)
    logger.info(f"Loaded {len(rows)} rows.")
    process_rows(rows, outdir, headless=HEADLESS_DEFAULT, delay=DELAY_DEFAULT, jitter=JITTER_DEFAULT)

if __name__ == "__main__":
    main()
