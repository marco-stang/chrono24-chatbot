"""Einmaliger lokaler Scraper für die Chrono24-Hilfeseiten. Läuft nie im Deployment."""
from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

BASE_URL = "https://www.chrono24.de"
START_URL = f"{BASE_URL}/info/faqs.htm"
SEED_URLS = [START_URL, f"{BASE_URL}/info/index.htm"]
RAW_DIR = Path("data/raw")
REQUEST_DELAY_S = 1.0
CHALLENGE_TITLE_MARKER = "Just a moment"
CHALLENGE_MIN_CONTENT_LENGTH = 20_000
CHALLENGE_MAX_WAIT_S = 90.0
CHALLENGE_POLL_INTERVAL_S = 3.0


def collect_info_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    base_host = urlparse(base_url).netloc
    links: set[str] = set()
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"]).split("#")[0].split("?")[0]
        parsed = urlparse(href)
        if parsed.netloc == base_host and parsed.path.startswith("/info/") and parsed.path.endswith(".htm"):
            links.add(href)
    return sorted(links)


def url_to_filename(url: str) -> str:
    return urlparse(url).path.strip("/").replace("/", "__") + ".html"


def filename_to_url(name: str) -> str:
    return f"{BASE_URL}/" + name.removesuffix(".html").replace("__", "/")


async def _wait_for_real_content(page) -> bool:
    """Pollt bis zu CHALLENGE_MAX_WAIT_S lang alle CHALLENGE_POLL_INTERVAL_S,
    ob eine Cloudflare-Challenge vorbei ist: Titel ohne "Just a moment" und
    Body länger als CHALLENGE_MIN_CONTENT_LENGTH Zeichen. True = echter
    Inhalt da, False = Timeout ohne echten Inhalt."""
    elapsed = 0.0
    while True:
        title = await page.title()
        html = await page.content()
        if CHALLENGE_TITLE_MARKER not in title and len(html) > CHALLENGE_MIN_CONTENT_LENGTH:
            return True
        if elapsed >= CHALLENGE_MAX_WAIT_S:
            return False
        await asyncio.sleep(CHALLENGE_POLL_INTERVAL_S)
        elapsed += CHALLENGE_POLL_INTERVAL_S


async def scrape() -> None:
    """Lädt beide Seed-Seiten, sammelt deren /info/-Links und lädt jede Seite genau
    einmal. Bewusst nur eine Ebene tief — keine Rekursion.

    Läuft headed mit echtem Chrome (channel="chrome"), nicht headless
    Chromium: Chrono24 sitzt hinter Cloudflare-Bot-Schutz, der headless
    Chromium mit HTTP 403 ("Just a moment...") blockt. Im sichtbaren Fenster
    kann eine Person die Challenge manuell lösen, während der Scraper auf
    echten Seiteninhalt wartet."""
    from playwright.async_api import async_playwright

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(channel="chrome", headless=False)
        page = await browser.new_page()
        saved: set[str] = set()
        queue: list[str] = list(SEED_URLS)
        queued: set[str] = set(SEED_URLS)
        while queue:
            url = queue.pop(0)
            if saved:
                await asyncio.sleep(REQUEST_DELAY_S)
            await page.goto(url, wait_until="domcontentloaded")
            if not await _wait_for_real_content(page):
                print(
                    f"BLOCKED: Cloudflare-Challenge nach {CHALLENGE_MAX_WAIT_S:.0f}s "
                    f"nicht gelöst: {url}"
                )
                await browser.close()
                return
            html = await page.content()
            (RAW_DIR / url_to_filename(url)).write_text(html, encoding="utf-8")
            saved.add(url)
            print(f"gespeichert: {url}")
            if url in SEED_URLS:
                for link in collect_info_links(html, BASE_URL):
                    if link not in queued:
                        queued.add(link)
                        queue.append(link)
        await browser.close()
        print(f"{len(saved)} Seiten gespeichert")


if __name__ == "__main__":
    asyncio.run(scrape())
