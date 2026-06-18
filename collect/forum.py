"""
forum.py — Scrapes planted-tank advice content from plantedtank.net forums.
Uses requests + BeautifulSoup. Respects robots.txt delays and uses a
descriptive User-Agent.
"""

import time
import hashlib
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.plantedtank.net"
HEADERS = {
    "User-Agent": "TakeMeter-DataCollector/1.0 (academic project; contact: student)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Forum sub-sections most likely to contain evaluable advice claims.
# Tweak if plantedtank.net changes its URL structure.
FORUM_PATHS = [
    "/forums/fertilizers-and-water-parameters.15/",
    "/forums/co2-injection.16/",
    "/forums/plants.9/",
    "/forums/lighting.20/",
    "/forums/substrate.10/",
]

MAX_PAGES_PER_FORUM = 3   # pages of thread listings to crawl per forum section
MAX_THREADS_PER_PAGE = 10 # threads to open per listing page
MIN_POST_CHARS = 150       # skip very short posts


def _get_soup(url: str, delay: float = 2.0) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup object, or None on failure."""
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception as exc:
        print(f"  [forum] fetch error {url}: {exc}")
        return None


def _extract_post_text(post_el) -> str | None:
    """Extract readable text from a forum post element."""
    # Remove quotes/citations (nested replies) to avoid duplicating content
    for quote in post_el.select(".bbCodeBlock--quote, blockquote, .quote"):
        quote.decompose()
    text = post_el.get_text(separator=" ", strip=True)
    # Collapse whitespace
    text = " ".join(text.split())
    return text if len(text) >= MIN_POST_CHARS else None


def _scrape_thread(thread_url: str) -> list[dict]:
    """Open a thread and extract first-post content (OP only)."""
    soup = _get_soup(thread_url)
    if not soup:
        return []

    posts_found = []

    # XenForo structure: article.message or div.message-body
    selectors = [
        "article.message--post",
        "div.message",
        "li.message",
        "div.postContainer",  # older vBulletin fallback
    ]

    messages = []
    for sel in selectors:
        messages = soup.select(sel)
        if messages:
            break

    # Take only the OP (first post) to keep examples independent
    for msg in messages[:1]:
        body = (
            msg.select_one(".message-body .bbWrapper")
            or msg.select_one(".message-body")
            or msg.select_one(".postContent")
            or msg
        )
        text = _extract_post_text(body)
        if text:
            posts_found.append({
                "text": text,
                "source": "plantedtank.net",
                "url": thread_url,
                "score": 0,
                "id": hashlib.md5(thread_url.encode()).hexdigest(),
            })

    return posts_found


def _scrape_forum_listing(forum_path: str) -> list[str]:
    """Return a list of thread URLs from a forum listing page (multi-page aware)."""
    thread_urls: list[str] = []

    for page in range(1, MAX_PAGES_PER_FORUM + 1):
        if page == 1:
            url = BASE_URL + forum_path
        else:
            url = BASE_URL + forum_path + f"page-{page}/"

        soup = _get_soup(url)
        if not soup:
            break

        # XenForo thread links sit inside <div class="structItem-title">
        links = soup.select(
            "div.structItem-title a[href*='/threads/'], "
            "h3.structItem-title a[href*='/threads/'], "
            "a.PreviewTooltip"
        )

        if not links:
            # Fallback: any link containing /threads/
            links = [
                a for a in soup.find_all("a", href=True)
                if "/threads/" in a["href"] and "unread" not in a["href"]
            ]

        added = 0
        for a in links[:MAX_THREADS_PER_PAGE]:
            href = a["href"]
            full_url = href if href.startswith("http") else BASE_URL + href
            if full_url not in thread_urls:
                thread_urls.append(full_url)
                added += 1

        print(f"  [forum] {forum_path} page {page} → {added} threads found")

        if added == 0:
            break  # no more pages

    return thread_urls


def scrape_forum(delay: float = 2.0) -> list[dict]:
    """
    Crawl plantedtank.net forum sections and extract OP posts.
    Returns a list of post dicts: text, source, url, score, id.
    """
    seen_ids: set[str] = set()
    posts: list[dict] = []

    for forum_path in FORUM_PATHS:
        print(f"\n  [forum] scanning {forum_path} ...")
        thread_urls = _scrape_forum_listing(forum_path)

        for thread_url in thread_urls:
            extracted = _scrape_thread(thread_url)
            for post in extracted:
                if post["id"] not in seen_ids:
                    seen_ids.add(post["id"])
                    posts.append(post)

        print(f"  [forum] section total so far: {len(posts)} posts")

    return posts
