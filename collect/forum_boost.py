"""
forum_boost.py — Scrape plantedtank.net forum posts, label them with Groq,
and merge into dataset_labeled.csv until we reach TARGET_TOTAL rows.

Usage:
    python3 collect/forum_boost.py
"""

import os
import sys
import time
import json
import re
import hashlib
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from groq import Groq

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")

TARGET_TOTAL = 200
INPUT_CSV  = BASE / "data" / "dataset_labeled.csv"
OUTPUT_CSV = BASE / "data" / "dataset_labeled.csv"
CHECKPOINT = BASE / "data" / "forum_boost_ckpt.csv"

# ── Forum config ─────────────────────────────────────────────────────────────
FORUM_BASE = "https://www.plantedtank.net"
HEADERS = {
    "User-Agent": "TakeMeter-DataCollector/1.0 (academic project)",
    "Accept-Language": "en-US,en;q=0.9",
}
FORUM_PATHS = [
    "/forums/fertilizers-and-water-parameters.15/",
    "/forums/co2-injection.16/",
    "/forums/plants.9/",
    "/forums/lighting.20/",
    "/forums/substrate.10/",
    "/forums/general-planted-tank-discussion.9/",
]
MAX_PAGES = 4
MIN_CHARS = 150

# ── Groq labeling ─────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"
SYSTEM = (
    "You label aquarium/fishkeeping posts. Return ONLY a JSON object, no prose.\n\n"
    "Labels (pick exactly one when not_a_claim=false):\n"
    "  analysis   – post explains a mechanism, names a technique/method, or gives a specific\n"
    "               how-to recommendation with a reason. Personal framing is fine if reasoning present.\n"
    "  anecdote   – claim backed only by personal experience with no transferable reasoning;\n"
    "               'works for me', vague tips with no why.\n\n"
    "Set not_a_claim=true for pure questions, photo showcases, or posts with no evaluable claim.\n"
    "JSON keys: not_a_claim (bool), label (str), confidence (high|medium|low), reasoning (1 sentence)"
)
VALID = {"analysis", "anecdote"}

# ── Skip-phrase filter (questions / showcases) ────────────────────────────────
SKIP_STARTS = (
    "what ", "how ", "why ", "when ", "where ", "which ", "should i",
    "is it ", "can i ", "am i ", "anyone ", "does ", "do i ",
    "has anyone", "have you", "help me", "advice ",
)


def _get_soup(url: str, delay: float = 2.0):
    try:
        time.sleep(delay)
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        return BeautifulSoup(r.text, "lxml")
    except Exception as e:
        print(f"  [fetch error] {url}: {e}")
        return None


def _extract_text(msg_el) -> str | None:
    for q in msg_el.select(".bbCodeBlock--quote, blockquote, .quote"):
        q.decompose()
    text = " ".join(msg_el.get_text(separator=" ", strip=True).split())
    return text if len(text) >= MIN_CHARS else None


def _is_question(text: str) -> bool:
    low = text.lower().strip()
    if low.endswith("?"):
        return True
    first = low[:60]
    return any(first.startswith(s) for s in SKIP_STARTS)


def scrape_forum(need: int) -> list[dict]:
    """Scrape up to need×3 threads and return raw post dicts."""
    seen_urls: set[str] = set()
    posts: list[dict] = []

    for forum_path in FORUM_PATHS:
        if len(posts) >= need * 3:
            break
        print(f"\n[forum] scanning {FORUM_BASE + forum_path}")
        for page in range(1, MAX_PAGES + 1):
            page_url = FORUM_BASE + forum_path + (f"page-{page}/" if page > 1 else "")
            soup = _get_soup(page_url)
            if not soup:
                break

            links = soup.select("div.structItem-title a[href*='/threads/'], h3.structItem-title a[href*='/threads/']")
            if not links:
                links = [a for a in soup.find_all("a", href=True) if "/threads/" in a["href"] and "unread" not in a["href"]]

            found_on_page = 0
            for a in links[:15]:
                href = a["href"]
                thread_url = href if href.startswith("http") else FORUM_BASE + href
                if thread_url in seen_urls:
                    continue
                seen_urls.add(thread_url)

                tsoup = _get_soup(thread_url, delay=2.5)
                if not tsoup:
                    continue

                # Try XenForo selectors
                messages = (
                    tsoup.select("article.message--post") or
                    tsoup.select("div.message") or
                    tsoup.select("li.message")
                )
                for msg in messages[:1]:  # OP only
                    body = (
                        msg.select_one(".message-body .bbWrapper") or
                        msg.select_one(".message-body") or
                        msg
                    )
                    text = _extract_text(body)
                    if text and not _is_question(text):
                        uid = hashlib.md5(thread_url.encode()).hexdigest()
                        posts.append({
                            "text": text,
                            "source": "plantedtank.net",
                            "url": thread_url,
                            "score": 0,
                            "id": uid,
                        })
                        found_on_page += 1

            print(f"  page {page}: {found_on_page} posts collected (total {len(posts)})")
            if found_on_page == 0:
                break

    return posts


def call_groq(client: Groq, text: str, retries: int = 4) -> tuple[str, str]:
    """Returns (label, confidence) or ('_skip', '') for not_a_claim."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"POST:\n{text[:600]}"},
                ],
                temperature=0.1,
                max_tokens=100,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            if not raw:
                raise ValueError("empty response")
            obj = json.loads(raw)
            if obj.get("not_a_claim"):
                return "_skip", ""
            label = obj.get("label", "").strip().lower()
            if label not in VALID:
                label = "anecdote"
            return label, obj.get("confidence", "low")
        except Exception as e:
            msg = str(e)
            m = re.search(r"try again in ([\d.]+)s", msg)
            wait = float(m.group(1)) + 1 if m else (30 if "429" in msg else 3)
            print(f"  retry {attempt+1}: {msg[:70]}  wait {wait:.0f}s")
            time.sleep(wait)
    return "anecdote", "low"


def main():
    df_existing = pd.read_csv(INPUT_CSV)
    current_n = len(df_existing)
    need = TARGET_TOTAL - current_n
    print(f"Current dataset: {current_n} rows. Need {need} more to reach {TARGET_TOTAL}.")

    if need <= 0:
        print("Already at target. Nothing to do.")
        return

    # Resume from checkpoint
    done_urls: set[str] = set()
    new_rows: list[dict] = []
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        done_urls = set(ckpt["url"].astype(str).tolist())
        new_rows = ckpt.to_dict("records")
        print(f"Resuming: {len(new_rows)} already labeled in checkpoint")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])

    # Scrape
    raw_posts = scrape_forum(need * 2)  # scrape extra as buffer for claim filter
    print(f"\nScraped {len(raw_posts)} candidate posts from plantedtank.net")

    # Label
    labeled_count = len(new_rows)
    for post in raw_posts:
        if labeled_count >= need:
            break
        if post["url"] in done_urls:
            continue
        # Skip if text already in existing dataset (dedup)
        if post["text"][:100] in df_existing["text"].str[:100].values:
            continue

        label, conf = call_groq(client, post["text"])
        done_urls.add(post["url"])

        if label == "_skip":
            print(f"  [skip] {post['text'][:60]}")
            continue

        rec = {
            "text":   post["text"],
            "source": post["source"],
            "url":    post["url"],
            "notes":  f"forum; conf={conf}",
            "label":  label,
        }
        new_rows.append(rec)
        labeled_count += 1
        print(f"  [{labeled_count:3d}/{need}] {label:<10} ({conf}) | {post['text'][:65]}")

        if labeled_count % 5 == 0:
            pd.DataFrame(new_rows).to_csv(CHECKPOINT, index=False)

        time.sleep(1.3)

    # Merge and save
    df_new = pd.DataFrame(new_rows)[["text", "source", "url", "notes", "label"]]
    df_out = pd.concat([df_existing, df_new], ignore_index=True)
    df_out.to_csv(OUTPUT_CSV, index=False)

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\n=== Final dataset ===")
    print(df_out["label"].value_counts())
    print(f"\nTotal: {len(df_out)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
