"""
reddit_boost.py — Scrape additional Reddit posts from unused subreddits
via Arctic Shift API to reach TARGET_TOTAL labeled examples.

Targets subreddits not in the original collection:
  r/Goldfish, r/Cichlid, r/FishTank, r/Aquarium, r/PlantedTank (older windows)

Usage:
    python3 collect/reddit_boost.py
"""

import os
import sys
import time
import json
import re
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from groq import Groq

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")

TARGET_TOTAL = 200
INPUT_CSV  = BASE / "data" / "dataset_labeled.csv"
OUTPUT_CSV = BASE / "data" / "dataset_labeled.csv"
CHECKPOINT = BASE / "data" / "reddit_boost_ckpt.csv"

API = "https://arctic-shift.photon-reddit.com/api"

# Additional subreddits not scraped before, or earlier windows of existing ones
SOURCES = [
    {"sub": "Goldfish",         "limit": 100, "after": "2020-01-01", "before": "2024-01-01"},
    {"sub": "Cichlid",          "limit": 100, "after": "2020-01-01", "before": "2024-01-01"},
    {"sub": "FishTank",         "limit": 100, "after": "2021-01-01", "before": "2024-01-01"},
    {"sub": "PlantedTank",      "limit": 100, "after": "2016-01-01", "before": "2019-01-01"},
    {"sub": "Aquariums",        "limit": 100, "after": "2016-01-01", "before": "2019-01-01"},
    {"sub": "shrimptank",       "limit": 100, "after": "2018-01-01", "before": "2022-01-01"},
    {"sub": "Goldfish",         "limit": 100, "after": "2017-01-01", "before": "2020-01-01"},
    {"sub": "Cichlid",          "limit": 100, "after": "2017-01-01", "before": "2020-01-01"},
    {"sub": "bettafish",        "limit": 100, "after": "2016-01-01", "before": "2019-01-01"},
    {"sub": "Aquascape",        "limit": 100, "after": "2019-01-01", "before": "2022-01-01"},
    {"sub": "FishTank",         "limit": 100, "after": "2018-01-01", "before": "2021-01-01"},
    {"sub": "shrimptank",       "limit": 100, "after": "2015-01-01", "before": "2018-01-01"},
]

MIN_SCORE  = 1
MIN_CHARS  = 100
SKIP_STARTS = (
    "what ", "how ", "why ", "when ", "where ", "which ", "should i",
    "is it ", "can i ", "am i ", "anyone ", "does ", "do i ",
    "has anyone", "have you", "help me", "advice ", "hi ", "hey ",
    "update:", "[update"
)
SKIP_PHRASES = ("just wanted to share", "look at my", "meet my", "finally got", "check out")

# ── Groq labeler ──────────────────────────────────────────────────────────────
MODEL = "llama-3.1-8b-instant"
SYSTEM = (
    "You label aquarium/fishkeeping posts. Return ONLY a JSON object, no prose.\n\n"
    "Labels (pick exactly one when not_a_claim=false):\n"
    "  analysis   – post explains a mechanism, names a technique/method, or gives a specific\n"
    "               how-to recommendation with a reason. Personal framing fine if reasoning present.\n"
    "  anecdote   – claim backed only by personal experience with no transferable reasoning;\n"
    "               'works for me', vague tips with no why.\n\n"
    "Set not_a_claim=true for pure questions, photo showcases, or posts with no evaluable claim.\n"
    "JSON keys: not_a_claim (bool), label (str), confidence (high|medium|low), reasoning (1 sentence)"
)
VALID = {"analysis", "anecdote"}


def fetch_posts(sub: str, limit: int, after: str, before: str, delay: float = 1.0) -> list[dict]:
    """Fetch self-posts from Arctic Shift."""
    try:
        time.sleep(delay)
        r = requests.get(
            f"{API}/posts/search",
            params={"subreddit": sub, "limit": limit, "sort": "desc", "after": after, "before": before},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"  [fetch error] r/{sub}: {e}")
        return []

    posts = []
    for raw in data:
        if not raw.get("is_self"):
            continue
        score = raw.get("score", 0)
        if score < MIN_SCORE:
            continue
        title = raw.get("title", "")
        body  = raw.get("selftext", "")
        # Filter removed/deleted
        if body.strip() in ("", "[removed]", "[deleted]"):
            body = ""
        text = (title + "\n\n" + body).strip() if body else title
        if len(text) < MIN_CHARS:
            continue
        posts.append({
            "text":   text,
            "source": f"r/{sub}",
            "url":    f"https://reddit.com{raw.get('permalink', '')}",
            "score":  score,
            "id":     raw.get("id", ""),
        })
    return posts


def is_skip(text: str) -> bool:
    low = text.lower().strip()
    if low.endswith("?") and len(low) < 200:
        return True
    first = low[:80]
    if any(first.startswith(s) for s in SKIP_STARTS):
        return True
    if any(p in low[:200] for p in SKIP_PHRASES):
        return True
    return False


def call_groq(client: Groq, text: str, retries: int = 4) -> tuple[str, str]:
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
    existing_texts = set(df_existing["text"].str[:120].tolist())
    existing_urls  = set(df_existing["url"].tolist())
    current_n = len(df_existing)
    need = TARGET_TOTAL - current_n
    print(f"Current: {current_n} rows. Need {need} more → target {TARGET_TOTAL}.")

    if need <= 0:
        print("Already at target.")
        return

    # Resume checkpoint
    done_urls: set[str] = set()
    new_rows: list[dict] = []
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        done_urls = set(ckpt["url"].astype(str).tolist())
        new_rows  = ckpt.to_dict("records")
        print(f"Checkpoint: {len(new_rows)} already labeled")

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    labeled = len(new_rows)

    for src in SOURCES:
        if labeled >= need:
            break
        sub = src["sub"]
        print(f"\n[reddit] r/{sub} {src['after']} → {src['before']}")
        raw_posts = fetch_posts(sub, src["limit"], src["after"], src["before"])
        print(f"  fetched {len(raw_posts)} candidates")

        for post in raw_posts:
            if labeled >= need:
                break
            if post["url"] in done_urls or post["url"] in existing_urls:
                continue
            if post["text"][:120] in existing_texts:
                continue
            if is_skip(post["text"]):
                continue

            label, conf = call_groq(client, post["text"])
            done_urls.add(post["url"])

            if label == "_skip":
                continue

            rec = {
                "text":   post["text"],
                "source": post["source"],
                "url":    post["url"],
                "notes":  f"reddit_boost; conf={conf}",
                "label":  label,
            }
            new_rows.append(rec)
            labeled += 1
            print(f"  [{labeled:3d}/{need}] {label:<10} ({conf}) | {post['text'][:65]}")

            if labeled % 5 == 0:
                pd.DataFrame(new_rows).to_csv(CHECKPOINT, index=False)

            time.sleep(1.3)

    # Merge
    if new_rows:
        df_new = pd.DataFrame(new_rows)[["text", "source", "url", "notes", "label"]]
        df_out = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_out = df_existing

    df_out.to_csv(OUTPUT_CSV, index=False)
    if CHECKPOINT.exists():
        CHECKPOINT.unlink()

    print(f"\n=== Final dataset ===")
    print(df_out["label"].value_counts())
    pct = df_out["label"].value_counts(normalize=True).mul(100).round(1)
    for lbl, p in pct.items():
        print(f"  {lbl:<12} {p}%")
    print(f"\nTotal: {len(df_out)} rows saved → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
