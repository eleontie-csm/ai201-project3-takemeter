"""
reddit.py — Scrapes planted-tank subreddits via Arctic Shift API (free, no credentials).

Confirmed valid params:
  subreddit, limit (max 100), sort (asc|desc by created_utc),
  after (YYYY-MM-DD), before (YYYY-MM-DD)
No server-side score/is_self filter — handled client-side.
"""

import time
import requests

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
HEADERS = {
    "User-Agent": "TakeMeter-DataCollector/1.0 (academic project)",
    "Accept": "application/json",
}

# Date windows for pagination (each window = 1 API call of 100 posts).
# Text posts on aquarium subreddits score low (1-5); we fetch many windows
# and rely on the Groq agent for quality filtering instead of score thresholds.
DATE_WINDOWS = [
    ("2025-01-01", "2026-06-01"),
    ("2024-01-01", "2025-01-01"),
    ("2023-01-01", "2024-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2019-01-01", "2021-01-01"),
]

# Per-subreddit config: min_score = 1 (just exclude brand-new/spam posts),
# target = how many usable self-posts to collect before moving on.
SUBREDDIT_TARGETS = [
    {"sub": "PlantedTank", "min_score": 1, "target": 150},
    {"sub": "Aquascape",   "min_score": 1, "target": 80},
    {"sub": "Aquariums",   "min_score": 1, "target": 120},  # higher misinformation density
    {"sub": "bettafish",   "min_score": 1, "target": 80},
    {"sub": "shrimptank",  "min_score": 1, "target": 50},
]

SKIP_PHRASES = [
    "rate my", "what do you think of my", "please help", "is this okay",
    "am i doing this right", "can someone identify", "id this plant",
    "i just set up", "just started my", "first tank", "new to this",
    "thoughts on my", "check out my",
]


def _extract_post(raw: dict) -> dict | None:
    if not raw.get("is_self"):
        return None
    title = (raw.get("title") or "").strip()
    body = (raw.get("selftext") or "").strip()
    if body in ("[removed]", "[deleted]"):
        body = ""
    text = f"{title}\n\n{body}".strip() if body else title
    if not text or len(text) < 100:
        return None
    lower = text.lower()
    if any(phrase in lower for phrase in SKIP_PHRASES) and len(text) < 300:
        return None
    permalink = raw.get("permalink", "")
    return {
        "text": text,
        "source": f"reddit/r/{raw.get('subreddit', '')}",
        "url": f"https://reddit.com{permalink}" if permalink else "",
        "score": raw.get("score", 0),
        "id": raw.get("id", raw.get("name", "")),
    }


def _fetch_window(sub: str, after: str, before: str, delay: float) -> list[dict]:
    url = f"{ARCTIC_SHIFT_BASE}/posts/search"
    params = {"subreddit": sub, "limit": 100, "sort": "desc", "after": after, "before": before}
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            print(f"  [reddit] API error: {data['error']}")
            return []
        return data.get("data") or []
    except Exception as exc:
        print(f"  [reddit] fetch error ({sub} {after}-{before}): {exc}")
        return []


def scrape_subreddits(delay: float = 1.2) -> list[dict]:
    """Fetch self-posts from all subreddits. Returns list of {text, source, url, score, id}."""
    seen_ids: set[str] = set()
    posts: list[dict] = []

    for target in SUBREDDIT_TARGETS:
        sub, min_score, goal = target["sub"], target["min_score"], target["target"]
        sub_posts: list[dict] = []
        print(f"\n  [reddit] r/{sub} — targeting {goal} posts (min_score>={min_score})")

        for after, before in DATE_WINDOWS:
            if len(sub_posts) >= goal:
                break
            raw_batch = _fetch_window(sub, after, before, delay)
            added = 0
            for raw in raw_batch:
                if raw.get("score", 0) < min_score:
                    continue
                post = _extract_post(raw)
                if post and post["id"] not in seen_ids:
                    seen_ids.add(post["id"])
                    sub_posts.append(post)
                    posts.append(post)
                    added += 1
            print(f"           {after}-{before}: {added} kept (sub total: {len(sub_posts)}/{goal})")

        print(f"           r/{sub} final: {len(sub_posts)} posts")

    return posts
