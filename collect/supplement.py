"""
supplement.py — Targeted collection to fix label imbalance after the main run.

Problems this solves:
  1. misinformation is severely underrepresented (OP posts with bad advice get downvoted/removed)
  2. evidence_based needs more examples

Strategy:
  - For misinformation: scrape COMMENTS from aquarium subreddits
    (comments have higher myth density; OP posts rarely contain overt misinformation)
  - For evidence_based: scrape comments from known high-quality threads
    (wiki posts, pinned guides, expert Q&A threads)
  - Undersample experiential in the final CSV if it exceeds 70%

Arctic Shift comments API:
  GET /api/comments/search?subreddit=X&limit=100&sort=desc&after=YYYY-MM-DD
"""

import time
import requests
import pandas as pd
from pathlib import Path

ARCTIC_SHIFT_BASE = "https://arctic-shift.photon-reddit.com/api"
HEADERS = {
    "User-Agent": "TakeMeter-DataCollector/1.0 (academic project)",
    "Accept": "application/json",
}

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_CSV = DATA_DIR / "dataset_raw.csv"
SUPPLEMENT_CSV = DATA_DIR / "dataset_supplement_raw.csv"

# ---------------------------------------------------------------------------
# Date windows to sweep for comments
# ---------------------------------------------------------------------------
COMMENT_DATE_WINDOWS = [
    ("2023-01-01", "2024-01-01"),
    ("2022-01-01", "2023-01-01"),
    ("2021-01-01", "2022-01-01"),
    ("2020-01-01", "2021-01-01"),
]

# Subreddits with higher misinformation comment density
MISINFO_SUBREDDITS = [
    {"sub": "Aquariums",  "target": 120},
    {"sub": "bettafish",  "target": 100},
    {"sub": "shrimptank", "target": 60},
    {"sub": "PlantedTank","target": 60},
]

# Subreddits for evidence_based comments (expert advice threads)
EVIDENCE_SUBREDDITS = [
    {"sub": "PlantedTank", "target": 80},
    {"sub": "Aquariums",   "target": 60},
]

# Skip phrases for comment pre-filter
COMMENT_SKIP = [
    "lol", "haha", "😂", "🤣", "beautiful", "gorgeous", "amazing", "love it",
    "congrats", "nice", "cute", "aww", "wow", "<3", "❤️",
]

MIN_COMMENT_LEN = 80   # characters


def _fetch_comments(sub: str, after: str, before: str, delay: float = 1.2) -> list[dict]:
    url = f"{ARCTIC_SHIFT_BASE}/comments/search"
    params = {"subreddit": sub, "limit": 100, "sort": "desc", "after": after, "before": before}
    try:
        time.sleep(delay)
        resp = requests.get(url, headers=HEADERS, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            print(f"  [supp] API error: {data['error']}")
            return []
        return data.get("data") or []
    except Exception as exc:
        print(f"  [supp] fetch error ({sub} {after}): {exc}")
        return []


def _extract_comment(raw: dict) -> dict | None:
    body = (raw.get("body") or "").strip()
    if body in ("[removed]", "[deleted]", ""):
        return None
    if len(body) < MIN_COMMENT_LEN:
        return None
    lower = body.lower()
    if any(p in lower for p in COMMENT_SKIP) and len(body) < 150:
        return None
    return {
        "text": body,
        "source": f"reddit/r/{raw.get('subreddit', '')}/comments",
        "url": f"https://reddit.com/r/{raw.get('subreddit','')}/comments/{raw.get('link_id','').replace('t3_','')}",
        "score": raw.get("score", 0),
        "id": f"c_{raw.get('id', '')}",
    }


def scrape_comment_supplements(seen_ids: set[str], delay: float = 1.2) -> list[dict]:
    """
    Scrape comments from aquarium subreddits across multiple date windows.
    Returns list of comment post dicts.
    """
    posts: list[dict] = []

    all_targets = (
        [{"sub": t["sub"], "target": t["target"]} for t in MISINFO_SUBREDDITS] +
        [{"sub": t["sub"], "target": t["target"]} for t in EVIDENCE_SUBREDDITS]
    )
    # Deduplicate sub targets
    seen_subs: dict[str, int] = {}
    unique_targets = []
    for t in all_targets:
        if t["sub"] not in seen_subs:
            seen_subs[t["sub"]] = t["target"]
            unique_targets.append(t)
        else:
            seen_subs[t["sub"]] = max(seen_subs[t["sub"]], t["target"])

    for target in unique_targets:
        sub = target["sub"]
        goal = seen_subs[sub]
        sub_posts: list[dict] = []
        print(f"\n  [supp] r/{sub} comments — targeting {goal}")

        for after, before in COMMENT_DATE_WINDOWS:
            if len(sub_posts) >= goal:
                break
            raw_batch = _fetch_comments(sub, after, before, delay)
            added = 0
            for raw in raw_batch:
                c = _extract_comment(raw)
                if c and c["id"] not in seen_ids:
                    seen_ids.add(c["id"])
                    sub_posts.append(c)
                    posts.append(c)
                    added += 1
            print(f"           {after}-{before}: {added} kept (sub total: {len(sub_posts)}/{goal})")

        print(f"           r/{sub} final: {len(sub_posts)} comments")

    return posts


def undersample_experiential(df: pd.DataFrame, cap: int = 90) -> pd.DataFrame:
    """
    If experiential exceeds `cap`, randomly sample down to cap.
    Other labels are kept in full.
    """
    exp = df[df["label"] == "experiential"]
    other = df[df["label"] != "experiential"]
    if len(exp) > cap:
        exp = exp.sample(n=cap, random_state=42)
        print(f"  Undersampled experiential: {len(df[df['label']=='experiential'])} → {cap}")
    return pd.concat([other, exp], ignore_index=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    # Load existing IDs to avoid duplicates
    existing_ids: set[str] = set()
    if RAW_CSV.exists():
        existing_df = pd.read_csv(RAW_CSV)
        if "id" in existing_df.columns:
            existing_ids = set(existing_df["id"].astype(str))
        print(f"  Loaded {len(existing_ids)} existing IDs from dataset_raw.csv")

    # Scrape comment supplements
    print("\n=== Scraping comment supplements ===")
    new_posts = scrape_comment_supplements(existing_ids)
    print(f"\n  Total new comment posts: {len(new_posts)}")

    # Save supplement raw
    DATA_DIR.mkdir(exist_ok=True)
    supp_df = pd.DataFrame(new_posts)[["text", "source", "url", "score", "id"]]
    supp_df.to_csv(SUPPLEMENT_CSV, index=False, encoding="utf-8")
    print(f"  Saved → {SUPPLEMENT_CSV}")

    # Label with Groq agent
    from collect.label_agent import label_posts
    print(f"\n=== Labeling {len(new_posts)} supplement posts ===")
    labeled = label_posts(new_posts, api_key=api_key, delay=1.5, verbose=True)

    # Merge with existing labeled CSV and apply undersampling
    LABELED_CSV = DATA_DIR / "dataset_prelabeled.csv"
    existing_labeled = pd.read_csv(LABELED_CSV) if LABELED_CSV.exists() else pd.DataFrame()

    new_labeled_df = pd.DataFrame(labeled)
    if not new_labeled_df.empty:
        new_labeled_df["notes"] = ""

    merged = pd.concat([existing_labeled, new_labeled_df], ignore_index=True)
    # Drop error rows
    merged = merged[~merged["label"].isin(["_api_error", "_parse_error"])].copy()

    print(f"\n=== Label distribution before undersampling ===")
    print(merged["label"].value_counts().to_string())

    merged = undersample_experiential(merged, cap=90)

    print(f"\n=== Final label distribution ===")
    print(merged["label"].value_counts().to_string())

    merged.to_csv(LABELED_CSV, index=False, encoding="utf-8")
    print(f"\n  Saved merged dataset → {LABELED_CSV} ({len(merged)} rows)")
