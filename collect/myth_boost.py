#!/usr/bin/env python3
"""
myth_boost.py — Targeted scrape to add more misinformation examples.

Strategy: scrape older date windows with 'asc' sort (oldest first) from
high-noise subreddits. Older comments have more myth content (pre-wiki era).
Filters for comments containing known myth keyphrases before sending to Groq.
"""
import sys
import time
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path(__file__).parent.parent / "data"
LABELED_CSV = DATA_DIR / "dataset_prelabeled.csv"

HEADERS = {
    "User-Agent": "TakeMeter-DataCollector/1.0 (academic project)",
    "Accept": "application/json",
}
BASE = "https://arctic-shift.photon-reddit.com/api"

# Older date windows more likely to contain pre-correction myths
DATE_WINDOWS = [
    ("2017-01-01", "2019-01-01"),
    ("2015-01-01", "2017-01-01"),
    ("2019-01-01", "2021-01-01"),
]

TARGETS = [
    {"sub": "Aquariums",   "limit": 100},
    {"sub": "bettafish",   "limit": 100},
    {"sub": "Aquariums",   "limit": 100},
    {"sub": "Goldfish",    "limit": 100},
    {"sub": "Cichlid",     "limit": 100},
]

# Misinformation keyphrases — if a comment contains one, it's a strong candidate
MYTH_PHRASES = [
    "fish grow to", "fish will grow", "bowl is fine", "bowls are fine",
    "dont need to cycle", "don't need to cycle", "no need to cycle",
    "excel is the same", "excel replaces co2", "excel instead of co2",
    "fish waste is enough", "fish waste provides", "fish poop fertiliz",
    "doesnt need a filter", "doesn't need a filter", "no filter needed",
    "fish in cycling is fine", "fish-in cycling works", "just add fish",
    "gravel is fine for plants", "fish don't need", "fish dont need",
    "no water changes needed", "don't need water changes",
    "tap water is always fine", "chlorine evaporates", "just let it sit",
]

MIN_LEN = 80


def fetch_comments(sub, after, before, delay=1.2):
    url = f"{BASE}/comments/search"
    params = {"subreddit": sub, "limit": 100, "sort": "asc", "after": after, "before": before}
    try:
        time.sleep(delay)
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        return r.json().get("data") or []
    except Exception as e:
        print(f"  [myth] error: {e}")
        return []


def has_myth(text: str) -> bool:
    lower = text.lower().replace("'", "").replace("'", "")
    return any(p in lower for p in MYTH_PHRASES)


def main():
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY not set")
        sys.exit(1)

    # Load existing IDs
    existing = pd.read_csv(LABELED_CSV)
    existing_ids = set(existing.get("id", pd.Series(dtype=str)).astype(str))
    print(f"Existing labeled rows: {len(existing)}")
    print(existing["label"].value_counts().to_string())

    # Scrape and pre-filter for myth phrases
    candidates = []
    seen: set[str] = set(existing_ids)

    print(f"\n=== Scraping for misinformation candidates ===")
    for target in TARGETS:
        sub = target["sub"]
        for after, before in DATE_WINDOWS:
            comments = fetch_comments(sub, after, before)
            found = 0
            for c in comments:
                body = (c.get("body") or "").strip()
                if body in ("[removed]", "[deleted]", "") or len(body) < MIN_LEN:
                    continue
                cid = f"c_{c.get('id','')}"
                if cid in seen:
                    continue
                if has_myth(body):
                    seen.add(cid)
                    candidates.append({
                        "text": body,
                        "source": f"reddit/r/{sub}/comments",
                        "url": f"https://reddit.com/r/{sub}",
                        "score": c.get("score", 0),
                        "id": cid,
                    })
                    found += 1
            print(f"  r/{sub} {after}-{before}: {found} myth candidates (total: {len(candidates)})")

    print(f"\nTotal myth candidates: {len(candidates)}")
    if not candidates:
        print("No candidates found — try expanding DATE_WINDOWS or MYTH_PHRASES")
        sys.exit(0)

    # Label with Groq
    from collect.label_agent import label_posts
    print(f"\n=== Labeling {len(candidates)} candidates ===")
    labeled = label_posts(candidates, api_key=api_key, delay=1.5, verbose=True)

    new_misinfo = [p for p in labeled if p.get("label") == "misinformation"]
    new_evidence = [p for p in labeled if p.get("label") == "evidence_based"]
    new_exp = [p for p in labeled if p.get("label") == "experiential"]
    print(f"\nFrom {len(candidates)} candidates:")
    print(f"  misinformation: {len(new_misinfo)}")
    print(f"  evidence_based: {len(new_evidence)}")
    print(f"  experiential:   {len(new_exp)}")

    if not labeled:
        print("Nothing to add.")
        sys.exit(0)

    # Merge and save
    new_df = pd.DataFrame(labeled)
    new_df["notes"] = ""
    COLS = ["text", "label", "confidence", "reasoning", "needs_review", "source", "url", "id", "notes"]
    for col in COLS:
        if col not in new_df.columns:
            new_df[col] = ""
    new_df = new_df[COLS]

    merged = pd.concat([existing, new_df], ignore_index=True)
    # Drop error rows
    merged = merged[~merged["label"].isin(["_api_error", "_parse_error"])].copy()

    print(f"\n=== Final distribution ({len(merged)} total) ===")
    vc = merged["label"].value_counts()
    print(vc.to_string())
    for lbl, n in vc.items():
        pct = 100 * n / len(merged)
        flag = "✓" if pct >= 20 else "⚠️  < 20%"
        print(f"  {lbl:<22} {n:>4}  ({pct:.1f}%)  {flag}")

    merged.to_csv(LABELED_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {LABELED_CSV} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
