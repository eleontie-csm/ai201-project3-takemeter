"""
run.py — Main orchestrator for TakeMeter data collection.

Pipeline:
  1. Scrape Reddit (r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish)
  2. Scrape plantedtank.net forums
  3. Deduplicate by near-exact text match
  4. Run Groq label agent: filter non-claims + pre-label {evidence_based, experiential, misinformation}
  5. Balance check: warn if any label is underrepresented (<60 examples)
  6. Save two CSVs to ./data/:
       dataset_raw.csv         — everything scraped (before filtering)
       dataset_prelabeled.csv  — claims only, agent-labeled (ready for human review)

Usage:
  1. Copy .env.example to .env and fill in your GROQ_API_KEY
  2. pip install -r requirements.txt
  3. python collect/run.py

Human review step (after this script):
  Open data/dataset_prelabeled.csv in any spreadsheet tool or the notebook.
  Review the `label` column (especially rows where needs_review=True).
  Correct any wrong labels. The corrected file is your final labeled dataset.
"""

import os
import sys
import hashlib
import argparse
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Allow running as `python collect/run.py` from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from collect.reddit import scrape_subreddits
from collect.forum import scrape_forum
from collect.label_agent import label_posts, MODEL

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_CSV = DATA_DIR / "dataset_raw.csv"
LABELED_CSV = DATA_DIR / "dataset_prelabeled.csv"

TARGET_PER_LABEL = 80   # aim for 80 per label (240 total) for buffer above 200 minimum
MIN_PER_LABEL = 55      # warn threshold


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _text_fingerprint(text: str) -> str:
    """Short fingerprint: first 200 chars, lowercased, whitespace-collapsed."""
    normalized = " ".join(text.lower().split())[:200]
    return hashlib.md5(normalized.encode()).hexdigest()


def deduplicate(posts: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for p in posts:
        fp = _text_fingerprint(p["text"])
        if fp not in seen:
            seen.add(fp)
            unique.append(p)
    return unique


# ---------------------------------------------------------------------------
# Balance check & targeted re-scrape hint
# ---------------------------------------------------------------------------

def balance_check(df: pd.DataFrame) -> None:
    print("\n  ── Label distribution ──────────────────────────────")
    counts = df["label"].value_counts()
    for label in ["evidence_based", "experiential", "misinformation"]:
        n = counts.get(label, 0)
        flag = "⚠️  UNDERREPRESENTED" if n < MIN_PER_LABEL else "✓"
        print(f"    {label:<22} {n:>4}  {flag}")
    print("  ────────────────────────────────────────────────────")

    under = [l for l in ["evidence_based", "experiential", "misinformation"]
             if counts.get(l, 0) < MIN_PER_LABEL]
    if under:
        print(f"\n  ⚠️  Labels below minimum ({MIN_PER_LABEL}): {under}")
        print("  Suggestions:")
        if "misinformation" in under:
            print("    • Re-run with --extra-myths to fetch more misinformation posts")
            print("    • Try r/Aquariums and r/bettafish search for myth keywords")
        if "evidence_based" in under:
            print("    • Sort r/PlantedTank wiki/guides for parameter-heavy posts")
        if "experiential" in under:
            print("    • Search r/Aquascape for 'in my experience' posts")
    else:
        print("\n  ✓ All labels meet the minimum threshold. Ready for human review.")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

RAW_COLS = ["text", "source", "url", "score"]
LABELED_COLS = ["text", "label", "confidence", "reasoning", "needs_review", "source", "url", "id", "notes"]


def save_raw(posts: list[dict]) -> None:
    df = pd.DataFrame(posts)[RAW_COLS]
    df.to_csv(RAW_CSV, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} raw posts → {RAW_CSV}")


def save_labeled(labeled: list[dict]) -> None:
    df = pd.DataFrame(labeled)
    df["notes"] = ""  # blank column for annotator notes during human review
    # Keep only expected columns (add missing ones as empty)
    for col in LABELED_COLS:
        if col not in df.columns:
            df[col] = ""
    df = df[LABELED_COLS]
    df.to_csv(LABELED_CSV, index=False, encoding="utf-8")
    print(f"  Saved {len(df)} pre-labeled posts → {LABELED_CSV}")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(skip_forum: bool = False, skip_reddit: bool = False,
         dry_run: bool = False, label_only: bool = False) -> None:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key and not dry_run:
        print("ERROR: GROQ_API_KEY not set. Copy .env.example to .env and fill it in.")
        print("       Run with --dry-run to scrape only (no Groq calls).")
        sys.exit(1)

    DATA_DIR.mkdir(exist_ok=True)

    # ── Step 1: Scrape (or reload from raw CSV) ─────────────────────────────
    if label_only:
        if not RAW_CSV.exists():
            print(f"ERROR: {RAW_CSV} not found. Run without --label-only first to scrape.")
            sys.exit(1)
        print(f"\n=== Step 1: Loading existing raw CSV ({RAW_CSV}) ===")
        raw_df = pd.read_csv(RAW_CSV)
        unique_posts = raw_df.to_dict("records")
        print(f"  Loaded {len(unique_posts)} posts from dataset_raw.csv")
    else:
        all_posts: list[dict] = []

        if not skip_reddit:
            print("\n=== Step 1a: Reddit scraping ===")
            reddit_posts = scrape_subreddits()
            print(f"\n  Reddit: {len(reddit_posts)} posts collected")
            all_posts.extend(reddit_posts)
        else:
            print("  [skip] Reddit scraping skipped via flag")

        if not skip_forum:
            print("\n=== Step 1b: plantedtank.net forum scraping ===")
            forum_posts = scrape_forum()
            print(f"\n  Forum: {len(forum_posts)} posts collected")
            all_posts.extend(forum_posts)
        else:
            print("  [skip] Forum scraping skipped via flag")

        print(f"\n  Total before dedup: {len(all_posts)}")
        unique_posts = deduplicate(all_posts)
        print(f"  After dedup: {len(unique_posts)} unique posts")
        save_raw(unique_posts)

    # ── Step 3: Groq label agent ────────────────────────────────────────────
    if dry_run:
        print("\n=== [DRY RUN] Skipping Groq labeling ===")
        print(f"  dataset_raw.csv written with {len(unique_posts)} posts.")
        print("  Re-run with --label-only (and GROQ_API_KEY set) to label them.")
        return

    print("\n=== Step 2: Groq label agent (filter + pre-label) ===")
    print(f"  Sending {len(unique_posts)} posts to Groq {MODEL} ...")
    labeled = label_posts(unique_posts, api_key=api_key, delay=1.5, verbose=True,
                          checkpoint_path=str(LABELED_CSV))

    # ── Step 4: Save + balance check ────────────────────────────────────────
    print("\n=== Step 3: Saving output ===")
    df = save_labeled(labeled)
    balance_check(df)

    print("\n=== Done ===")
    print("Next step: open data/dataset_prelabeled.csv")
    print("  • Review each row — especially needs_review=True rows")
    print("  • Correct any wrong labels in the 'label' column")
    print("  • Add notes in the 'notes' column for hard cases (required for planning.md §8)")
    print("  • Rename to dataset_labeled.csv when review is complete")
    print("  • Upload to Colab as your training CSV\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TakeMeter data collection pipeline")
    parser.add_argument("--skip-forum",  action="store_true", help="Skip plantedtank.net scraping")
    parser.add_argument("--skip-reddit", action="store_true", help="Skip Reddit scraping")
    parser.add_argument("--dry-run",     action="store_true",
                        help="Scrape and save dataset_raw.csv but skip Groq labeling")
    parser.add_argument("--label-only",  action="store_true",
                        help="Skip scraping; load dataset_raw.csv and run labeling only")
    args = parser.parse_args()
    main(skip_forum=args.skip_forum, skip_reddit=args.skip_reddit,
         dry_run=args.dry_run, label_only=args.label_only)
