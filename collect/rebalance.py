#!/usr/bin/env python3
"""
rebalance.py — Post-processing script to:
1. Remove obvious question slip-throughs (heuristic filter)
2. Undersample evidence_based to match target balance
3. Save the clean, balanced dataset as dataset_prelabeled.csv (in-place)

Run: python3 collect/rebalance.py
"""
import re
import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
CSV = DATA_DIR / "dataset_prelabeled.csv"

# Target counts (sum must stay >= 200)
TARGET = {
    "evidence_based": 80,
    "experiential": 90,
    "misinformation": 29,   # keep all; it's genuinely rare
}

QUESTION_STARTS = re.compile(
    r"^(what |how |why |is |are |can |should |does |do |will |which |"
    r"help |advice |suggestions |looking for|help with|trouble with|"
    r"anyone know|anyone have|i need help|i'm confused|i am confused)",
    re.IGNORECASE
)


def is_likely_question(text: str) -> bool:
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if not lines:
        return False
    first_line = lines[0]
    if first_line.endswith("?"):
        return True
    if QUESTION_STARTS.match(first_line):
        return True
    if len(text) < 200 and text.count("?") > 0:
        return True
    return False


def main():
    df = pd.read_csv(CSV)
    print(f"Loaded {len(df)} rows from {CSV.name}")
    print("\nBefore cleaning:")
    print(df["label"].value_counts().to_string())

    # ── Step 1: remove slip-through questions ──────────────────────────────
    q_mask = df["text"].apply(is_likely_question)
    questions_removed = q_mask.sum()
    df = df[~q_mask].copy()
    print(f"\nRemoved {questions_removed} likely-question slip-throughs")
    print("After question filter:")
    print(df["label"].value_counts().to_string())

    # ── Step 2: undersample over-represented labels ────────────────────────
    parts = []
    for label, target in TARGET.items():
        subset = df[df["label"] == label]
        if len(subset) > target:
            subset = subset.sample(n=target, random_state=42)
            print(f"  Undersampled {label}: {df[df['label']==label].shape[0]} → {target}")
        elif len(subset) < target:
            print(f"  {label}: only {len(subset)} available (target {target}) — keeping all")
        parts.append(subset)

    # Keep any labels not in TARGET (error rows, etc.) if they exist
    known = set(TARGET.keys())
    others = df[~df["label"].isin(known)]
    if len(others):
        print(f"  Dropping {len(others)} rows with unexpected labels: {others['label'].unique()}")

    balanced = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=42)

    print(f"\nFinal dataset: {len(balanced)} rows")
    print(balanced["label"].value_counts().to_string())
    pct = balanced["label"].value_counts(normalize=True) * 100
    for lbl, p in pct.items():
        flag = "✓" if p >= 20 else "⚠️ below 20%"
        print(f"  {lbl:<22} {p:.1f}%  {flag}")

    balanced.to_csv(CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {CSV}")


if __name__ == "__main__":
    main()
