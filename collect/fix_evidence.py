#!/usr/bin/env python3
"""
fix_evidence.py — Rescues evidence_based rows that GPT-4o-mini over-demoted to experiential.

Root cause: GPT-4o-mini applied academic citation standards to hobby forum posts.
Posts like "I dose EI (KNO3/KH2PO4/K2SO4 3x/week + 50% WC) at 20ppm CO2" were
demoted because they use "I", even though they cite specific named methodology + parameters.

Fix: Rows where prelabel=evidence_based and review corrected to experiential
are re-evaluated by heuristic. If the text contains specific numeric parameters
OR a named methodology, restore them to evidence_based.
"""
import re
import sys
from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
AUDIT_CSV  = DATA_DIR / "dataset_review_audit.csv"
OUTPUT_CSV = DATA_DIR / "dataset_labeled.csv"

# Patterns that indicate specific, verifiable claim content
PARAMETER_PATTERNS = [
    r"\b\d+\s*(ppm|ppb|mg/l|dkh|dgh|gh|kh|tds|par|lux|kelvin|nm)\b",   # numeric + unit
    r"\bph\s*\d",                  # pH value
    r"\b\d+\s*°?\s*[fc]\b",       # temperature
    r"\b\d+\s*%\s*(water change|wc)\b",  # water change schedule
    r"\b\d+\s*(gallon|gal|liter|litre)\b",
    r"\b(ammonia|nitrite|nitrate|phosphate|potassium|iron)\s*[:<]\s*\d",
    r"\b(kno3|kh2po4|k2so4|csm\+b|seachem|flourish|ei dosing|pps-pro|walstad|"
    r"estimative index|tom barr|diana walstad|ada aqua soil|fluval stratum)\b",
    r"\b(fishless cycle|nitrogen cycle|beneficial bacteria|nitrifying bacteria)\b",
    r"\b\d+x\s*per\s*(week|day|month)\b",   # dosing frequency
    r"\b(drop checker|4dkh|api test|master test kit|inline|co2 reactor|diffuser)\b",
]
PARAM_RE = re.compile("|".join(PARAMETER_PATTERNS), re.IGNORECASE)


def has_parameters(text: str) -> bool:
    return bool(PARAM_RE.search(text))


def main():
    audit = pd.read_csv(AUDIT_CSV)
    print(f"Loaded audit: {len(audit)} rows")

    # Identify the wrongly-demoted rows:
    # pre-label was evidence_based, GPT-4o-mini corrected to experiential
    wrongly_demoted = audit[
        (audit["prelabel"] == "evidence_based") &
        (audit["review_action"] == "correct") &
        (audit["label"] == "experiential")
    ].copy()
    print(f"\nWrongly-demoted candidates (eb→exp): {len(wrongly_demoted)}")

    # Apply parameter heuristic
    wrongly_demoted["has_params"] = wrongly_demoted["text"].apply(has_parameters)
    restore = wrongly_demoted[wrongly_demoted["has_params"]]
    keep_exp = wrongly_demoted[~wrongly_demoted["has_params"]]

    print(f"  → Restore to evidence_based (have parameters): {len(restore)}")
    print(f"  → Keep as experiential (no parameters found): {len(keep_exp)}")

    if len(restore) > 0:
        print("\nSample restored rows:")
        for _, r in restore.head(3).iterrows():
            print(f"  {str(r['text'])[:130]}")
            print()

    # Build the corrected labeled dataset
    # Start from the current labeled CSV
    current = pd.read_csv(OUTPUT_CSV)

    # Re-run: rebuild from audit with the fix applied
    fixed_rows = []
    for _, row in audit.iterrows():
        if row["review_action"] == "remove":
            continue  # correctly excluded

        lbl = row["label"]
        # Restore wrongly-demoted evidence_based
        if (row["prelabel"] == "evidence_based"
                and row["review_action"] == "correct"
                and row["label"] == "experiential"
                and has_parameters(str(row["text"]))):
            lbl = "evidence_based"

        fixed_rows.append({
            "text":   row["text"],
            "label":  lbl,
            "source": row.get("source", ""),
            "url":    row.get("url", ""),
            "notes":  row.get("notes", ""),
        })

    out = pd.DataFrame(fixed_rows)
    out = out[out["label"].isin({"evidence_based", "experiential", "misinformation"})]

    print(f"\nFinal dataset: {len(out)} rows")
    vc = out["label"].value_counts()
    for lbl, n in vc.items():
        pct = 100 * n / len(out)
        flag = "✓" if pct >= 20 else ("⚠️  low (<20%)" if pct >= 10 else "❌ critical (<10%)")
        print(f"  {lbl:<22} {n:>4}  ({pct:.1f}%)  {flag}")

    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved → {OUTPUT_CSV}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
