#!/usr/bin/env python3
"""
review_agent.py — Uses GPT-4o-mini to perform human-equivalent label review.

For each row in dataset_prelabeled.csv the model receives:
  - The post text
  - The pre-assigned label (from Groq 8b-instant)
  - The pre-assigned reasoning
  - The full label definitions and decision rules

It returns:
  - confirmed_label  — the reviewed label (may differ from the pre-label)
  - action           — "confirm" | "correct" | "remove"
                       remove = post is not actually an evaluable claim
  - reasoning        — updated one-sentence justification
  - review_note      — what changed and why (empty string if confirmed)

Output: data/dataset_labeled.csv  (ready for Colab upload)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

DATA_DIR = Path(__file__).parent.parent / "data"
INPUT_CSV  = DATA_DIR / "dataset_prelabeled.csv"
OUTPUT_CSV = DATA_DIR / "dataset_labeled.csv"

MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a senior annotator reviewing pre-labeled posts from aquarium and fishkeeping communities (Reddit r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank).

LABEL DEFINITIONS:

  analysis — The post explains a mechanism, names a technique or methodology, or gives a
    specific how-to recommendation with a reason. Personal framing ("I dose EI because...") is
    fine as long as the reasoning is transferable to another keeper.
    EXAMPLES that ARE analysis:
      "CO2 should be 20-30 ppm; verify with a drop checker using 4dKH reference — green = 30 ppm"
      "I dose EI (KNO3, KH2PO4, K2SO4 3x/week) because high-light tanks burn through macros fast"
      "Run both filters in parallel for a few weeks so the new one seeds bacteria from the old"
      "Ammonia 0, Nitrite 0, Nitrates 10ppm means the cycle is complete — safe to add fish"

  anecdote — The claim is supported only by personal experience with no transferable reasoning.
    Plausible or even correct, but the only justification is personal practice or outcome.
    EXAMPLES: "I've kept it this way for 3 years with no issues", "works great in my tank",
    "just try it", "I use Excel and growth is fine"

DECISION RULES:
  1. Personal framing + reasoning = analysis. Personal framing alone = anecdote.
  2. Naming a technique alone (without a why or how) is NOT enough for analysis.
  3. If the post is a pure question, one-word reaction, pure showcase, or contains no evaluable claim: action=remove.

Return ONLY a JSON object with these keys:
  action           (string): "confirm" | "correct" | "remove"
  confirmed_label  (string): analysis | anecdote
  reasoning        (string): one sentence justification
  review_note      (string): if action=correct or remove, explain briefly; else ""
"""

VALID_LABELS = {"analysis", "anecdote"}
VALID_ACTIONS = {"confirm", "correct", "remove"}


def review_one(client: OpenAI, text: str, prelabel: str, prereasoning: str) -> dict:
    truncated = text[:500] + ("..." if len(text) > 500 else "")
    user_msg = (
        f"POST:\n{truncated}\n\n"
        f"PRE-ASSIGNED LABEL: {prelabel}\n"
        f"PRE-ASSIGNED REASONING: {prereasoning}"
    )
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                temperature=0.1,
                max_tokens=180,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            result = json.loads(raw)
            # Validate
            if result.get("action") not in VALID_ACTIONS:
                result["action"] = "confirm"
            lbl = result.get("confirmed_label", prelabel)
            if lbl not in VALID_LABELS:
                result["confirmed_label"] = prelabel
            return result
        except Exception as exc:
            if "429" in str(exc) or "rate" in str(exc).lower():
                print(f"\n  [review] rate limit — waiting 20s ...")
                time.sleep(20)
                continue
            if attempt < 2:
                time.sleep(3)
                continue
            return {
                "action": "confirm",
                "confirmed_label": prelabel,
                "reasoning": prereasoning,
                "review_note": f"review error: {exc}",
            }


def main():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set in .env")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV.name}")
    print("Pre-label distribution:")
    print(df["label"].value_counts().to_string())
    print(f"\nModel: {MODEL}")
    print(f"Estimated cost: ~${len(df) * 0.0002:.2f} (gpt-4o-mini pricing)\n")

    results = []
    confirmed = corrected = removed = errors = 0

    for i, row in enumerate(df.itertuples(index=False), 1):
        if i % 20 == 1:
            print(f"  [review] {i}/{len(df)} | confirmed={confirmed} corrected={corrected} removed={removed}")

        result = review_one(
            client,
            text=str(row.text),
            prelabel=str(row.label),
            prereasoning=str(row.reasoning) if hasattr(row, "reasoning") else "",
        )
        time.sleep(0.3)  # ~200 RPM well under 500 RPM limit

        action = result.get("action", "confirm")
        if action == "confirm":
            confirmed += 1
        elif action == "correct":
            corrected += 1
        elif action == "remove":
            removed += 1

        results.append({
            "text":          row.text,
            "prelabel":      row.label,
            "label":         result.get("confirmed_label", row.label),
            "confidence":    getattr(row, "confidence", ""),
            "reasoning":     result.get("reasoning", ""),
            "review_action": action,
            "review_note":   result.get("review_note", ""),
            "source":        getattr(row, "source", ""),
            "url":           getattr(row, "url", ""),
            "notes":         getattr(row, "notes", ""),
        })

    print(f"\n  [review] done: {len(df)} reviewed")
    print(f"  confirmed: {confirmed}  corrected: {corrected}  removed: {removed}")

    out_df = pd.DataFrame(results)

    # Drop removed rows
    kept = out_df[out_df["review_action"] != "remove"].copy()
    print(f"\nRows after removing non-claims: {len(kept)}")

    print("\nFinal label distribution:")
    vc = kept["label"].value_counts()
    print(vc.to_string())
    for lbl, n in vc.items():
        pct = 100 * n / len(kept)
        flag = "✓" if pct >= 15 else "⚠️  low"
        print(f"  {lbl:<22} {n:>3}  ({pct:.1f}%)  {flag}")

    # Save only the columns the Colab notebook expects
    final = kept[["text", "label", "source", "url", "notes"]].reset_index(drop=True)
    final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    # Also save the full audit trail (pre-label, confirmed label, review_action, review_note)
    audit_path = DATA_DIR / "dataset_review_audit.csv"
    out_df.to_csv(audit_path, index=False, encoding="utf-8")
    print(f"Audit trail saved → {audit_path}")

    print(f"\nSaved → {OUTPUT_CSV}  ({len(final)} rows)")
    print("\nNext step: upload dataset_labeled.csv to Colab as your training CSV.")


if __name__ == "__main__":
    main()
