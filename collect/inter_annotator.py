"""
inter_annotator.py — GPT-4o-mini as second annotator for inter-annotator reliability.

Labels 30 randomly-sampled examples from dataset_labeled.csv independently,
then computes Cohen's kappa and percentage agreement against our primary labels.

Usage:
    python3 collect/inter_annotator.py
Output:
    data/inter_annotator_results.csv  — row-level agreement table
    Prints: Cohen's kappa, percentage agreement, disagreement analysis
"""

import os
import random
import json
import time
import re
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

BASE = Path(__file__).parent.parent
load_dotenv(BASE / ".env")
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Can be swapped to "gpt-4.5-mini" or any newer OpenAI model
MODEL = "gpt-4o-mini"

SAMPLE_N = 40       # sample more than 30 to have buffer for parse errors
RANDOM_SEED = 42

SYSTEM_PROMPT = """You are classifying posts from aquarium and fishkeeping communities on Reddit
(r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank).

Assign each post to exactly one of the following two categories:

analysis: The post explains a mechanism, names a technique or method, or gives a specific
recommendation with a reason. Personal framing ("I dose EI because...") is fine as long
as transferable reasoning is present — a reader could apply the logic to their own tank.
Example: "Run both filters in parallel for 4–6 weeks before removing the old one — bacteria
colonise the new media from the established filter, so you don't crash your cycle."

anecdote: The post is backed only by personal experience with no transferable reasoning.
"Works for me", "I've done this for years", vague tips with no explanation of why or how.
Example: "I've had my betta in a 5-gallon without a heater for two years and he's totally fine.
Just keep the room warm."

Respond with ONLY a JSON object with two keys:
  label      (string): "analysis" or "anecdote"
  confidence (string): "high", "medium", or "low"

Do not explain your reasoning. Output nothing but the JSON object."""

VALID = {"analysis", "anecdote"}


def label_one(text: str, retries: int = 3) -> tuple[str, str]:
    """Returns (label, confidence). Falls back to ('anecdote', 'low') on error."""
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Post:\n{text[:800]}"},
                ],
                temperature=0.0,
                max_tokens=40,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content.strip()
            obj = json.loads(raw)
            label = obj.get("label", "").strip().lower()
            if label not in VALID:
                label = "anecdote"
            conf = obj.get("confidence", "low")
            return label, conf
        except Exception as e:
            wait = 5 if "429" not in str(e) else 30
            print(f"  retry {attempt+1}: {str(e)[:60]}  waiting {wait}s")
            time.sleep(wait)
    return "anecdote", "low"


def cohen_kappa(labels_a: list, labels_b: list, classes: list) -> float:
    """Compute Cohen's kappa for two annotators."""
    n = len(labels_a)
    assert n == len(labels_b)
    # observed agreement
    po = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    # expected agreement
    pe = 0.0
    for c in classes:
        pa = labels_a.count(c) / n
        pb = labels_b.count(c) / n
        pe += pa * pb
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def main():
    df = pd.read_csv(BASE / "data" / "dataset_labeled.csv")
    print(f"Loaded {len(df)} rows. Sampling {SAMPLE_N} for inter-annotator study.")

    random.seed(RANDOM_SEED)
    sample = df.sample(n=min(SAMPLE_N, len(df)), random_state=RANDOM_SEED).reset_index(drop=True)

    second_labels = []
    second_confs = []

    for i, row in sample.iterrows():
        label, conf = label_one(str(row["text"]))
        second_labels.append(label)
        second_confs.append(conf)
        agree = "✓" if label == row["label"] else "✗"
        print(f"[{i+1:2d}/{len(sample)}] primary={row['label']:<9} second={label:<9} {agree}  {str(row['text'])[:55]}")
        time.sleep(0.3)

    sample["second_label"] = second_labels
    sample["second_confidence"] = second_confs
    sample["agree"] = sample["label"] == sample["second_label"]

    # ── Metrics ───────────────────────────────────────────────────────────
    pct_agree = sample["agree"].mean()
    kappa = cohen_kappa(
        sample["label"].tolist(),
        sample["second_label"].tolist(),
        ["analysis", "anecdote"],
    )

    print("\n" + "=" * 55)
    print("INTER-ANNOTATOR RELIABILITY")
    print("=" * 55)
    print(f"  Annotator 1 (Groq llama-3.1-8b-instant + GPT-4o-mini review)")
    print(f"  Annotator 2 ({MODEL} zero-shot)")
    print(f"  Examples compared: {len(sample)}")
    print(f"  Percentage agreement: {pct_agree:.1%}")
    print(f"  Cohen's kappa:        {kappa:.3f}")

    # Kappa interpretation
    if kappa >= 0.8:
        interp = "Almost perfect"
    elif kappa >= 0.6:
        interp = "Substantial"
    elif kappa >= 0.4:
        interp = "Moderate"
    elif kappa >= 0.2:
        interp = "Fair"
    else:
        interp = "Slight / poor"
    print(f"  Kappa interpretation: {interp}")

    # ── Disagreements ─────────────────────────────────────────────────────
    disagree = sample[~sample["agree"]]
    print(f"\n  Disagreements: {len(disagree)} / {len(sample)}")
    if len(disagree):
        # Directionality
        analysis_to_anecdote = ((disagree["label"] == "analysis") & (disagree["second_label"] == "anecdote")).sum()
        anecdote_to_analysis = ((disagree["label"] == "anecdote") & (disagree["second_label"] == "analysis")).sum()
        print(f"    primary=analysis → second=anecdote: {analysis_to_anecdote}")
        print(f"    primary=anecdote → second=analysis: {anecdote_to_analysis}")
        print("\n  Sample disagreements:")
        for _, r in disagree.head(5).iterrows():
            print(f"    [{r['label']} → {r['second_label']}] {str(r['text'])[:90]}")
            print()

    # ── Save ──────────────────────────────────────────────────────────────
    out = BASE / "data" / "inter_annotator_results.csv"
    sample.to_csv(out, index=False)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
