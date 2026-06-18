"""
Relabel dataset with new 3-label taxonomy:
  analysis      – post explains WHY or HOW; names technique, gives mechanism, or
                  specific how-to with a reason (personal framing OK if reasoning present)
  anecdote      – personal experience without transferable reasoning ("works for me")
  misinformation – factually incorrect claim stated as universal aquarium advice

Input:  data/dataset_labeled.csv  (150 rows, current labels)
Output: data/dataset_labeled.csv  (overwritten with new labels)
Checkpoint: data/dataset_relabeled_ckpt.csv
"""

import os, json, time, re
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE, ".env"))
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYSTEM = (
    "You label aquarium/fishkeeping posts with exactly one label. "
    "Return ONLY a JSON object with keys: label, confidence, reasoning.\n\n"
    "Labels:\n"
    "analysis – post explains a mechanism, names a technique/method, or gives a "
    "specific how-to recommendation with a reason. Personal framing ('I dose EI "
    "because the plants need macro nutrients') is fine as long as reasoning is present.\n"
    "anecdote – personal experience or vague endorsement with no transferable "
    "reasoning. 'Works great for me', 'I've done this for years', 'just try it'.\n"
    "misinformation – factually incorrect claim stated as universal aquarium advice "
    "(e.g. fish grow to tank size, you don't need to cycle, bettas are fine in bowls).\n\n"
    "confidence: high | medium | low"
)

VALID = {"analysis", "anecdote", "misinformation"}
CHECKPOINT = os.path.join(BASE, "data", "dataset_relabeled_ckpt.csv")
INPUT_CSV  = os.path.join(BASE, "data", "dataset_labeled.csv")
OUTPUT_CSV = os.path.join(BASE, "data", "dataset_labeled.csv")


def call_groq(text, retries=4):
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": f"Post:\n{text[:800]}"},
                ],
                temperature=0.1,
                max_tokens=120,
            )
            raw = resp.choices[0].message.content.strip()
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
            if not raw:
                raise ValueError("empty response from API")
            obj = json.loads(raw)
            label = obj.get("label", "").strip().lower()
            if label not in VALID:
                label = "anecdote"
            return label, obj.get("confidence", "low"), obj.get("reasoning", "")
        except Exception as e:
            msg = str(e)
            # Rate-limit: wait the time Groq specifies
            m = re.search(r"try again in ([\d.]+)s", msg)
            if m:
                wait = float(m.group(1)) + 1
            elif "429" in msg or "rate" in msg.lower():
                wait = 30
            else:
                # Empty / bad JSON — short retry
                wait = 3
            print(f"  retry {attempt+1}: {msg[:80]}  waiting {wait:.0f}s")
            time.sleep(wait)
    return "anecdote", "low", "error"


def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows from {INPUT_CSV}")
    print(f"Current label distribution:\n{df['label'].value_counts()}\n")

    # Resume from checkpoint
    done_urls = set()
    results = []
    if os.path.exists(CHECKPOINT):
        ckpt = pd.read_csv(CHECKPOINT)
        if "url" in ckpt.columns:
            done_urls = set(ckpt["url"].astype(str).tolist())
        results = ckpt.to_dict("records")
        print(f"Resuming from checkpoint: {len(done_urls)} already done")

    for i, row in df.iterrows():
        uid = str(row.get("url", i))
        if uid in done_urls:
            continue

        label, conf, reasoning = call_groq(str(row["text"]))

        rec = row.to_dict()
        rec["old_label"] = rec.get("label", "")
        rec["label"] = label
        rec["relabel_confidence"] = conf
        rec["relabel_reasoning"] = reasoning
        results.append(rec)
        done_urls.add(uid)

        print(f"[{len(results):3d}/{len(df)}] {label:<14} ({conf:<6}) | {str(row['text'])[:65]}")

        if len(results) % 5 == 0:
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

        time.sleep(1.2)

    out = pd.DataFrame(results)
    # Keep only columns that were in original + label (overwritten)
    keep_cols = [c for c in df.columns if c != "label"] + ["label"]
    out = out[[c for c in keep_cols if c in out.columns]]

    out.to_csv(OUTPUT_CSV, index=False)
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print("\n=== Final distribution ===")
    print(out["label"].value_counts())
    pct = out["label"].value_counts(normalize=True).mul(100).round(1)
    for lbl, p in pct.items():
        ok = "✓" if p >= 15 else "⚠️"
        print(f"  {lbl:<14} {p}%  {ok}")
    print(f"\nSaved {len(out)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
