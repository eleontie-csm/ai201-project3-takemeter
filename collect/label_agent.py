"""
label_agent.py — Groq-powered agent for two tasks:
  1. FILTER: decide if a post contains an evaluable claim (not a pure question/showcase)
  2. LABEL:  assign one of {analysis, anecdote}

Uses llama-3.1-8b-instant (higher free-tier token budget than 70b).
Compact system prompt to minimise tokens per call (~150 tokens vs ~730 before).
Handles 429 rate-limit errors with backoff and supports checkpoint resume.
"""

import json
import time
import re
import os
from groq import Groq

# ---------------------------------------------------------------------------
# Compact system prompt — ~150 tokens (was ~730)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You label aquarium/fishkeeping posts. Return ONLY a JSON object, no prose.

Labels (pick exactly one when not_a_claim=false):
  analysis   – post explains a mechanism, names a technique/method, or gives a specific
               how-to recommendation with a reason. Personal framing ("I dose EI because...")
               is fine as long as transferable reasoning is present.
  anecdote   – claim backed only by personal experience with no transferable reasoning;
               "works for me", "I've done this for years", vague tips with no why.

Set not_a_claim=true for pure questions, photo showcases, or posts with no evaluable claim.
Set needs_review=true if the post sits on a label boundary.

JSON keys: not_a_claim (bool), label (str), confidence (high|medium|low), reasoning (1 sentence), needs_review (bool)"""


# Use the lighter model — faster and higher free-tier token budget
MODEL = "llama-3.1-8b-instant"

VALID_LABELS = {"analysis", "anecdote"}


def _call_groq(client: Groq, post_text: str, retries: int = 3) -> dict:
    """Send one post to Groq. Handles 429 backoff. Returns parsed JSON dict."""
    # Truncate to 400 chars — compact prompt + short text keeps each call ~250 tokens
    truncated = post_text[:400] + ("..." if len(post_text) > 400 else "")

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"POST:\n{truncated}"},
                ],
                temperature=0.1,
                max_tokens=120,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown code fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw.strip())
            result = json.loads(raw)
            # Validate label field
            if not result.get("not_a_claim") and result.get("label") not in VALID_LABELS:
                result["label"] = "anecdote"
                result["needs_review"] = True
                result["_label_coerced"] = True
            return result

        except json.JSONDecodeError:
            if attempt < retries - 1:
                time.sleep(2)
                continue
            return {
                "not_a_claim": False, "label": "_parse_error",
                "confidence": "low", "reasoning": "JSON parse error",
                "needs_review": True, "_parse_error": True,
            }
        except Exception as exc:
            err_str = str(exc)
            # 429 — extract wait time from error message and backoff
            if "429" in err_str:
                import re as _re
                m = _re.search(r"try again in (\d+)m([\d.]+)s", err_str)
                if m:
                    wait = int(m.group(1)) * 60 + float(m.group(2)) + 5
                else:
                    wait = 70  # default 70s backoff
                print(f"\n  [agent] 429 rate limit — waiting {wait:.0f}s before retry ...")
                time.sleep(wait)
                continue  # retry after backoff (don't count against retries)
            if attempt < retries - 1:
                time.sleep(4)
                continue
            return {
                "not_a_claim": False, "label": "_api_error",
                "confidence": "low", "reasoning": f"API error: {err_str[:120]}",
                "needs_review": True, "_api_error": True,
            }


def label_posts(
    posts: list[dict],
    api_key: str,
    delay: float = 1.5,
    verbose: bool = True,
    checkpoint_path: str | None = None,
) -> list[dict]:
    """
    Run the Groq label agent over every post.
    Returns only posts that are evaluable claims (not_a_claim=False).

    Args:
        posts:           list of dicts from reddit.py / forum.py
        api_key:         Groq API key
        delay:           seconds between API calls (default 1.5s = 40 RPM)
        verbose:         print progress
        checkpoint_path: if set, skip posts whose 'id' already appears in this CSV
    """
    client = Groq(api_key=api_key)
    labeled: list[dict] = []

    # ── Resume from checkpoint ─────────────────────────────────────────────
    already_done: set[str] = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        import pandas as pd
        ck = pd.read_csv(checkpoint_path)
        # Rows without API/parse errors are considered done
        done_rows = ck[~ck["label"].isin(["_api_error", "_parse_error"])]
        already_done = set(done_rows["id"].astype(str)) if "id" in done_rows.columns else set()
        if already_done:
            print(f"  [agent] checkpoint: skipping {len(already_done)} already-labeled posts")
            # Re-add done claim rows to output
            for _, row in done_rows.iterrows():
                if row["label"] not in ("_filtered_not_a_claim",):
                    labeled.append(row.to_dict())

    total = len(posts)
    processed = 0
    for i, post in enumerate(posts, 1):
        pid = str(post.get("id", ""))
        if pid in already_done:
            continue

        if verbose and processed % 10 == 0:
            print(f"  [agent] labeling {i}/{total} (processed this run: {processed}) ...")

        result = _call_groq(client, post["text"])
        processed += 1
        time.sleep(delay)

        post["_raw_agent"] = result

        if result.get("not_a_claim", False):
            post["label"] = "_filtered_not_a_claim"
            post["confidence"] = "n/a"
            post["reasoning"] = result.get("reasoning", "")
            post["needs_review"] = False
            continue

        post["label"] = result.get("label", "_api_error")
        post["confidence"] = result.get("confidence", "low")
        post["reasoning"] = result.get("reasoning", "")
        post["needs_review"] = result.get("needs_review", True)
        labeled.append(post)

    if verbose:
        _print_summary(labeled, total)

    return labeled


def _print_summary(labeled: list[dict], total_scraped: int) -> None:
    counts: dict[str, int] = {}
    needs_review = 0
    errors = 0
    for p in labeled:
        lbl = p.get("label", "?")
        counts[lbl] = counts.get(lbl, 0) + 1
        if p.get("needs_review"):
            needs_review += 1
        if str(lbl).startswith("_"):
            errors += 1

    print("\n  ── Agent labeling summary ──────────────────────────")
    print(f"  Total input:       {total_scraped}")
    print(f"  Kept as claims:    {len(labeled)}")
    print(f"  Filtered (not claim): {total_scraped - len(labeled)}")
    for label, n in sorted(counts.items()):
        pct = 100 * n / len(labeled) if labeled else 0
        flag = " ← ERROR (needs re-run)" if str(label).startswith("_") else ""
        print(f"    {label:<22} {n:>4}  ({pct:.1f}%){flag}")
    print(f"  Flagged needs_review: {needs_review}")
    if errors:
        print(f"  ⚠️  Error rows: {errors} — re-run to retry these")
    print("  ────────────────────────────────────────────────────\n")
