# TakeMeter — Aquarium Discourse Quality Classifier

A fine-tuned DistilBERT text classifier that categorizes aquarium and fishkeeping posts from Reddit as **`analysis`** (post transfers reasoning) or **`anecdote`** (post asserts personal outcome without transferable reasoning).

---

## Community

**Aquarium and fishkeeping communities on Reddit:** r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank.

These communities produce a high volume of assertion-heavy text — care advice, fertilization recommendations, disease treatment, stocking guidance — where quality varies enormously. Some posts explain mechanisms and name methodologies; others simply assert personal outcomes. The distinction matters: a reader following anecdote-only advice about disease treatment or cycling may harm their livestock. This makes the `analysis` vs. `anecdote` boundary a meaningful and practically useful classification task.

---

## Label Taxonomy

| Label | Definition |
|---|---|
| `analysis` | The post explains a mechanism, names a technique or methodology, or gives a specific how-to recommendation with a reason. Personal framing ("I dose EI because...") is fine as long as transferable reasoning is present. |
| `anecdote` | The post is supported only by personal experience with no transferable reasoning — "works for me", "I've done this for years", vague tips with no why or how. |

**Why two labels:** The original design had three labels (`evidence_based`, `experiential`, `misinformation`). After full pipeline execution on 917 scraped posts, two structural problems emerged: (1) Reddit comments almost never produce citation-grade evidence-based content — hobbyists always use personal framing even when reasoning is present; (2) misinformation is actively suppressed by community moderation, leaving fewer than 5 clean examples in the corpus. The two-label taxonomy captures the most important real distinction in hobbyist discourse.

### Example posts

**`analysis`:**
> "Run both filters in parallel for 4–6 weeks before removing the old one — bacteria colonise the new media from the established filter, so you don't crash your cycle."

> "I dose EI (KNO3, KH2PO4, K2SO4 three times a week) because high-light tanks burn through macros fast — you'll see deficiency symptoms before you see algae if you skip doses."

**`anecdote`:**
> "I've been running my 75-gallon low-tech for six years without CO2 and my crypts, java ferns, and anubias are all thriving. Just dose Flourish once a week. Works great for me."

> "In my experience, capping your substrate with pool filter sand gives you the best of both worlds. I've had zero algae on the substrate in three years doing it this way."

---

## Dataset

**Source:** Arctic Shift public Reddit archive API (no credentials required). Scraped posts and comments from 5 subreddits across date windows from 2019–2026.

**Size:** 200 labeled examples — `analysis`: 101 (50.5%), `anecdote`: 99 (49.5%).

**Labeling process:**
1. **Scraping:** 917 posts/comments collected via Arctic Shift API; filtered client-side for self-posts, minimum score, and minimum length.
2. **Claim filter:** Groq `llama-3.1-8b-instant` applied to remove pure questions, showcases, and posts with no evaluable claim.
3. **Pre-labeling:** Groq `llama-3.1-8b-instant` assigned initial labels under the final `analysis`/`anecdote` taxonomy.
4. **AI review:** GPT-4o-mini reviewed each pre-label against detailed decision rules, confirming, correcting, or removing each example.
5. **Relabeling pass:** After taxonomy revision from 3 labels to 2, all surviving rows were relabeled by Groq `llama-3.1-8b-instant` under the final definitions.

All AI annotation is disclosed; no human review of individual labels was performed due to time constraints. Annotation decisions are documented in `planning.md` Section 8.

**Train / Val / Test split:** 88 / 19 / 20 (70% / 15% / 15%), stratified.

| Split | analysis | anecdote | Total |
|---|---|---|---|
| Train | 46 | 42 | 88 |
| Val | 10 | 9 | 19 |
| Test | 11 | 9 | 20 |

### Hard annotation cases

| Post excerpt | Decision | Reason |
|---|---|---|
| "I've been keeping discus for 12 years and they need 84-86°F stable. Fluctuations over 2°F cause immune issues." | `anecdote` | Temperature range is correct but the 2°F fluctuation threshold cites no mechanism — only personal observation. Parameters alone without a why → `anecdote`. |
| "I dose EI (KNO3/KH2PO4/K2SO4 3x/week) because high-light tanks burn through macros fast — you'll see deficiency before algae if you skip doses." | `analysis` | Personal framing but explains the mechanism (high-light macro consumption rate). Reasoning is transferable. |
| "Seachem Excel is essentially a liquid carbon source. I use it instead of CO2 and growth is fine." | `anecdote` | No mechanism given for how Excel provides carbon. "Growth is fine" is a personal outcome, not an explanation. |

---

## Fine-Tuning Pipeline

**Base model:** `distilbert-base-uncased` (HuggingFace Transformers)

**Training platform:** Google Colab (T4 GPU)

**Key training decisions:**

- **Epochs: 3.** The default starting point; with 88 training examples, more epochs risked overfitting. In practice, 3 epochs were insufficient — the model converged to predicting `anecdote` almost exclusively, suggesting it overfit to surface cues (casual tone) rather than learning the semantic distinction.
- **Learning rate: 2e-5.** Standard for DistilBERT fine-tuning; no tuning was performed.
- **Batch size: 16.** Default; with 88 training examples this means only ~5–6 gradient steps per epoch, which is very limited.

---

## Baseline Comparison

**Approach:** Each of the 20 test-set posts was sent to Groq `llama-3.3-70b-versatile` with a zero-shot system prompt defining both labels with one example each. The model was instructed to respond with only the label name (`analysis` or `anecdote`). All 20 responses were parseable.

**Prompt used:**
```
You are classifying posts from aquarium and fishkeeping communities on Reddit.

analysis: The post explains a mechanism, names a technique or method, or gives a specific
recommendation with a reason. Personal framing is fine as long as transferable reasoning is present.
Example: "Run both filters in parallel for 4–6 weeks — bacteria colonise the new media from the
established filter, so you don't crash your cycle."

anecdote: The post is backed only by personal experience with no transferable reasoning.
Example: "I've had my betta in a 5-gallon without a heater for two years and he's totally fine."

Respond with ONLY the label name — one word, lowercase, nothing else.
```

| Model | Accuracy |
|---|---|
| Zero-shot baseline (Groq llama-3.3-70b-versatile) | **0.750** |
| Fine-tuned DistilBERT | **0.600** |

Fine-tuning regressed accuracy by 0.150. The zero-shot model outperformed the fine-tuned model.

---

## Evaluation Report

### Per-class metrics

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` (baseline) | 1.00 | 0.55 | 0.71 | 11 |
| `anecdote` (baseline) | 0.64 | 1.00 | 0.78 | 9 |
| **Baseline accuracy** | | | **0.75** | 20 |
| `analysis` (fine-tuned) | 1.00 | 0.27 | 0.43 | 11 |
| `anecdote` (fine-tuned) | 0.53 | 1.00 | 0.69 | 9 |
| **Fine-tuned accuracy** | | | **0.60** | 20 |

### Confusion matrix (fine-tuned model, test set)

|  | Predicted: analysis | Predicted: anecdote |
|---|---|---|
| **True: analysis** | 3 | 8 |
| **True: anecdote** | 0 | 9 |

All 8 errors are `analysis → anecdote`. The model never incorrectly predicts `analysis`.

### Error analysis — 3 wrong predictions

**#1 — Implicit mechanism, no "because" clause**
> "get Maracyn Oxy at your local store if youre in USA put fish in quarantine tank with Maracyn Oxy and google how you should feed and quarantine your tank"
> True: `analysis` | Predicted: `anecdote` (confidence: 0.54)

This post names a specific medication and protocol — both transferable recommendations — making it `analysis`. But there is no explicit causal connector. The model appears to require "because"-style language to predict `analysis`, and treats bare imperative instructions as `anecdote`.

**#2 — Short post with implicit symptom-protocol link**
> "Check for pineconing. If there is pineconing I would isolate and possibly euthanize"
> True: `analysis` | Predicted: `anecdote` (confidence: 0.52)

The post names a specific disease symptom (pineconing = dropsy) and a clinical management protocol (isolation, euthanasia). This is transferable. But it is extremely terse and contains no "I because" framing. The model's near-random confidence (0.52) indicates genuine uncertainty. This is the hardest type for fine-tuned DistilBERT: short, jargon-containing, protocol-stating posts.

**#3 — Parameter reasoning in a corrective reply**
> "My tank sits at 40ppm a lot and sometimes gets up to 80ppm with my tetras. This isn't the smoking gun you think it is. The fact you said the ammonia went down to 0 means it was at some other level you..."
> True: `analysis` | Predicted: `anecdote` (confidence: 0.52)

This post cites specific nitrate values (40–80 ppm), interprets a parameter trend (ammonia dropping to 0 = cycle completion), and corrects a misinterpretation. It is clearly `analysis`. However, it is written as a conversational corrective reply with casual phrasing. The model cannot distinguish parameter-based reasoning embedded in casual discourse from personal anecdote.

### Failure pattern reflection

The fine-tuned model amplified the zero-shot bias rather than correcting it. Zero-shot analysis recall = 0.55; fine-tuned analysis recall = 0.27. The specific failure: **the model learned to predict `anecdote` on any post with casual/personal phrasing**, regardless of whether transferable reasoning is present.

The discriminating signal between `analysis` and `anecdote` is semantic (does the post contain a why or how), not syntactic. Reddit `analysis` posts say "I dose EI because plants need X" — the surface form is nearly identical to an anecdote. DistilBERT needed more than 88 training examples to learn this subtle pattern. The large zero-shot model outperformed fine-tuning because it has strong prior knowledge about discourse structure and what constitutes reasoning vs. assertion.

**Root cause:** Insufficient training data (88 examples) for a semantically subtle binary classification task. Data augmentation or 300+ examples would be the minimum for reliable improvement.

---

## Stretch Features

### Inter-Annotator Reliability

GPT-4o-mini was used as a second independent annotator on 40 randomly sampled examples (seed 42) from `data/dataset_labeled.csv`. The second annotator received the same label definitions and examples via a zero-shot prompt, with no access to the primary labels.

| Metric | Value |
|---|---|
| Examples compared | 40 |
| Percentage agreement | **70.0%** |
| Cohen's kappa (κ) | **0.412** |
| Kappa interpretation | Moderate |

**Disagreement breakdown:**
- Primary `analysis` → Second `anecdote`: 9 cases
- Primary `anecdote` → Second `analysis`: 3 cases

The directional asymmetry mirrors the fine-tuned model's failure pattern: the boundary between `analysis` and `anecdote` is genuinely ambiguous when reasoning is present but implicit (no "because" clause), or when a specific technique is named without explanation. The κ = 0.412 reflects real label difficulty — not annotation noise — and is consistent with the fine-tuned model's poor analysis recall (0.27). Even a larger LLM with full label definitions disagrees with the primary annotator 30% of the time on this exact class boundary.

Raw results saved in `data/inter_annotator_results.csv`.

---

### Deployed Interface

A Gradio web interface (`app.py`) that accepts a post and returns the label and confidence.

**To run:**
```bash
.venv/bin/pip install gradio
.venv/bin/python app.py
# Opens at http://127.0.0.1:7860
```

> **Note:** The project uses a `.venv` virtual environment. If running outside the venv, install gradio with `pip install gradio` into your active environment.

**Model priority:**
1. Fine-tuned DistilBERT — place model weights in `./model/` (download from Colab: zip and download the `saved_model/` directory, unzip here)
2. Groq `llama-3.3-70b-versatile` zero-shot fallback (uses `GROQ_API_KEY` from `.env`) — active when `./model/` is absent

The interface shows the label, confidence percentage, a confidence slider, and an explanation of what the label means.

---

## AI Usage

This project used AI tools throughout. All uses are disclosed below.

**1. Data collection — Arctic Shift API scraping (automated)**
Directed Groq `llama-3.1-8b-instant` to filter scraped Reddit posts for evaluable claims and remove pure questions/showcases. Overrode: manually removed automoderator bot messages that survived the claim filter (e.g., r/bettafish automod posts that link to care sheets were labeled `analysis` but are structurally noise).

**2. Pre-labeling — Groq llama-3.1-8b-instant**
Used for initial label assignment across all 200 final examples. The model pre-labeled under a compact system prompt. Override: original 3-label taxonomy collapsed to 2 labels after discovering that (a) Reddit posts don't produce citation-grade evidence-based content and (b) misinformation is suppressed by community moderation — findings the model could not have known from the prompt alone.

**3. AI review — GPT-4o-mini**
Used to review pre-labels against detailed decision rules. Override: GPT-4o-mini applied an overly strict academic citation standard to `evidence_based`, demoting posts with personal framing even when parameters were present. This was overridden by redesigning the taxonomy around the `analysis`/`anecdote` distinction, which does not require citation-grade framing.

**4. Failure analysis — GPT-4o-mini (planned)**
After collecting wrong predictions, patterns were identified by reviewing all 8 misclassified posts manually. The systematic `analysis → anecdote` collapse was confirmed as the primary failure mode.

### Spec reflection

**One way the spec helped:** The requirement to define a concrete "good enough" threshold (Section 6 of planning.md) forced a specific target before training — accuracy ≥ 70%, per-class F1 ≥ 0.65, +10pp over baseline. This made the regression immediately legible as a result rather than leaving it ambiguous whether 60% was acceptable.

**One way implementation diverged:** The spec assumes 200+ labeled examples and treats data collection as tractable manual work. In practice, the community moderation dynamics of Reddit (misinformation removed, evidence-based content written in personal framing) made the original 3-label taxonomy untrainable on scraped data. The taxonomy had to be redesigned based on what the data actually contained, not what was planned — a constraint the spec did not anticipate.
