# TakeMeter — Planning Document

## 1. Community

**Chosen community:** Planted freshwater aquarium enthusiasts, primarily from [r/PlantedTank](https://www.reddit.com/r/PlantedTank/), [r/Aquascape](https://www.reddit.com/r/Aquascape/), and the [Planted Tank forums](https://www.plantedtank.net/forums/).

**Why this community:** Planted tank hobbyists produce a high volume of assertion-heavy text — care advice, fertilization recommendations, CO2 guidance, substrate opinions — where the quality of information varies enormously. Some posts cite established parameters and recognized aquarium science; others rely entirely on personal anecdote; others actively spread myths that can harm livestock and plants (e.g., "fish waste is enough fertilizer," "Excel is equivalent to CO2 injection"). The distinction between these three tiers of advice quality is one that experienced hobbyists recognize immediately and care about, because bad advice costs money and lives of animals.

**Why it fits a classification task:** The community is text-heavy, active, and public. Top-level posts and comment threads regularly contain evaluable claims (not just photos or one-word reactions), and the three label categories correspond to natural patterns in how advice is presented in this space.

**Data collection scope:** Only top-level posts and comments that make an evaluable claim (assertion, recommendation, explanation, correction) will be collected. Pure questions, showcase posts, and link-only posts are excluded at collection time.

**Actual collection (post-run update):** Automated pipeline collected posts across 10 subreddits (r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank, r/Goldfish, r/Cichlid, r/FishTank, r/Aquascape, r/shrimptank — multiple date windows) via the Arctic Shift public Reddit archive API. After filtering non-claims (questions, showcases, one-line reactions), Groq pre-labeling, GPT-4o-mini review, and final taxonomy consolidation, the dataset is **200 examples** with two labels.

---

## 2. Label Taxonomy

### Labels

| Label | Definition |
|---|---|
| `analysis` | The post explains a mechanism, names a technique or methodology, or gives a specific how-to recommendation with a reason. Personal framing ("I dose EI because plants need macros") is fine as long as reasoning is present and transferable to another keeper. |
| `anecdote` | The claim is supported only by personal experience with no transferable reasoning — "it works for me", "I've done this for years", vague endorsements. No mechanism, no named technique, no specific parameter with a why. |

**Why two labels (design decision):** The initial taxonomy included `evidence_based`, `experiential`, and `misinformation`. After full pipeline execution on 917 scraped posts: (1) Reddit comments almost never produce citation-grade evidence-based content — hobbyists always use personal framing; (2) misinformation is actively suppressed by moderation and downvoting, leaving fewer than 5 clean examples in the scraped corpus. The two-label taxonomy captures the most important real distinction in hobbyist discourse: posts that transfer reasoning vs. posts that assert personal outcomes.

---

### Examples per Label

#### `analysis`

**Example 1:**
> "CO2 should target 20–30 ppm in a planted tank. The easiest way to verify this is with a drop checker using 4dKH reference solution — green = ~30 ppm, yellow = overshooting, blue = too low. Don't rely on bubble count since it's pressure-dependent."

*Rationale: Explains a parameter range, a measurable verification method, and why an alternative is unreliable. Reasoning is transferable regardless of who the author is.*

**Example 2:**
> "I dose EI (KNO3, KH2PO4, K2SO4 three times a week) because high-light tanks consume macros faster than low-tech systems — if you don't replenish them, deficiency symptoms appear even before you see algae."

*Rationale: Personal framing ("I dose") but explains the mechanism. A reader can evaluate and apply the logic.*

---

#### `anecdote`

**Example 1:**
> "I've been running my 75-gallon low-tech for six years without CO2 and my crypts, java ferns, and anubias are all thriving. Just dose Flourish once a week and do regular water changes. Works great for me."

*Rationale: Only support is personal success over time. No mechanism or reasoning a new keeper can evaluate or reuse.*

**Example 2:**
> "In my experience, capping your substrate with a thin layer of pool filter sand gives you the best of both worlds. I've had zero algae on the substrate in three years doing it this way."

*Rationale: A technique is named but the only justification is the author's three-year outcome. No explanation of why it works.*

---

## 3. Hard Edge Cases

### Primary ambiguous boundary: `analysis` vs. `anecdote`

**Example ambiguous post:**
> "I've kept discus for 12 years and the single most important factor is temperature — they need stable 84–86°F. In my tanks I've found that fluctuations beyond 2°F within a day cause stress and immune suppression. Don't let anyone tell you 78°F is fine."

**Why it's ambiguous:** The author cites specific temperature parameters and a threshold. These look like analysis. But the sole support is personal observation over 12 years with no mechanism explained.

**Decision rule:** If a post contains specific parameters *and* explains a mechanism or names a methodology, label it `analysis` — even with personal framing. If parameters appear but the only justification is "I've observed this," with no causal reasoning, label it `anecdote`. The discus post → **`anecdote`**, because no mechanism is given for why 2°F fluctuation causes immune suppression.

---

### Secondary ambiguous case: technique named but no reason given

**Example ambiguous post:**
> "Just use the Excel spot-treatment method — works every time for BBA."

**Why it's ambiguous:** A named technique is referenced (Excel spot-treatment) but no reason is given for why it works.

**Decision rule:** Naming a technique alone is not enough for `analysis`. The post must include at least a brief why or how. → **`anecdote`**. If the post said "Excel (glutaraldehyde) works on BBA because it disrupts the algae cell membrane at high local concentration," that would be `analysis`.

---

## 4. Data Collection Plan

### Sources

| Source | Type | Expected label mix |
|---|---|---|
| r/PlantedTank — Top posts (past year) | Varied | All three labels |
| r/Aquascape — Top posts | Technique-heavy | Mostly `evidence_based` / `experiential` |
| plantedtank.net forums — Care guides, technique threads | Long-form | All three labels, older myths common |

### Collection strategy

- Collect **240 posts** targeting 80 per label (buffer for rejects during review)
- Filter at collection time: skip posts that are pure questions, showcase-only, or contain no evaluable claim
- For `misinformation`, actively search for myth-heavy content using queries like:
  - `"fish waste is enough"`, `"Excel is the same as CO2"`, `"fish grow to tank size"`, `"don't need to cycle"`, `"gravel is fine for planted tanks"`
- Use Reddit's Top/Year sort for higher-quality (upvoted) `evidence_based` examples
- Use Reddit's Controversial sort and older forum threads for `misinformation` candidates

### Actual collection results (post-run update)

Automated pipeline collected 917 posts/comments across 5 subreddits via the Arctic Shift public Reddit archive API. After Groq filtering and rebalancing, the final dataset is **178 examples**.

| Label | Count | % |
|---|---|---|
| `analysis` | 67 | 52.8% |
| `anecdote` | 60 | 47.2% |
| **Total** | **200** | |

### Why the taxonomy was reduced from 3 labels to 2 — and why that's an honest finding

The original 3-label taxonomy (`evidence_based` / `experiential` / `misinformation`) was revised after full pipeline execution. **Two structural problems emerged:**

1. **`evidence_based` is not present in Reddit discourse at scale.** Hobbyists consistently use personal framing even when citing correct parameters. After GPT-4o-mini review, only 8–13 of 81 pre-labeled posts survived as genuinely evidence-based. The new `analysis` label captures the same intent without requiring citation-grade framing.
2. **`misinformation` is suppressed by community moderation.** Targeted myth-keyword scraping found fewer than 5 clean examples across the full 917-post corpus. Moderation, downvoting, and corrective replies remove bad advice before it reaches the public archive. The remaining borderline posts were reclassified as `anecdote` by the relabeling pass.

The 2-label taxonomy is not a compromise — it reflects what aquarium discourse on Reddit actually contains.

### CSV format

`data/dataset_labeled.csv` — 200 rows, two labels. Pre-labeled by Groq `llama-3.1-8b-instant`, reviewed by GPT-4o-mini, relabeled under the final `analysis`/`anecdote` taxonomy by Groq. The notebook handles the 70/15/15 train/val/test split automatically.

---

## 5. Evaluation Metrics

### Primary metrics

| Metric | Why |
|---|---|
| **Per-class F1** | Most important. With two roughly balanced classes (~53%/47%), F1 catches when the model is defaulting to the majority class. |
| **Confusion matrix** | Shows directional errors — which direction the model confuses analysis↔anecdote and how often. |
| **Overall accuracy** | Required by spec; also interpretable for binary classification (50% = random baseline). |

### Secondary metrics (reported)

| Metric | Why |
|---|---|
| **Precision per class** | Tells us whether the model is over-predicting `analysis` (flagging anecdotes as analysis). |
| **Recall per class** | Tells us whether the model is missing `analysis` examples — anecdote-heavy predictions would reduce its utility as a content quality filter. |

### Why accuracy alone is insufficient

With two classes at ~53%/47%, a model that always predicts `analysis` would achieve 53% accuracy but 0% recall on `anecdote`. Per-class F1 reveals this. Binary classification also makes the confusion matrix small and easy to interpret — every wrong prediction falls into exactly one of two error types.

---

## 6. Definition of Success

A classifier is genuinely useful for this community if:

- **Fine-tuned model accuracy ≥ 70%** on the test set (significantly above 50% random baseline for binary)
- **Per-class F1 ≥ 0.65 for both labels** — the model is not defaulting to the majority class
- **Fine-tuned model beats zero-shot baseline by ≥ 10 percentage points** in overall accuracy, demonstrating that community-specific fine-tuning added real value

A result would be "good enough to deploy" as a post-quality filter if all three criteria are met. The classifier could be used to surface `analysis` posts in community wikis or to flag `anecdote`-only responses in high-stakes threads (disease treatment, breeding, equipment safety).

---

## 7. Baseline Results (Zero-Shot, Groq llama-3.3-70b-versatile)

**Approach:** Each test-set post was sent to `llama-3.3-70b-versatile` with the zero-shot system prompt below. The model was instructed to respond with only the label name (`analysis` or `anecdote`). All 20 test posts produced parseable responses.

**System prompt used:**
```
You are classifying posts from aquarium and fishkeeping communities on Reddit
(r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank).

analysis: The post explains a mechanism, names a technique or method, or gives a
specific recommendation with a reason. Personal framing is fine as long as
transferable reasoning is present.
Example: "Run both filters in parallel for 4–6 weeks before removing the old one —
bacteria colonise the new media from the established filter, so you don't crash your cycle."

anecdote: The post is backed only by personal experience with no transferable reasoning —
"works for me", "I've done this for years", vague tips with no explanation of why.
Example: "I've had my betta in a 5-gallon without a heater for two years and he's totally fine."

Respond with ONLY the label name — one word, lowercase, nothing else.
```

**Results on 20-example test set:**

| | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `analysis` | 1.00 | 0.55 | 0.71 | 11 |
| `anecdote` | 0.64 | 1.00 | 0.78 | 9 |
| **accuracy** | | | **0.75** | **20** |
| macro avg | 0.82 | 0.77 | 0.74 | 20 |

**Reflection:**

The zero-shot model achieves 75% accuracy — well above the 50% random baseline, showing the LLM has substantial prior knowledge about what constitutes reasoning vs. anecdote in discourse. However, a clear bias emerges: the model never predicts `analysis` when uncertain (precision = 1.00, recall = 0.55). It defaulted to `anecdote` on 5 of the 11 true `analysis` posts.

This is the expected zero-shot failure mode: casual Reddit text — even when it contains reasoning — reads as conversational and personal ("I dose because..."), which surface-level cues push the LLM toward `anecdote`. The model lacks the community-specific calibration to recognise that mechanism-bearing language in first-person framing should still be `analysis`. Fine-tuning on 88 community-specific examples should teach that distinction and push analysis recall above 0.80.

---

## 8. AI Tool Plan

### Label stress-testing (before annotation)

Provide the label definitions and edge case rules above to an LLM (Claude or GPT-4) and ask it to generate 10 posts that sit at the `evidence_based`/`experiential` boundary and 10 that sit at the `experiential`/`misinformation` boundary. If any generated post cannot be cleanly labeled using the decision rules in Section 3, tighten the definitions before starting annotation.

### Annotation assistance (during data collection)

Option: Paste batches of 20–30 unlabeled posts into an LLM with the label definitions and ask for pre-labels before reviewing manually. If used:
- Every pre-assigned label must be individually reviewed and either confirmed or corrected
- Track which examples were pre-labeled in the `notes` column of the CSV
- Disclose use in the README AI usage section

Decision: Will attempt manual labeling for the first 60 examples to calibrate personal judgment, then decide whether pre-labeling accelerates the remaining 140 without sacrificing consistency.

### Failure analysis (after fine-tuning)

After collecting wrong predictions from the test set, paste all misclassified examples into an LLM and prompt: *"Identify any systematic patterns in these misclassified posts — look for common post length, sarcasm, hedging language, specific topic areas, or label pairs that are repeatedly confused."* Then verify each identified pattern by re-reading the examples manually. Include both confirmed patterns and patterns the LLM suggested that turned out not to hold.

---

## 8. Hard Annotation Decisions Log

### Systematic hard case: posts with technical vocabulary that are still questions

During automated collection, 45 posts were flagged and removed post-hoc because they started with a question opener but contained aquarium-specific parameters (CO2 ppm, KH/GH values, fertilizer names). The agent labeled these with confident labels because of the technical vocabulary, even though the posts were seeking advice rather than providing it.

**Decision rule applied:** A post is a claim only if it asserts a relationship or recommendation. Asking "What should my CO2 be?" with the word "ppm" in the body is not a claim — it is a request. Removed from dataset.

### Three hard individual cases under the final taxonomy

| # | Post excerpt | Could be | Decision | Reason |
|---|---|---|---|---|
| 1 | "I've been keeping discus for 12 years and they need 84-86°F stable. Fluctuations over 2°F cause immune issues." | `analysis` (cites specific parameters) OR `anecdote` (sole support is 12 years personal observation) | `anecdote` | The temperature range is correct but the 2°F fluctuation threshold cites no mechanism — only personal observation. Parameters alone without a why → `anecdote`. |
| 2 | "I dose EI (KNO3/KH2PO4/K2SO4 3x/week) because high-light tanks burn through macros fast — you'll see deficiency before algae if you skip doses." | `analysis` (names method + explains mechanism) OR `anecdote` (personal practice) | `analysis` | Personal framing but explains the mechanism (high-light macro consumption rate). The reasoning is transferable. |
| 3 | "Seachem Excel is essentially a liquid carbon source. I use it instead of CO2 and growth is fine." | `analysis` (makes a specific claim about chemistry) OR `anecdote` (personal outcome, no mechanism) | `anecdote` | No mechanism is given for how Excel provides carbon. "Growth is fine" is a personal outcome, not an explanation. |

---

## 10. Fine-Tuned Model Results and Error Analysis

### Results summary

| | Zero-shot baseline | Fine-tuned DistilBERT |
|---|---|---|
| Accuracy | 0.75 | **0.60** |
| analysis F1 | 0.71 | **0.43** |
| anecdote F1 | 0.78 | **0.69** |
| macro F1 | 0.74 | 0.56 |

Fine-tuning made performance *worse* across all metrics. The model did not meet any of the success criteria defined in Section 6.

### Confusion matrix (test set, n=20)

|  | Predicted: analysis | Predicted: anecdote |
|---|---|---|
| **True: analysis** | 3 | 8 |
| **True: anecdote** | 0 | 9 |

All 8 errors are `analysis → anecdote` misclassifications. The model never predicts `analysis` when it is wrong — it only predicts `analysis` with high confidence on 3 posts, and defaults to `anecdote` on everything else (recall 0.27 on analysis).

### Three wrong prediction analyses

**#1 — Implicit mechanism, no explicit "because"**
> "get Maracyn Oxy at your local store if youre in USA put fish in quarantine tank with Maracyn Oxy and google how you should feed and quarantine your tank"
> True: `analysis` | Predicted: `anecdote` (confidence: 0.54)

This post names a specific medication (Maracyn Oxy) and a specific protocol (quarantine), both of which constitute transferable recommendations. However, the reasoning is implicit — there is no "because" clause. The model appears to have learned that explicit causal connectors mark `analysis`, and treats posts with bare imperative instructions as `anecdote`.

**#2 — Low confidence, ambiguous short post**
> "Check for pineconing. If there is pineconing I would isolate and possibly euthanize"
> True: `analysis` | Predicted: `anecdote` (confidence: 0.52)

This is a genuinely hard case: the post names a symptom (pineconing = dropsy, a bacterial infection) and a specific management protocol. But it is extremely short and gives no mechanism. The model's near-random confidence (0.52) is appropriate — the post sits right at the label boundary.

**#3 — Specific parameters embedded in a corrective reply**
> "My tank sits at 40ppm a lot and sometimes gets up to 80ppm with my tetras. This isn't the smoking gun you think it is. The fact you said the ammonia went down to 0 means it was at some other level you..."
> True: `analysis` | Predicted: `anecdote` (confidence: 0.52)

This post cites specific nitrate values (40–80 ppm), interprets a parameter trend (ammonia dropping to 0 = cycle completion), and corrects a misinterpretation. It is clearly `analysis`. However, it is written as a conversational reply with no explicit methodology. The model appears unable to distinguish parameter-based reasoning embedded in casual discourse from personal anecdote.

### Systematic failure pattern

**All 8 errors are `analysis → anecdote`.** The fine-tuned model amplified the zero-shot bias rather than correcting it. Zero-shot: analysis recall = 0.55. Fine-tuned: analysis recall = 0.27.

Root causes:

1. **Training set is too small (88 examples).** DistilBERT fine-tuning on 88 examples is at the lower bound of reliable convergence. The model did not have enough signal to learn that first-person framing + reasoning = `analysis`, and instead overfit to surface cues (casual tone, no explicit "because") that associate with `anecdote`.

2. **`analysis` posts on Reddit look like `anecdote` on the surface.** The key discriminating feature — presence of transferable reasoning — is semantic and subtle, not lexical. Reddit `analysis` posts say "I dose EI because plants need X" not "EI dosing works because plants need X." The surface form is nearly identical to anecdote, and 88 examples may not be enough for DistilBERT to learn the deeper pattern.

3. **One training example is a bot post (r/bettafish automod message).** Post #6 in the wrong predictions is a Reddit automoderator message containing a link to a care sheet. It was labeled `analysis` (the linked care sheet is evidence-based content) but its text is structurally unlike all other posts. Including this in training likely added noise.

4. **The `analysis` class may have higher within-class variance.** `Anecdote` posts share strong surface features ("I", "me", "my tank", "years"). `Analysis` posts range from terse protocol statements ("quarantine with Maracyn Oxy") to parameter-heavy explanations to mechanism-bearing replies. This variance is harder to learn from a small dataset.

### Reflection on the gap between intended and actual

The classifier was designed to distinguish posts that transfer reasoning from posts that only assert personal outcomes. The fine-tuned model did not learn this distinction — it learned only to predict `anecdote` with slightly varying confidence. The gap is not primarily a labeling quality problem: the labels were applied consistently. It is a data quantity problem combined with a task difficulty problem: the discriminating signal is semantic (does the post contain a why or how), not syntactic, and DistilBERT needed more than 88 examples to learn it.

The zero-shot LLM outperformed the fine-tuned model because large language models have strong prior knowledge about discourse structure and what constitutes reasoning vs. assertion. Fine-tuning on 88 community-specific examples was not enough to overcome DistilBERT's smaller representational capacity.

**What would help:** (1) More training data — 300–500 examples would be the minimum for reliable binary fine-tuning on this task. (2) Class-weighted loss to penalise `analysis` misses more heavily. (3) Data augmentation — paraphrase `analysis` examples without the personal framing to reduce surface-form overlap with `anecdote`.

---

## 9. Stretch Features

*(Update this section before starting any stretch feature)*

| Feature | Status | Notes |
|---|---|---|
| Inter-annotator reliability | Not started | Would need a second annotator for 30+ examples |
| Confidence calibration | Not started | |
| Error pattern analysis | Not started | |
| Deployed interface | Not started | |
