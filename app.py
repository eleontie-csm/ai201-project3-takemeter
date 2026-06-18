"""
app.py — TakeMeter Gradio interface.

Classifies aquarium/fishkeeping posts as 'analysis' or 'anecdote'.
Uses the fine-tuned DistilBERT model if available locally; falls back to
Groq llama-3.3-70b-versatile (zero-shot) if model weights are not present.

Setup:
    pip install gradio transformers torch
    # Optionally: place fine-tuned model in ./model/ (download from Colab)
    python3 app.py

Fine-tuned model download (from Colab):
    from google.colab import files
    # Zip and download the saved_model directory from your Colab notebook,
    # then unzip into ./model/ in this repo root.
"""

import os
import json
import re
from pathlib import Path

from dotenv import load_dotenv
import gradio as gr

load_dotenv(Path(__file__).parent / ".env")

MODEL_DIR = Path(__file__).parent / "model"
LABELS = ["analysis", "anecdote"]

# ── Load fine-tuned model (if available) ──────────────────────────────────────
_pipeline = None

def _load_pipeline():
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    if MODEL_DIR.exists():
        try:
            from transformers import pipeline
            _pipeline = pipeline(
                "text-classification",
                model=str(MODEL_DIR),
                tokenizer=str(MODEL_DIR),
                return_all_scores=True,
            )
            print(f"Loaded fine-tuned DistilBERT from {MODEL_DIR}")
            return _pipeline
        except Exception as e:
            print(f"Could not load fine-tuned model: {e}")
    return None


def classify_distilbert(text: str) -> tuple[str, float, str]:
    """Returns (label, confidence, source)."""
    pipe = _load_pipeline()
    if pipe is None:
        return None, None, None
    scores = pipe(text[:512])[0]
    best = max(scores, key=lambda x: x["score"])
    label = best["label"].lower()
    if label not in LABELS:
        label = LABELS[int(best["label"].split("_")[-1])]
    return label, best["score"], "fine-tuned DistilBERT"


# ── Groq zero-shot fallback ────────────────────────────────────────────────────
GROQ_SYSTEM = """You are classifying posts from aquarium and fishkeeping communities on Reddit
(r/PlantedTank, r/Aquascape, r/Aquariums, r/bettafish, r/shrimptank).

analysis: The post explains a mechanism, names a technique or method, or gives a specific
recommendation with a reason. Personal framing is fine as long as transferable reasoning is present.
Example: "Run both filters in parallel for 4-6 weeks — bacteria colonise the new media so you
don't crash your cycle."

anecdote: The post is backed only by personal experience with no transferable reasoning.
"Works for me", "I've done this for years", vague tips with no why or how.
Example: "I've had my betta in a 5-gallon without a heater for two years and he's totally fine."

Respond with ONLY a JSON object: {"label": "analysis" or "anecdote", "confidence": "high"/"medium"/"low"}"""


def classify_groq(text: str) -> tuple[str, float, str]:
    from groq import Groq
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        return "anecdote", 0.5, "error (no GROQ_API_KEY)"
    client = Groq(api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": GROQ_SYSTEM},
                {"role": "user", "content": f"Post:\n{text[:800]}"},
            ],
            temperature=0.0,
            max_tokens=40,
        )
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.DOTALL).strip()
        obj = json.loads(raw)
        label = obj.get("label", "anecdote").lower()
        if label not in LABELS:
            label = "anecdote"
        conf_map = {"high": 0.90, "medium": 0.72, "low": 0.55}
        conf = conf_map.get(obj.get("confidence", "medium"), 0.72)
        return label, conf, "Groq llama-3.3-70b-versatile (zero-shot)"
    except Exception as e:
        return "anecdote", 0.5, f"error: {str(e)[:60]}"


# ── Main classify function ─────────────────────────────────────────────────────
LABEL_DESCRIPTIONS = {
    "analysis": "This post explains a mechanism, names a technique, or gives a recommendation with a reason. The reasoning is transferable to other aquarium keepers.",
    "anecdote": "This post is backed only by personal experience with no transferable reasoning — 'works for me' without explaining why or how.",
}

LABEL_EMOJI = {"analysis": "🔬", "anecdote": "💬"}


def classify(text: str) -> tuple:
    text = text.strip()
    if not text:
        return "—", "Please enter a post.", "", 0.0

    label, conf, source = classify_distilbert(text)
    if label is None:
        label, conf, source = classify_groq(text)

    conf_pct = f"{conf * 100:.1f}%"
    emoji = LABEL_EMOJI.get(label, "")
    description = LABEL_DESCRIPTIONS.get(label, "")
    detail = f"**Model:** {source}\n\n**Confidence:** {conf_pct}\n\n**What this means:** {description}"

    return f"{emoji} `{label}`", conf_pct, detail, conf


# ── Example posts ─────────────────────────────────────────────────────────────
EXAMPLES = [
    ["Run both filters in parallel for 4–6 weeks before removing the old one — bacteria colonise the new media from the established filter, so you don't crash your cycle."],
    ["I've been keeping bettas for 5 years and I've never had a problem with a 5-gallon unheated tank. They're tougher than people say."],
    ["CO2 should target 20–30 ppm in a planted tank. Verify with a drop checker using 4dKH reference solution — green = ~30 ppm, yellow = overshooting, blue = too low."],
    ["I dose Flourish once a week and do a 30% water change every Sunday. Works great for me, plants are thriving."],
    ["get Maracyn Oxy at your local store, put the fish in a quarantine tank and follow the dosing instructions on the package — it's the most reliable treatment for columnaris."],
]


# ── Gradio UI ─────────────────────────────────────────────────────────────────
model_status = f"Fine-tuned DistilBERT ({MODEL_DIR})" if MODEL_DIR.exists() else "Groq llama-3.3-70b-versatile (zero-shot fallback — place fine-tuned model in ./model/ to use it)"

with gr.Blocks(title="TakeMeter") as demo:
    gr.Markdown(
        f"""
# 🐠 TakeMeter — Aquarium Discourse Classifier

Classifies aquarium and fishkeeping posts as **analysis** (post transfers reasoning)
or **anecdote** (post asserts personal outcome without transferable reasoning).

**Active model:** {model_status}
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Post text",
                placeholder="Paste a Reddit post or comment from an aquarium community...",
                lines=6,
            )
            classify_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_out = gr.Markdown(label="Label")
            conf_out = gr.Textbox(label="Confidence", interactive=False)
            conf_bar = gr.Slider(minimum=0, maximum=1, value=0, label="Confidence", interactive=False)
            detail_out = gr.Markdown(label="Details")

    gr.Examples(
        examples=EXAMPLES,
        inputs=text_input,
        label="Example posts",
    )

    classify_btn.click(
        fn=classify,
        inputs=text_input,
        outputs=[label_out, conf_out, detail_out, conf_bar],
    )
    text_input.submit(
        fn=classify,
        inputs=text_input,
        outputs=[label_out, conf_out, detail_out, conf_bar],
    )

if __name__ == "__main__":
    demo.launch(share=False)
