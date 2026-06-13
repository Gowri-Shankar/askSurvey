"""Quick end-to-end test on 5 sample reviews — no CLI needed."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import pandas as pd
from model_loader import load_text_generation_pipeline
from classification import classify_dataframe
from preprocessing import clean_text_column

SAMPLE_REVIEWS = [
    "Great product, fast delivery and well packaged. Very happy!",
    "Terrible quality, broke after one day. Complete waste of money.",
    "Took 3 weeks to arrive and the box was crushed. Very disappointed with shipping.",
    "Exactly as described, matches the photos perfectly. Good value.",
    "Customer service was unhelpful and rude when I tried to return the item.",
]

MODEL = "microsoft/Phi-3-mini-4k-instruct"

print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"\nLoading model: {MODEL} (4-bit quantized)...")
pipe = load_text_generation_pipeline(MODEL, use_4bit=True, use_8bit=False)
print("Model loaded.\n")

df = pd.DataFrame({"text": SAMPLE_REVIEWS})
df = clean_text_column(df, "text")
result = classify_dataframe(df, pipe, review_column="text", max_new_tokens=64, temperature=0.2)

print("=" * 70)
for _, row in result.iterrows():
    print(f"Review   : {row['text'][:80]}...")
    print(f"Topic    : {row['model_pred_sub_topic']}")
    print(f"Sentiment: {row['model_pred_sentiment']}")
    print(f"Raw      : {row['model_raw_response'][:120]}")
    if row["model_error"]:
        print(f"ERROR    : {row['model_error']}")
    print("-" * 70)
