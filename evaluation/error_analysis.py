"""
Error analysis — finds failure cases for detection and generation.
Run: python evaluation/error_analysis.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from app.model import (load_detection_model, load_generation_model,
                       detect, generate)

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


# ── detection errors ──────────────────────────────────────────────────────────
def detection_errors():
    print("\n── Detection Error Analysis ─────────────────────")
    det_model, det_tok = load_detection_model()
    df = pd.read_csv(DATA_DIR / "detection_test.csv")

    errors = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detection errors"):
        slang  = str(row["egyptian_arabic"]).strip()
        formal = str(row["formal_arabic"]).strip()
        label  = int(row["label"])
        result = detect(slang, formal, det_model, det_tok)
        pred   = 1 if result["label"] == "correct" else 0

        if pred != label:
            errors.append({
                "egyptian":   slang,
                "formal":     formal,
                "gold":       "correct" if label == 1 else "incorrect",
                "predicted":  result["label"],
                "confidence": result["confidence"],
            })

    df_errors = pd.DataFrame(errors)
    df_errors.to_csv(PLOTS_DIR / "detection_errors.csv", index=False)

    print(f"Total errors   : {len(errors)} / {len(df)} ({len(errors)/len(df)*100:.1f}%)")
    print(f"\nFalse positives (predicted correct, actually wrong): "
          f"{len(df_errors[df_errors['predicted']=='correct'])}")
    print(f"False negatives (predicted wrong, actually correct): "
          f"{len(df_errors[df_errors['predicted']=='incorrect'])}")
    print(f"\nSample errors:")
    print(df_errors.head(5).to_string(index=False))
    print(f"\nSaved to evaluation/plots/detection_errors.csv")


# ── generation errors ─────────────────────────────────────────────────────────
def generation_errors():
    print("\n── Generation Error Analysis ────────────────────")
    import sacrebleu
    gen_model, gen_tok = load_generation_model()
    df = pd.read_csv(DATA_DIR / "generation_test.csv")

    rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generation errors"):
        slang     = str(row["egyptian_arabic"]).strip()
        reference = str(row["formal_arabic"]).strip()
        hypothesis = generate(slang, gen_model, gen_tok)
        chrf = sacrebleu.sentence_chrf(hypothesis, [reference]).score

        rows.append({
            "egyptian":   slang,
            "reference":  reference,
            "generated":  hypothesis,
            "chrF":       round(chrf, 2),
        })

    df_gen = pd.DataFrame(rows).sort_values("chrF")
    df_gen.to_csv(PLOTS_DIR / "generation_scores.csv", index=False)

    print(f"Mean chrF  : {df_gen['chrF'].mean():.2f}")
    print(f"Median chrF: {df_gen['chrF'].median():.2f}")
    print(f"\nWorst 5 examples:")
    print(df_gen.head(5).to_string(index=False))
    print(f"\nBest 5 examples:")
    print(df_gen.tail(5).to_string(index=False))
    print(f"\nSaved to evaluation/plots/generation_scores.csv")


if __name__ == "__main__":
    detection_errors()
    generation_errors()