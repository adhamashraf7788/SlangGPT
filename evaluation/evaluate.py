"""
Full evaluation of fine-tuned models vs zero-shot baseline.
Run: python evaluation/evaluate.py
"""

import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

import torch
import math
import pandas as pd
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score, classification_report
from app.model import (load_detection_model, load_generation_model,
                       detect, generate)

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── detection evaluation ──────────────────────────────────────────────────────
def evaluate_detection():
    print("\n── Detection ────────────────────────────────────")
    det_model, det_tok = load_detection_model()
    df = pd.read_csv(DATA_DIR / "detection_test.csv")

    preds, labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detection"):
        slang  = str(row["egyptian_arabic"]).strip()
        formal = str(row["formal_arabic"]).strip()
        label  = int(row["label"])
        result = detect(slang, formal, det_model, det_tok)
        preds.append(1 if result["label"] == "correct" else 0)
        labels.append(label)

    acc = accuracy_score(labels, preds)
    print(f"Fine-tuned Detection Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["incorrect", "correct"]))
    return acc


# ── generation evaluation ─────────────────────────────────────────────────────
def evaluate_generation():
    print("\n── Generation ───────────────────────────────────")
    gen_model, gen_tok = load_generation_model()
    df         = pd.read_csv(DATA_DIR / "generation_test.csv")
    references = df["formal_arabic"].tolist()
    hypotheses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generation"):
        slang = str(row["egyptian_arabic"]).strip()
        hypotheses.append(generate(slang, gen_model, gen_tok))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    print(f"Fine-tuned Generation BLEU : {bleu:.2f}")
    print(f"Fine-tuned Generation chrF : {chrf:.2f}")
    return bleu, chrf


# ── comparison plot ───────────────────────────────────────────────────────────
def plot_comparison(zs_det, ft_det, zs_chrf, ft_chrf, zs_bleu, ft_bleu):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for ax, title, zs, ft in [
        (axes[0], "Detection Accuracy ↑", zs_det,  ft_det),
        (axes[1], "Generation chrF ↑",    zs_chrf, ft_chrf),
        (axes[2], "Generation BLEU ↑",    zs_bleu, ft_bleu),
    ]:
        ax.bar(["Zero-shot", "Fine-tuned"], [zs, ft],
               color=["#94a3b8", "#1e3a5f"], width=0.4)
        ax.set_title(title, fontweight="bold")
        for i, v in enumerate([zs, ft]):
            ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontweight="bold")

    plt.suptitle("AraGPT-2: Zero-shot vs Fine-tuned", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "full_comparison.png", dpi=150)
    plt.show()
    print(f"Saved to evaluation/plots/full_comparison.png")


if __name__ == "__main__":
    # load baseline numbers from baseline.py output
    baseline = pd.read_csv(PLOTS_DIR / "baseline_results.csv")
    zs_det   = float(baseline[baseline["Task"].str.contains("Detection")]["Value"].values[0])
    zs_vals  = baseline[baseline["Task"].str.contains("Generation")]["Value"].values[0].split(" / ")
    zs_chrf, zs_bleu = float(zs_vals[0]), float(zs_vals[1])

    ft_det          = evaluate_detection()
    ft_bleu, ft_chrf = evaluate_generation()

    plot_comparison(zs_det, ft_det, zs_chrf, ft_chrf, zs_bleu, ft_bleu)