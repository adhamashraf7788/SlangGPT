"""
Zero-shot baselines for detection and generation.
Run: python evaluation/baseline.py
"""

import torch
import math
import pandas as pd
import sacrebleu
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
PLOTS_DIR = Path(__file__).resolve().parent / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

BASE_MODEL   = "aubmindlab/aragpt2-base"
DET_PROMPT   = 'عامية: "{slang}"\nفصحى: "{formal}"\nهل الترجمة صحيحة؟ أجب بـ "نعم" أو "لا": '
GEN_PROMPT   = 'عامية: "{slang}"\nفصحى: '
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {device}")
print("Loading zero-shot model...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL).to(device).eval()


# ── detection baseline ────────────────────────────────────────────────────────
def run_detection_baseline():
    from sklearn.metrics import accuracy_score, classification_report

    df     = pd.read_csv(DATA_DIR / "detection_test.csv")
    yes_id = tokenizer.convert_tokens_to_ids("نعم")
    no_id  = tokenizer.convert_tokens_to_ids("لا")
    preds, labels = [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Detection baseline"):
        slang  = str(row["egyptian_arabic"]).strip()
        formal = str(row["formal_arabic"]).strip()
        label  = int(row["label"])
        prompt = DET_PROMPT.format(slang=slang, formal=formal)

        enc = tokenizer(prompt, return_tensors="pt",
                        max_length=160, truncation=True).to(device)
        with torch.no_grad():
            out    = model(**enc)
            seq    = enc["attention_mask"].sum(dim=1) - 1
            logits = out.logits[0, seq[0]]

        preds.append(int(logits[yes_id] > logits[no_id]))
        labels.append(label)

    acc = accuracy_score(labels, preds)
    print(f"\nZero-shot Detection Accuracy: {acc:.4f}")
    print(classification_report(labels, preds, target_names=["incorrect", "correct"]))
    return acc


# ── generation baseline ───────────────────────────────────────────────────────
def run_generation_baseline():
    df         = pd.read_csv(DATA_DIR / "generation_test.csv")
    references = df["formal_arabic"].tolist()
    hypotheses = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generation baseline"):
        slang  = str(row["egyptian_arabic"]).strip()
        prompt = GEN_PROMPT.format(slang=slang)
        enc    = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=60,
                                  temperature=0.7, top_k=50, top_p=0.92,
                                  repetition_penalty=1.3, do_sample=True,
                                  pad_token_id=tokenizer.pad_token_id)
        out = tokenizer.decode(ids[0], skip_special_tokens=True)
        hypotheses.append(out[len(prompt):].split("\n")[0].strip())

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    ppl  = math.exp(sum(
        model(tokenizer(GEN_PROMPT.format(slang=str(r["egyptian_arabic"])) + str(r["formal_arabic"]),
                        return_tensors="pt", max_length=128, truncation=True,
                        padding="max_length").input_ids.to(device),
               labels=tokenizer(GEN_PROMPT.format(slang=str(r["egyptian_arabic"])) + str(r["formal_arabic"]),
                                return_tensors="pt", max_length=128, truncation=True,
                                padding="max_length").input_ids.to(device)).loss.item()
        for _, r in df.iterrows()
    ) / len(df))

    print(f"\nZero-shot Generation:")
    print(f"  BLEU : {bleu:.2f}")
    print(f"  chrF : {chrf:.2f}")
    print(f"  PPL  : {ppl:.2f}")
    return bleu, chrf, ppl


if __name__ == "__main__":
    det_acc = run_detection_baseline()
    bleu, chrf, ppl = run_generation_baseline()

    results = pd.DataFrame({
        "Task":    ["Detection (Zero-shot)", "Generation (Zero-shot)"],
        "Metric":  ["Accuracy",              "chrF / BLEU / PPL"],
        "Value":   [f"{det_acc:.4f}",        f"{chrf:.2f} / {bleu:.2f} / {ppl:.1f}"],
    })
    results.to_csv(PLOTS_DIR / "baseline_results.csv", index=False)
    print("\nSaved to evaluation/plots/baseline_results.csv")