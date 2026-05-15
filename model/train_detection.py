"""
Fine-tune AraGPT-2 for Egyptian-slang ↔ Formal-Arabic detection (binary classification).
Uses last-token hidden state + linear head, mirroring the Stanford paper's setup.
"""

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

from config import (
    DATA_DIR, DET_SAVE_PATH,
    DETECTION_BASE_MODEL,
    DET_MAX_SEQ_LEN, DET_BATCH_SIZE,
    DET_EPOCHS, DET_LR, DET_WEIGHT_DECAY, DET_WARMUP_RATIO,
    DET_PROMPT_TEMPLATE, SEED,
)


# ── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ───────────────────────────────────────────────────────────────────

class SlangDetectionDataset(Dataset):
    def __init__(self, csv_path: str, tokenizer, max_len: int):
        df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.examples  = []

        for _, row in df.iterrows():
            slang  = str(row["egyptian_arabic"]).strip()
            formal = str(row["formal_arabic"]).strip()
            label  = int(row["label"])
            if slang and formal:
                self.examples.append((slang, formal, label))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        slang, formal, label = self.examples[idx]
        prompt = DET_PROMPT_TEMPLATE.format(slang=slang, formal=formal)
        enc = self.tokenizer(
            prompt,
            max_length=self.max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ── model ─────────────────────────────────────────────────────────────────────

class SlangDetector(nn.Module):
    """GPT-2 backbone + linear classification head on the last non-padding token."""

    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size   = self.backbone.config.hidden_size
        self.dropout  = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state  # (B, T, H)

        # take the last *real* (non-padding) token for each example
        seq_lengths = attention_mask.sum(dim=1) - 1          # (B,)
        last_hidden = hidden[torch.arange(hidden.size(0)), seq_lengths]  # (B, H)

        last_hidden = self.dropout(last_hidden)
        logits      = self.classifier(last_hidden)
        return logits


# ── training ──────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    criterion  = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return total_loss / len(loader), acc, all_preds, all_labels


def zero_shot_evaluate(model_name: str, loader, device, tokenizer):
    """Zero-shot baseline: uses raw vocab logits for yes/no tokens, no classifier head."""
    zs_model = AutoModelForCausalLM.from_pretrained(model_name).to(device).eval()

    # Arabic yes/no token ids
    yes_id = tokenizer.convert_tokens_to_ids("نعم")
    no_id  = tokenizer.convert_tokens_to_ids("لا")
    print(f"Zero-shot token IDs → 'نعم': {yes_id}  |  'لا': {no_id}")

    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Zero-shot eval"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            out      = zs_model(input_ids=input_ids, attention_mask=attention_mask)
            # last real token position for each example
            seq_len  = attention_mask.sum(dim=1) - 1
            logits   = out.logits[torch.arange(out.logits.size(0)), seq_len]  # (B, vocab)

            preds = (logits[:, yes_id] > logits[:, no_id]).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    return acc, all_preds, all_labels


def train():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(DETECTION_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = SlangDetector(DETECTION_BASE_MODEL)
    model.backbone.resize_token_embeddings(len(tokenizer))
    model.to(device)

    train_ds = SlangDetectionDataset(DATA_DIR / "detection_train.csv", tokenizer, DET_MAX_SEQ_LEN)
    dev_ds   = SlangDetectionDataset(DATA_DIR / "detection_dev.csv",   tokenizer, DET_MAX_SEQ_LEN)
    test_ds  = SlangDetectionDataset(DATA_DIR / "detection_test.csv",  tokenizer, DET_MAX_SEQ_LEN)

    train_loader = DataLoader(train_ds, batch_size=DET_BATCH_SIZE, shuffle=True,  num_workers=2)
    dev_loader   = DataLoader(dev_ds,   batch_size=DET_BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_ds,  batch_size=DET_BATCH_SIZE, shuffle=False, num_workers=2)

    optimizer    = AdamW(model.parameters(), lr=DET_LR, weight_decay=DET_WEIGHT_DECAY)
    total_steps  = len(train_loader) * DET_EPOCHS
    warmup_steps = int(total_steps * DET_WARMUP_RATIO)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion    = nn.CrossEntropyLoss()

    DET_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    best_dev_acc = 0.0
    patience     = 3
    no_improve   = 0

    for epoch in range(1, DET_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        dev_loss, dev_acc, _, _ = evaluate(model, dev_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | dev_loss={dev_loss:.4f} | dev_acc={dev_acc:.4f}")

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            no_improve   = 0
            torch.save(model.state_dict(), DET_SAVE_PATH / "best_model.pt")
            tokenizer.save_pretrained(DET_SAVE_PATH)
            print(f"  ✓ Saved best model (dev_acc={best_dev_acc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # ── zero-shot baseline (AFTER training, using separate model) ─────────────
    print("\n" + "="*45)
    print("RUNNING ZERO-SHOT BASELINE")
    print("="*45)
    
    zs_acc, zs_preds, zs_labels = zero_shot_evaluate(
        DETECTION_BASE_MODEL, test_loader, device, tokenizer
    )
    print(f"Zero-shot test accuracy: {zs_acc:.4f}")
    print(classification_report(zs_labels, zs_preds, target_names=["incorrect", "correct"]))

    # ── fine-tuned evaluation (load best model) ───────────────────────────────
    print("\n" + "="*45)
    print("RUNNING FINE-TUNED EVALUATION")
    print("="*45)
    
    model.load_state_dict(torch.load(DET_SAVE_PATH / "best_model.pt"))
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, device)
    print(f"Fine-tuned test accuracy: {test_acc:.4f}")
    print(classification_report(labels, preds, target_names=["incorrect", "correct"]))

    # ── comparison summary ────────────────────────────────────────────────────
    print("\n" + "="*45)
    print("COMPARISON SUMMARY")
    print("="*45)
    print(f"{'Model':<30} {'Test Acc':>10}")
    print("-"*45)
    print(f"{'Zero-shot AraGPT-2':<30} {zs_acc:>10.4f}")
    print(f"{'Fine-tuned AraGPT-2':<30} {test_acc:>10.4f}")
    print(f"{'Gain':<30} {test_acc - zs_acc:>+10.4f}")
    print("="*45)


if __name__ == "__main__":
    train()