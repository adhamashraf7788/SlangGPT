"""
Fine-tune AraGPT-2 for Egyptian-slang → Formal-Arabic generation.
Uses causal LM with prompt masking (only the formal target contributes to loss).
"""

import os
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

from config import (
    DATA_DIR, GEN_SAVE_PATH,
    GENERATION_BASE_MODEL,
    GEN_MAX_INPUT_LEN, GEN_MAX_TARGET_LEN,
    GEN_BATCH_SIZE, GEN_GRAD_ACCUM,
    GEN_EPOCHS, GEN_LR, GEN_WEIGHT_DECAY, GEN_WARMUP_RATIO,
    GEN_PROMPT_TEMPLATE, SEED,
)


# ── reproducibility ──────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── dataset ──────────────────────────────────────────────────────────────────

class SlangGenerationDataset(Dataset):
    """
    Each example is a full sequence:
        <prompt> <formal_target> <eos>
    Loss is masked on the prompt portion — only the target tokens are trained.
    """

    def __init__(self, csv_path: str, tokenizer, max_input_len: int, max_target_len: int):
        df = pd.read_csv(csv_path)
        self.tokenizer     = tokenizer
        self.max_input_len  = max_input_len
        self.max_target_len = max_target_len
        self.examples = []

        for _, row in df.iterrows():
            slang  = str(row["egyptian_arabic"]).strip()
            formal = str(row["formal_arabic"]).strip()
            if slang and formal:
                self.examples.append((slang, formal))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        slang, formal = self.examples[idx]
        prompt = GEN_PROMPT_TEMPLATE.format(slang=slang)

        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(
            formal + self.tokenizer.eos_token, add_special_tokens=False
        )

        # truncate if needed
        prompt_ids = prompt_ids[-self.max_input_len:]
        target_ids = target_ids[:self.max_target_len]

        input_ids  = prompt_ids + target_ids
        # mask prompt tokens in labels with -100
        labels     = [-100] * len(prompt_ids) + target_ids

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels":    torch.tensor(labels,    dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int):
    """Left-pad all sequences to the same length within a batch."""
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids_list, labels_list, attn_list = [], [], []

    for item in batch:
        seq_len   = item["input_ids"].size(0)
        pad_len   = max_len - seq_len
        input_ids = torch.cat([
            torch.full((pad_len,), pad_token_id, dtype=torch.long),
            item["input_ids"]
        ])
        labels = torch.cat([
            torch.full((pad_len,), -100, dtype=torch.long),
            item["labels"]
        ])
        attn_mask = torch.cat([
            torch.zeros(pad_len,  dtype=torch.long),
            torch.ones(seq_len,   dtype=torch.long),
        ])
        input_ids_list.append(input_ids)
        labels_list.append(labels)
        attn_list.append(attn_mask)

    return {
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_list),
        "labels":         torch.stack(labels_list),
    }


# ── training loop ─────────────────────────────────────────────────────────────

def train(resume_from: str = None):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── tokenizer & model ───────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(GENERATION_BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        tokenizer.pad_token = "<pad>"

    model = AutoModelForCausalLM.from_pretrained(GENERATION_BASE_MODEL)
    model.resize_token_embeddings(len(tokenizer))

    if resume_from:
        state = torch.load(resume_from, map_location=device)
        model.load_state_dict(state)
        print(f"Resumed from {resume_from}")

    model.to(device)

    # ── data ────────────────────────────────────────────────────────────────
    train_ds = SlangGenerationDataset(
        DATA_DIR / "train.csv", tokenizer, GEN_MAX_INPUT_LEN, GEN_MAX_TARGET_LEN
    )
    dev_ds = SlangGenerationDataset(
        DATA_DIR / "dev.csv", tokenizer, GEN_MAX_INPUT_LEN, GEN_MAX_TARGET_LEN
    )

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)

    train_loader = DataLoader(train_ds, batch_size=GEN_BATCH_SIZE, shuffle=True,
                              collate_fn=_collate, num_workers=2, pin_memory=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=GEN_BATCH_SIZE, shuffle=False,
                              collate_fn=_collate, num_workers=2, pin_memory=True)

    # ── optimizer & scheduler ───────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=GEN_LR, weight_decay=GEN_WEIGHT_DECAY)
    total_steps   = (len(train_loader) // GEN_GRAD_ACCUM) * GEN_EPOCHS
    warmup_steps  = int(total_steps * GEN_WARMUP_RATIO)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ── training ────────────────────────────────────────────────────────────
    GEN_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    best_dev_loss = float("inf")
    no_improve    = 0
    patience      = 3

    for epoch in range(1, GEN_EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train")):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss    = outputs.loss / GEN_GRAD_ACCUM

            loss.backward()
            total_train_loss += outputs.loss.item()

            if (step + 1) % GEN_GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train = total_train_loss / len(train_loader)

        # ── validation ──────────────────────────────────────────────────────
        model.eval()
        total_dev_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(dev_loader, desc=f"Epoch {epoch} dev"):
                input_ids      = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels         = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                total_dev_loss += outputs.loss.item()

        avg_dev = total_dev_loss / len(dev_loader)
        print(f"Epoch {epoch:02d} | train_loss={avg_train:.4f} | dev_loss={avg_dev:.4f}")

        if avg_dev < best_dev_loss:
            best_dev_loss = avg_dev
            no_improve    = 0
            model.save_pretrained(GEN_SAVE_PATH / "best")
            tokenizer.save_pretrained(GEN_SAVE_PATH / "best")
            print(f"  ✓ Saved best model (dev_loss={best_dev_loss:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print("Early stopping triggered.")
                break

    # always save final checkpoint too
    model.save_pretrained(GEN_SAVE_PATH / "final")
    tokenizer.save_pretrained(GEN_SAVE_PATH / "final")
    print("Training complete.")


# ── zero-shot comparison ──────────────────────────────────────────────────────

def zero_shot_compare(fine_tuned_path: str, test_phrases: list):
    """Compare zero-shot vs fine-tuned generation side by side."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load zero-shot (base model, no fine-tuning)
    zs_tok = AutoTokenizer.from_pretrained(GENERATION_BASE_MODEL)
    if zs_tok.pad_token is None:
        zs_tok.add_special_tokens({"pad_token": "<pad>"})
    zs_mdl = AutoModelForCausalLM.from_pretrained(GENERATION_BASE_MODEL).to(device).eval()

    # load fine-tuned
    ft_tok = AutoTokenizer.from_pretrained(fine_tuned_path)
    ft_mdl = AutoModelForCausalLM.from_pretrained(fine_tuned_path).to(device).eval()

    def generate(model, tokenizer, phrase):
        prompt  = GEN_PROMPT_TEMPLATE.format(slang=phrase)
        enc     = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out_ids = model.generate(
                **enc, max_new_tokens=60,
                temperature=0.7, top_k=50, top_p=0.92,
                repetition_penalty=1.3, do_sample=True,
                pad_token_id=tokenizer.pad_token_id
            )
        generated = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        return generated[len(prompt):].split("\n")[0].strip()

    print(f"\n{'Input':<30} {'Zero-shot':<35} {'Fine-tuned':<35}")
    print("-" * 100)
    for phrase in test_phrases:
        zs_out = generate(zs_mdl, zs_tok, phrase)
        ft_out = generate(ft_mdl, ft_tok, phrase)
        print(f"{phrase:<30} {zs_out[:33]:<35} {ft_out[:33]:<35}")


if __name__ == "__main__":
    train()
    zero_shot_compare(
        fine_tuned_path=str(GEN_SAVE_PATH / "best"),
        test_phrases=[
            "يلا فين؟",
            "أنا محتاج أتكلم معاكي",
            "كنت فاكرك مش جاية",
        ]
    )