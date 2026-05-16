import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import sys
from pathlib import Path

# ── import shared config ──────────────────────────────────────────────────────
sys.path.append(str(Path(__file__).resolve().parent.parent))
from model.config import (
    GEN_WEIGHTS_PATH,
    DET_WEIGHTS_PATH,
    GEN_PROMPT_TEMPLATE,
    DET_PROMPT_TEMPLATE,
    DETECTION_BASE_MODEL,
    GEN_TEMPERATURE,
    GEN_TOP_K,
    GEN_TOP_P,
    GEN_REPETITION_PENALTY,
    DET_MAX_SEQ_LEN,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── detection model ───────────────────────────────────────────────────────────
class SlangDetector(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.backbone   = AutoModel.from_pretrained(model_name)
        H               = self.backbone.config.hidden_size
        self.drop       = nn.Dropout(0.1)
        self.classifier = nn.Linear(H, 2)

    def forward(self, input_ids, attention_mask):
        h       = self.backbone(input_ids=input_ids,
                                attention_mask=attention_mask).last_hidden_state
        seq_len = attention_mask.sum(dim=1) - 1
        last    = h[torch.arange(h.size(0)), seq_len]
        return self.classifier(self.drop(last))


def load_detection_model():
    tokenizer = AutoTokenizer.from_pretrained(str(DET_WEIGHTS_PATH))
    model     = SlangDetector(DETECTION_BASE_MODEL)
    model.backbone.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load(DET_WEIGHTS_PATH / "best_model.pt",
                                     map_location=device))
    model.to(device).eval()
    return model, tokenizer


def load_generation_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(GEN_WEIGHTS_PATH),
            use_fast=False
        )
    except (ValueError, OSError):
        print("⚠ Generation tokenizer not found locally, loading from HuggingFace...")
        tokenizer = AutoTokenizer.from_pretrained(
            DETECTION_BASE_MODEL,
            use_fast=False
        )
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})

    model = AutoModelForCausalLM.from_pretrained(str(GEN_WEIGHTS_PATH))
    model.to(device).eval()
    return model, tokenizer

# ── inference ─────────────────────────────────────────────────────────────────
def detect(slang: str, formal: str, model, tokenizer) -> dict:
    prompt = DET_PROMPT_TEMPLATE.format(slang=slang, formal=formal)
    enc    = tokenizer(prompt, return_tensors="pt",
                       max_length=DET_MAX_SEQ_LEN, truncation=True,
                       padding="max_length").to(device)
    with torch.no_grad():
        logits = model(**enc)
    pred  = logits.argmax(-1).item()
    probs = torch.softmax(logits, dim=-1)[0].tolist()
    return {
        "label":      "correct" if pred == 1 else "incorrect",
        "confidence": round(max(probs), 4),
    }


def generate(slang: str, model, tokenizer, max_new: int = 80) -> str:
    prompt = GEN_PROMPT_TEMPLATE.format(slang=slang)
    enc    = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = model.generate(
            **enc,
            max_new_tokens=max_new,
            temperature=GEN_TEMPERATURE,
            top_k=GEN_TOP_K,
            top_p=GEN_TOP_P,
            repetition_penalty=GEN_REPETITION_PENALTY,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    out = tokenizer.decode(ids[0], skip_special_tokens=True)
    return out[len(prompt):].split("\n")[0].strip()