"""
SlangGPT Flask Application
Egyptian Arabic Slang ↔ Formal Arabic — Detection & Generation
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from pathlib import Path
import os

app = Flask(__name__)

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parent.parent
WEIGHTS_DIR  = BASE_DIR / "model" / "weights"
BASE_MODEL   = "aubmindlab/aragpt2-base"
DET_PROMPT   = 'عامية: "{slang}"\nفصحى: "{formal}"\nهل الترجمة صحيحة؟ أجب بـ "نعم" أو "لا": '
GEN_PROMPT   = 'عامية: "{slang}"\nفصحى: '
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def load_models():
    global det_model, det_tok, gen_model, gen_tok

    det_path = WEIGHTS_DIR / "detection"
    gen_path = WEIGHTS_DIR / "generation" / "best"
    # detection
    det_tok   = AutoTokenizer.from_pretrained(str(det_path))
    det_model = SlangDetector(BASE_MODEL)
    det_model.backbone.resize_token_embeddings(len(det_tok))
    det_model.load_state_dict(
        torch.load(det_path / "best_model.pt", map_location=device)
    )
    det_model.to(device).eval()

    # generation — use_fast=False avoids sentencepiece requirement
    try:
        gen_tok = AutoTokenizer.from_pretrained(
            str(gen_path), use_fast=False
        )
    except (ValueError, OSError):
        print("⚠ Falling back to base model tokenizer...")
        gen_tok = AutoTokenizer.from_pretrained(
            BASE_MODEL, use_fast=False
        )
    if gen_tok.pad_token is None:
        gen_tok.add_special_tokens({"pad_token": "<pad>"})

    gen_model = AutoModelForCausalLM.from_pretrained(str(gen_path))
    gen_model.to(device).eval()

    print(f"✓ Models loaded on {device}")


# ── inference ─────────────────────────────────────────────────────────────────
def run_detect(slang: str, formal: str) -> dict:
    prompt = DET_PROMPT.format(slang=slang, formal=formal)
    enc    = det_tok(prompt, return_tensors="pt", max_length=160,
                     truncation=True, padding="max_length").to(device)
    with torch.no_grad():
        logits = det_model(enc["input_ids"], enc["attention_mask"])
    probs  = torch.softmax(logits, dim=-1)[0].tolist()
    pred   = int(logits.argmax(-1).item())
    return {
        "label":      "صحيحة" if pred == 1 else "غير صحيحة",
        "correct":    pred == 1,
        "confidence": round(max(probs) * 100, 1),
    }


def run_generate(slang: str) -> str:
    prompt = GEN_PROMPT.format(slang=slang)
    enc    = gen_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = gen_model.generate(
            **enc, max_new_tokens=80,
            temperature=0.7, top_k=50, top_p=0.92,
            repetition_penalty=1.3, do_sample=True,
            pad_token_id=gen_tok.pad_token_id,
        )
    out = gen_tok.decode(ids[0], skip_special_tokens=True)
    return out[len(prompt):].split("\n")[0].strip()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/detect", methods=["POST"])
def detect():
    data   = request.get_json()
    slang  = data.get("slang", "").strip()
    formal = data.get("formal", "").strip()
    if not slang or not formal:
        return jsonify({"error": "يرجى إدخال النص العامي والفصيح"}), 400
    result = run_detect(slang, formal)
    return jsonify(result)


@app.route("/generate", methods=["POST"])
def generate():
    data  = request.get_json()
    slang = data.get("slang", "").strip()
    if not slang:
        return jsonify({"error": "يرجى إدخال النص العامي"}), 400
    formal = run_generate(slang)
    return jsonify({"formal": formal})


@app.route("/health")
def health():
    return jsonify({"status": "ok", "device": str(device)})


# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_models()
    app.run(debug=False, host="0.0.0.0", port=5000)