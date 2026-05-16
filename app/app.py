"""
SlangGPT Flask Application
Egyptian Arabic → Formal Arabic Generation
"""

from flask import Flask, render_template, request, jsonify
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path

app = Flask(__name__)

# ── config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = BASE_DIR / "model" / "weights"
GEN_PROMPT  = 'عامية: "{slang}"\nفصحى: '
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gen_model = None
gen_tok   = None


# ── model loading ─────────────────────────────────────────────────────────────
def load_models():
    global gen_model, gen_tok

    gen_path = WEIGHTS_DIR / "generation" / "best"

    try:
        gen_tok = AutoTokenizer.from_pretrained(str(gen_path), use_fast=False)
    except (ValueError, OSError):
        print("⚠ Falling back to base model tokenizer...")
        gen_tok = AutoTokenizer.from_pretrained(
            "aubmindlab/aragpt2-base", use_fast=False
        )

    if gen_tok.pad_token is None:
        gen_tok.add_special_tokens({"pad_token": "<pad>"})

    gen_model = AutoModelForCausalLM.from_pretrained(str(gen_path))
    gen_model.to(device).eval()

    print(f"✓ Generation model loaded on {device}")


# ── inference ─────────────────────────────────────────────────────────────────
def run_generate(slang: str) -> str:
    prompt = GEN_PROMPT.format(slang=slang)
    enc    = gen_tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        ids = gen_model.generate(
            **enc,
            max_new_tokens=80,
            temperature=0.7,
            top_k=50,
            top_p=0.92,
            repetition_penalty=1.3,
            do_sample=True,
            pad_token_id=gen_tok.pad_token_id,
        )
    out = gen_tok.decode(ids[0], skip_special_tokens=True)
    return out[len(prompt):].split("\n")[0].strip()


# ── routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


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