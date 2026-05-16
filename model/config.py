"""
Central configuration for all SlangGPT models.
Edit these constants before training — no argparse soup everywhere.
"""

from pathlib import Path

# ── paths ───────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).parent.parent
DATA_DIR      = ROOT_DIR / "data" / "processed"
MODEL_DIR     = ROOT_DIR / "model" / "checkpoints"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR   = ROOT_DIR / "model" / "weights"      # ← ADD THIS

# ── base model ───────────────────────────────────────────────────────────────
# Arabic GPT-2 fine-tuned on Egyptian dialect; falls back gracefully to aubmindlab/aragpt2-base
GENERATION_BASE_MODEL = "aubmindlab/aragpt2-medium"   # ~355M params, good Arabic coverage
DETECTION_BASE_MODEL  = "aubmindlab/aragpt2-base"     # lighter for classification
# ── saved weights (populated after training) ─────────────────────────────────
GEN_WEIGHTS_PATH = WEIGHTS_DIR / "generation" / "best"   # written by train_generation.py
DET_WEIGHTS_PATH = WEIGHTS_DIR / "detection"             # written by train_detection.py

# ── generation ───────────────────────────────────────────────────────────────
GEN_MAX_INPUT_LEN  = 64    # max tokens for the Egyptian slang prompt
GEN_MAX_TARGET_LEN = 128   # max tokens for the formal Arabic output
GEN_BATCH_SIZE     = 8
GEN_GRAD_ACCUM     = 4     # effective batch = 32
GEN_EPOCHS         = 10
GEN_LR             = 5e-5
GEN_WEIGHT_DECAY   = 0.01
GEN_WARMUP_RATIO   = 0.1
GEN_SAVE_PATH      = MODEL_DIR / "generation"

# prompt template — Egyptian input goes between the markers
GEN_PROMPT_TEMPLATE = "عامية: {slang}\nفصحى: "   # "Colloquial: {slang}\nFormal: "

# decoding
GEN_TEMPERATURE    = 0.7
GEN_TOP_K          = 50
GEN_TOP_P          = 0.92
GEN_REPETITION_PENALTY = 1.3
GEN_NUM_BEAMS      = 4

# ── detection ────────────────────────────────────────────────────────────────
DET_MAX_SEQ_LEN    = 160   # prompt + both sentences
DET_BATCH_SIZE     = 16
DET_EPOCHS         = 8
DET_LR             = 2e-5
DET_WEIGHT_DECAY   = 0.01
DET_WARMUP_RATIO   = 0.1
DET_SAVE_PATH      = MODEL_DIR / "detection"

# cloze prompt for detection
# label 1 = correct translation, label 0 = incorrect
DET_PROMPT_TEMPLATE = (
    'عامية: "{slang}"\n'
    'فصحى: "{formal}"\n'
    'هل الترجمة صحيحة؟ أجب بـ "نعم" أو "لا": '
)

# ── evaluation ───────────────────────────────────────────────────────────────
EVAL_BATCH_SIZE    = 32

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42