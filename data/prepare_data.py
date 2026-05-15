"""
Preprocessing pipeline for Egyptian Arabic slang → Formal Arabic dataset.
Handles: vocalization removal, English token filtering, train/dev/test split.
"""

import re
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────
RAW_DIR       = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ── Arabic Unicode ranges ───────────────────────────────────────────────────
ARABIC_DIACRITICS = re.compile(
    r"[\u0610-\u061A"   # Arabic extended-A diacritics
    r"\u064B-\u065F"   # Fathatan … Wavy Hamza Below  (tashkeel / vocalization)
    r"\u0670"          # Superscript Alef
    r"\u06D6-\u06DC"   # Quranic annotation signs
    r"\u06DF-\u06E4"
    r"\u06E7\u06E8"
    r"\u06EA-\u06ED]"
)

ARABIC_PUNCTUATION = re.compile(r"[،؛؟٪٫٬]")
ENGLISH_ONLY       = re.compile(r"^[A-Za-z0-9\s\.,!?'\"@#\$%\^&\*\(\)\-_=\+]+$")
WHITESPACE         = re.compile(r"\s+")


# ── cleaning helpers ────────────────────────────────────────────────────────

def remove_vocalization(text: str) -> str:
    """Strip Arabic diacritics (tashkeel) from text."""
    return ARABIC_DIACRITICS.sub("", text)


def normalize_arabic(text: str) -> str:
    """
    Light normalization:
      - Unify alef variants → bare alef
      - Unify teh marbuta variants
      - Unify waw/ya variants
      - Normalize lamalef ligatures
    """
    text = re.sub(r"[إأآا]", "ا", text)
    text = re.sub(r"ة", "ه", text)
    text = re.sub(r"ى", "ي", text)
    text = re.sub(r"ؤ", "و", text)
    text = re.sub(r"ئ", "ي", text)
    return text


def remove_extra_whitespace(text: str) -> str:
    text = text.strip()
    return WHITESPACE.sub(" ", text)


def is_mostly_english(text: str, threshold: float = 0.7) -> bool:
    """Return True if more than `threshold` fraction of tokens are ASCII-only words."""
    tokens = text.split()
    if not tokens:
        return False
    english_tokens = sum(1 for t in tokens if re.match(r"^[A-Za-z]+$", t))
    return (english_tokens / len(tokens)) >= threshold


def clean_text(text: str, normalize: bool = False) -> str:
    """Full cleaning pipeline for a single string."""
    text = remove_vocalization(text)
    text = ARABIC_PUNCTUATION.sub(" ", text)
    # keep mixed Arabic/English — just strip surrounding quotes Colab sometimes adds
    text = text.strip('"').strip("'")
    text = remove_extra_whitespace(text)
    if normalize:
        text = normalize_arabic(text)
    return text


# ── dataset-level processing ────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert "egyptian_arabic" in df.columns and "formal_arabic" in df.columns, (
        "CSV must have 'egyptian_arabic' and 'formal_arabic' columns"
    )
    return df


def preprocess_df(df: pd.DataFrame, normalize: bool = False) -> pd.DataFrame:
    df = df.copy()

    # drop rows with missing values
    df.dropna(subset=["egyptian_arabic", "formal_arabic"], inplace=True)

    # clean both columns
    df["egyptian_arabic"] = df["egyptian_arabic"].astype(str).apply(
        lambda x: clean_text(x, normalize=normalize)
    )
    df["formal_arabic"] = df["formal_arabic"].astype(str).apply(
        lambda x: clean_text(x, normalize=normalize)
    )

    # drop rows where either side became empty
    df = df[(df["egyptian_arabic"].str.len() > 0) & (df["formal_arabic"].str.len() > 0)]

    # flag and optionally drop rows that are overwhelmingly English
    df["mostly_english"] = (
        df["egyptian_arabic"].apply(is_mostly_english) |
        df["formal_arabic"].apply(is_mostly_english)
    )
    # keep them but mark — downstream you can filter if needed
    # df = df[~df["mostly_english"]]  # uncomment to hard-drop

    # reset index
    df.reset_index(drop=True, inplace=True)
    return df


def split_dataset(
    df: pd.DataFrame,
    train_ratio: float = 0.80,
    dev_ratio:   float = 0.10,
    test_ratio:  float = 0.10,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + dev_ratio + test_ratio - 1.0) < 1e-6

    train_df, temp_df = train_test_split(df, test_size=(1 - train_ratio), random_state=seed)
    relative_test = test_ratio / (dev_ratio + test_ratio)
    dev_df, test_df = train_test_split(temp_df, test_size=relative_test, random_state=seed)

    train_df = train_df.reset_index(drop=True)
    dev_df   = dev_df.reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    return train_df, dev_df, test_df


def build_detection_examples(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Build binary classification examples for slang detection.
    Positive label (1): correct (egyptian, formal) pair.
    Negative label (0): mismatched pair — formal taken from a different row.
    Returns a balanced dataframe with columns: egyptian_arabic, formal_arabic, label.
    """
    rng = np.random.default_rng(seed)
    positives = df[["egyptian_arabic", "formal_arabic"]].copy()
    positives["label"] = 1

    # hard negatives: shuffle formal column, avoid trivial matches
    shuffled_indices = rng.permutation(len(df))
    # ensure no index maps to itself
    for i in range(len(shuffled_indices)):
        if shuffled_indices[i] == i:
            swap = (i + 1) % len(shuffled_indices)
            shuffled_indices[i], shuffled_indices[swap] = shuffled_indices[swap], shuffled_indices[i]

    negatives = pd.DataFrame({
        "egyptian_arabic": df["egyptian_arabic"].values,
        "formal_arabic":   df["formal_arabic"].iloc[shuffled_indices].values,
        "label":           0,
    })

    combined = pd.concat([positives, negatives], ignore_index=True)
    combined = combined.sample(frac=1, random_state=seed).reset_index(drop=True)
    return combined


# ── main ────────────────────────────────────────────────────────────────────

def main(raw_csv: str, normalize: bool = False):
    print(f"Loading {raw_csv} ...")
    df = load_raw(raw_csv)
    print(f"  Raw rows: {len(df)}")

    df = preprocess_df(df, normalize=normalize)
    print(f"  After cleaning: {len(df)} rows  |  mostly-English flagged: {df['mostly_english'].sum()}")

    train_df, dev_df, test_df = split_dataset(df)
    print(f"  Split → train={len(train_df)}, dev={len(dev_df)}, test={len(test_df)}")

    # save generation splits (columns: egyptian_arabic, formal_arabic, mostly_english)
    train_df.to_csv(PROCESSED_DIR / "generation_train.csv", index=False, encoding="utf-8-sig")
    dev_df.to_csv  (PROCESSED_DIR / "generation_dev.csv",   index=False, encoding="utf-8-sig")
    test_df.to_csv (PROCESSED_DIR / "generation_test.csv",  index=False, encoding="utf-8-sig")
    print("  Saved train/dev/test CSVs.")

    # save detection splits
    det_train = build_detection_examples(train_df)
    det_dev   = build_detection_examples(dev_df)
    det_test  = build_detection_examples(test_df)

    det_train.to_csv(PROCESSED_DIR / "detection_train.csv", index=False, encoding="utf-8-sig")
    det_dev.to_csv  (PROCESSED_DIR / "detection_dev.csv",   index=False, encoding="utf-8-sig")
    det_test.to_csv (PROCESSED_DIR / "detection_test.csv",  index=False, encoding="utf-8-sig")
    print("  Saved detection_train/dev/test CSVs.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_csv",   required=True, help="Path to raw CSV file")
    parser.add_argument("--normalize", action="store_true", help="Apply alef/ya normalization")
    args = parser.parse_args()
    main(args.raw_csv, normalize=args.normalize)