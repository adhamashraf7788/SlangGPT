<div align="center">

# 🌍 SlangGPT
### Egyptian Arabic → Modern Standard Arabic

**Fine-tuning AraGPT-2 for dialect-to-MSA generation and translation detection**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange?logo=huggingface)](https://huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/adhamashraf77/egyptian-2-arabic)

</div>

---

## Overview

SlangGPT fine-tunes **AraGPT-2** on a parallel Egyptian Arabic / Modern Standard Arabic corpus to solve two tasks:

- **Generation** — Given an Egyptian Arabic sentence, generate the equivalent MSA translation
- **Detection** — Given an (Egyptian, MSA) pair, classify whether the translation is correct

This project extends the methodology of [Hernandez & Naik (Stanford CS224N, 2025)](https://web.stanford.edu/class/cs224n/) — who adapted GPT-2 for English slang understanding — to the Arabic dialect setting, replacing the English backbone with AraGPT-2 and the slang dataset with a parallel Egyptian–MSA corpus of 18,250 sentence pairs.

---

## Results

| Task | Model | chrF | BLEU | Accuracy |
|---|---|---|---|---|
| Generation | Zero-shot AraGPT-2 | 10.62 | 0.02 | — |
| Generation | **Fine-tuned AraGPT-2** | **29.08** | **6.63** | — |
| Detection | Zero-shot AraGPT-2 | — | — | 0.500 |
| Detection | **Fine-tuned AraGPT-2** | — | — | **0.956** |

Fine-tuning improves detection accuracy by **+45.6 points** and generation chrF by **+18.5 points** over zero-shot baselines.

![Results](evaluation/plots/full_comparison.png)

### Generation Examples

| Input (Egyptian) | Zero-shot Output | Fine-tuned Output |
|---|---|---|
| يلا فين؟ | مالذي جاء به من خير... *(forum drift)* | هيا، أين أنت؟ |
| أنا محتاج أتكلم معاكي | ياام.. يآآإك ياروحيتي... *(social media drift)* | أحتاج أن أتحدث معك |
| كنت فاكرك مش جاية | يااللي ما انخطبتك... *(forum content)* | كنت أذكرك، لستِ قادمة |

---

## Dataset

**Egyptian Arabic → Modern Standard Arabic** — 18,250 parallel sentence pairs

| Split | Generation Pairs | Detection Examples |
|---|---|---|
| Train (80%) | 14,600 | 29,200 |
| Dev (10%) | 1,825 | 3,650 |
| Test (10%) | 1,825 | 3,650 |

**Download:**

| Platform | Link |
|---|---|
| 🤗 Hugging Face | [AdhamAshraf/egyptian-2-arabic](https://huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic) |
| 📦 Kaggle | [adhamashraf77/egyptian-2-arabic](https://www.kaggle.com/datasets/adhamashraf77/egyptian-2-arabic) |

<details>
<summary><b>Source & Derivation</b></summary>

This dataset is derived from [Abdalrahmankamel/Egyption_2_English](https://huggingface.co/datasets/Abdalrahmankamel/Egyption_2_English). The original dataset paired Egyptian Arabic sentences with English translations. This version repurposes the Egyptian Arabic content for Dialect → MSA conversion with the following modifications:

- Removed English translation column
- Added Modern Standard Arabic translations
- Applied Arabic normalization and diacritic (tashkeel) removal
- Reformatted for NLP dialect-to-MSA tasks

```bibtex
@dataset{ashraf2026,
  author    = {Adham Ashraf},
  title     = {Egyptian Arabic to Modern Standard Arabic Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic}
}
```

</details>

---

## Project Structure

```
SlangGPT/
├── app/                          # Flask web application
│   ├── app.py                    # Web server
│   ├── model.py                  # Model loading & inference
│   ├── templates/index.html      # Web UI
│   └── static/style.css          # Styles
│
├── data/
│   ├── prepare_data.py           # Preprocessing pipeline
│   ├── raw/NLP.csv               # Raw dataset          [git-ignored]
│   └── processed/                # Train/dev/test splits [git-ignored]
│       ├── generation_train.csv
│       ├── generation_dev.csv
│       ├── generation_test.csv
│       ├── detection_train.csv
│       ├── detection_dev.csv
│       └── detection_test.csv
│
├── model/
│   ├── config.py                 # Central config (paths + hyperparams)
│   ├── train_generation.py       # Generation training script
│   ├── train_detection.py        # Detection training script
│   └── weights/                  # Trained weights       [git-ignored]
│       ├── detection/            # best_model.pt + tokenizer
│       └── generation/best/      # model.safetensors + config
│
├── evaluation/
│   ├── baseline.py               # Zero-shot baseline evaluation
│   ├── evaluate.py               # Fine-tuned model evaluation
│   ├── error_analysis.py         # FP/FN error analysis
│   └── plots/                    # Results & figures
│       ├── baseline_results.csv
│       ├── detection_errors.csv
│       ├── generation_scores.csv
│       └── full_comparison.png
│
├── notebooks/                    # Colab training notebooks [git-ignored]
│   ├── 01_preprocessing.ipynb
│   ├── 02_train_generation.ipynb
│   └── 03_train_detection.ipynb
│
├── scripts/
│   └── download_weights.py       # Download weights from Google Drive
│
├── report/
│   ├── main.tex                  # LaTeX paper
│   └── references.bib
│
├── requirements.txt
└── .gitignore
```

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/adhamashraf7788/SlangGPT.git
cd SlangGPT
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** GPU strongly recommended. Training was done on a T4 GPU via Google Colab.

### 3. Download model weights

Weights are hosted on Google Drive (~1.9 GB total). Run:

```bash
python scripts/download_weights.py
```

This downloads and places weights at:

```
model/weights/detection/best_model.pt           (~527 MB)
model/weights/generation/best/model.safetensors  (~1.37 GB)
```

> ⚠️ If the download script fails, get the weights directly from the Google Drive link in `scripts/download_weights.py`.

### 4. Download and preprocess the dataset

```python
from datasets import load_dataset
dataset = load_dataset("AdhamAshraf/egyptian-2-arabic", split="train")
df = dataset.to_pandas()
df.to_csv("data/raw/NLP.csv", index=False, encoding="utf-8-sig")
```

Then run the preprocessing pipeline:

```bash
python data/prepare_data.py --raw_csv data/raw/NLP.csv
```

### 5. Run the web app

```bash
python app/app.py
```

Open `http://localhost:5000` — enter an Egyptian Arabic sentence to get the MSA translation and a detection confidence score.

---

## Training

Training was done on Google Colab (T4 GPU). Open the notebooks in order:

| Notebook | Description |
|---|---|
| `01_preprocessing.ipynb` | Download dataset, clean, split, build detection pairs |
| `02_train_generation.ipynb` | Fine-tune AraGPT-2 medium for generation |
| `03_train_detection.ipynb` | Fine-tune AraGPT-2 base for detection |

All notebooks mount Google Drive and save checkpoints automatically.

### Hyperparameters

| | Generation | Detection |
|---|---|---|
| Base model | aragpt2-medium | aragpt2-base |
| Parameters | ~355M | ~135M |
| Learning rate | 5e-5 | 2e-5 |
| Batch size | 8 (eff. 32) | 16 |
| LR schedule | Cosine | Linear |
| Warmup ratio | 10% | 10% |
| Weight decay | 0.01 | 0.01 |
| Epochs (actual) | 5 (early stop at 5, best at ep. 3) | 8 |
| Train loss (start → end) | 2.50 → 0.76 | 0.71 → 0.10 |

---

## Evaluation

```bash
# Zero-shot baseline
python evaluation/baseline.py

# Fine-tuned model evaluation (chrF, BLEU, PPL, accuracy)
python evaluation/evaluate.py

# Error analysis (false positives / false negatives)
python evaluation/error_analysis.py
```

Results are saved to `evaluation/plots/`.

### Detection Error Breakdown (Test Set)

| | Count | Rate |
|---|---|---|
| Total test examples | 3,650 | — |
| Correct predictions | 3,491 | 95.6% |
| False Positives | 101 | 2.8% |
| False Negatives | 58 | 1.6% |

The model is more prone to false positives — accepting an incorrect translation as correct — than false negatives. This occurs mainly on short or ambiguous Egyptian inputs where the mismatched MSA sentence is semantically plausible in isolation.

---

## Models

| Task | Base Model | Link |
|---|---|---|
| Generation | AraGPT-2 Medium | [aubmindlab/aragpt2-medium](https://huggingface.co/aubmindlab/aragpt2-medium) |
| Detection | AraGPT-2 Base | [aubmindlab/aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base) |

**Generation** uses causal language modeling with prompt masking — only the MSA target tokens contribute to the training loss. Inference uses nucleus sampling (temperature=0.7, top-k=50, top-p=0.92, repetition penalty=1.3).

**Detection** encodes a cloze-style Arabic prompt through AraGPT-2 and passes the last-token hidden state through a linear classifier head trained with binary cross-entropy.

---

## Dependencies

| Package | Version |
|---|---|
| torch | ≥ 2.0.0 |
| transformers | ≥ 4.30.0 |
| flask | ≥ 2.3.0 |
| pandas | ≥ 2.0.0 |
| numpy | ≥ 1.24.0 |
| scikit-learn | ≥ 1.2.0 |
| datasets | ≥ 2.12.0 |
| sacrebleu | ≥ 2.3.0 |
| sentencepiece | ≥ 0.2.1 |
| gdown | ≥ 5.0.0 |
| matplotlib | ≥ 3.7.0 |
| seaborn | ≥ 0.12.0 |
| tqdm | ≥ 4.65.0 |
| stanza | ≥ 1.5.0 |

---

## Related Work

This project extends:

> Hernandez & Naik, *Extending GPT-2 for Informal and Slang Aware Language Understanding*, Stanford CS224N, 2025

Which builds on:
- Antoun et al., [AraGPT2](https://arxiv.org/abs/2012.15520), 2021
- Sun et al., [Toward Informal Language Processing](https://arxiv.org/abs/2404.02323), 2024
- Radford et al., [GPT-2](https://openai.com/research/language-unsupervised), 2019

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.