<div align="center">

# SlangGPT
### Egyptian Arabic → Modern Standard Arabic

**Fine-tuning AraGPT-2 for dialect-to-MSA generation and translation detection**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

| Resource | Link |
|---|---|
| 🤗 Live Demo | [SlangGPT Space](https://huggingface.co/spaces/AdhamAshraf/SlangGPT) |
| 📄 Paper | [SlangGPT Report](https://github.com/adhamashraf7788/SlangGPT/blob/main/report/SlangGPT_report.pdf) |
| 📦 Training Dataset | [egyptian-2-arabic](https://huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic) |
| 💬 Feedback Dataset | [slanggpt-feedback-dataset](https://huggingface.co/datasets/AdhamAshraf/slanggpt-feedback-dataset) |
| 📦 Kaggle | [egyptian-2-arabic](https://www.kaggle.com/datasets/adhamashraf77/egyptian-2-arabic) |

</div>

---

## Motivation

Over 100 million Egyptians speak Egyptian Arabic daily — yet most Arabic NLP systems are trained almost entirely on Modern Standard Arabic (MSA). Egyptian Arabic is not simply a simplified form of MSA; it carries distinct vocabulary, grammar, and culturally loaded expressions that MSA-trained models consistently fail to handle. Words like **جدع** carry meanings of honor, loyalty, and social standing that have no direct MSA equivalent, yet they appear constantly in everyday Egyptian speech.

When we set out to build SlangGPT, we discovered there was no large, clean, publicly available Egyptian Arabic ↔ MSA parallel dataset suitable for this task. So we built one ourselves.

---

## Overview

SlangGPT fine-tunes **AraGPT-2** on an 18,250-sentence parallel Egyptian Arabic / MSA corpus to solve two tasks:

- **Generation** — Given an Egyptian Arabic sentence, generate the equivalent MSA translation
- **Detection** — Given an (Egyptian, MSA) pair, classify whether the translation is correct

This project extends the methodology of [Hernandez & Naik (Stanford CS224N, 2025)](https://drive.google.com/file/d/1jBi3u2VPgi4NS4pmXZb7D3E5c9aJRQ_8/view?usp=drive_link) — who fine-tuned GPT-2 for Gen-Z English slang understanding — to the Arabic dialect setting, replacing the English backbone with AraGPT-2 and the slang dataset with a parallel Egyptian–MSA corpus.

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

## Known Limitations

- **Generation quality is still limited.** 18K sentence pairs is relatively small for a full translation task. BLEU of 6.63, while a significant improvement over zero-shot, indicates the model is not yet production-ready for translation.
- **AraGPT-2 was not pretrained on dialectal Arabic.** The backbone was trained on MSA and online Arabic text, which creates an inherent ceiling on how well it can generate fluent dialectal-to-MSA translations without a much larger fine-tuning corpus.
- **Culturally loaded words are hard to translate.** Terms like جدع, أوي, or خلاص carry meanings that are difficult to capture in MSA even for humans. The model often produces technically correct but culturally flat translations.
- **Dataset coverage is uneven.** The 18K corpus skews toward conversational sentences and may not generalize well to domain-specific Egyptian Arabic (e.g., medical, legal, or technical speech).

We deployed the demo publicly with a community feedback loop specifically to address these limitations over time.

---

## Community Feedback & Retraining

The live demo includes a feedback system where users can rate translations and submit corrections. Every submission is automatically saved to the [open feedback dataset](https://huggingface.co/datasets/AdhamAshraf/slanggpt-feedback-dataset) and will be used for periodic retraining to improve generation quality continuously.

```python
from datasets import load_dataset
df = load_dataset("AdhamAshraf/slanggpt-feedback-dataset", split="train").to_pandas()

# High quality confirmed translations
high_quality = df[df["user_rating"] >= 4]

# Human corrections for fine-tuning
corrections = df[df["corrected_msa"] != ""]
```

---

## Datasets

### Training Dataset — egyptian-2-arabic

18,250 parallel Egyptian Arabic / MSA sentence pairs used to train both the generation and detection models.

| Split | Generation Pairs | Detection Examples |
|---|---|---|
| Train (80%) | 14,600 | 29,200 |
| Dev (10%) | 1,825 | 3,650 |
| Test (10%) | 1,825 | 3,650 |

🔗 [huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic](https://huggingface.co/datasets/AdhamAshraf/egyptian-2-arabic)

```python
from datasets import load_dataset
dataset = load_dataset("AdhamAshraf/egyptian-2-arabic", split="train")
df = dataset.to_pandas()
```

<details>
<summary><b>Source & Derivation</b></summary>

Derived from [Abdalrahmankamel/Egyptian_2_English](https://huggingface.co/datasets/Abdalrahmankamel/Egyption_2_English). Modifications:
- Removed English translation column
- Added Modern Standard Arabic translations
- Applied Arabic normalization and diacritic removal
- Reformatted for dialect-to-MSA tasks

</details>

### Feedback Dataset — slanggpt-feedback-dataset

Human feedback collected from the live Space. Users rate model translations and provide corrections, forming a growing dataset for future fine-tuning.

| Field | Type | Description |
|---|---|---|
| `egyptian_arabic` | string | Original Egyptian Arabic input |
| `generated_msa` | string | SlangGPT's generated translation |
| `user_label` | string | `correct` or `incorrect` |
| `user_rating` | int64 | Quality score 0–5 |
| `corrected_msa` | string | Human correction (required if incorrect or rating ≤ 2) |
| `timestamp` | string | ISO 8601 UTC timestamp |

---

## Quickstart

> **Note:** GPU required. CPU inference is too slow for practical use. Training was done on a T4 GPU via Google Colab.

### 1. Clone and install

```bash
git clone https://github.com/adhamashraf7788/SlangGPT.git
cd SlangGPT
pip install -r requirements.txt
```

### 2. Download model weights

```bash
python scripts/download_weights.py
```

Downloads weights to:
```
model/weights/detection/best_model.pt            (~527 MB)
model/weights/generation/best/model.safetensors  (~1.37 GB)
```

### 3. Download and preprocess the dataset

```python
from datasets import load_dataset
dataset = load_dataset("AdhamAshraf/egyptian-2-arabic", split="train")
df = dataset.to_pandas()
df.to_csv("data/raw/NLP.csv", index=False, encoding="utf-8-sig")
```

```bash
python data/prepare_data.py --raw_csv data/raw/NLP.csv
```

### 4. Run the web app

```bash
python app/app.py
```

Open `http://localhost:5000`

---

## Training

Training was done on Google Colab (T4 GPU). Open the notebooks in order:

| Notebook | Description |
|---|---|
| `01_preprocessing.ipynb` | Download dataset, clean, split, build detection pairs |
| `02_train_generation.ipynb` | Fine-tune AraGPT-2 medium for generation |
| `03_train_detection.ipynb` | Fine-tune AraGPT-2 base for detection |

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
| Epochs | 5 (best at ep. 3) | 8 |
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

### Detection Error Breakdown (Test Set)

| | Count | Rate |
|---|---|---|
| Total test examples | 3,650 | — |
| Correct predictions | 3,491 | 95.6% |
| False Positives | 101 | 2.8% |
| False Negatives | 58 | 1.6% |

---

## Models

| Task | Base Model | Link |
|---|---|---|
| Generation | AraGPT-2 Medium | [aubmindlab/aragpt2-medium](https://huggingface.co/aubmindlab/aragpt2-medium) |
| Detection | AraGPT-2 Base | [aubmindlab/aragpt2-base](https://huggingface.co/aubmindlab/aragpt2-base) |

**Generation** uses causal language modeling with prompt masking — only the MSA target tokens contribute to the training loss. Inference uses greedy decoding with repetition penalty.

**Detection** encodes a cloze-style Arabic prompt through AraGPT-2 and passes the last-token hidden state through a linear classifier head trained with binary cross-entropy.

---

## Project Structure

```
SlangGPT/
├── app/                          # Flask web application
│   ├── app.py
│   ├── model.py
│   ├── templates/index.html
│   └── static/style.css
├── data/
│   ├── prepare_data.py
│   ├── raw/NLP.csv               # [git-ignored]
│   └── processed/                # [git-ignored]
├── model/
│   ├── config.py
│   ├── train_generation.py
│   ├── train_detection.py
│   └── weights/                  # [git-ignored]
├── evaluation/
│   ├── baseline.py
│   ├── evaluate.py
│   ├── error_analysis.py
│   └── plots/
├── notebooks/                    # [git-ignored]
├── scripts/
│   └── download_weights.py
├── report/
│   ├── main.tex
│   └── references.bib
├── requirements.txt
└── .gitignore
```

---

## Related Work

This project extends:

> Hernandez & Naik, *Extending GPT-2 for Informal and Slang Aware Language Understanding*, Stanford CS224N, 2025

Which builds on:
- Antoun et al., [AraGPT2](https://arxiv.org/abs/2012.15520), 2021
- Sun et al., [Toward Informal Language Processing](https://arxiv.org/abs/2404.02323), 2024
- Radford et al., [GPT-2](https://openai.com/research/language-unsupervised), 2019

---

## Citation

```bibtex
@misc{slanggpt2026,
  title={SlangGPT: Fine-tuning AraGPT-2 for Egyptian Arabic to MSA Generation and Detection},
  author={Abdelrahman Ahmed and Adham Ashraf and Ahmed Fekry},
  year={2026},
  url={https://github.com/adhamashraf7788/SlangGPT}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
