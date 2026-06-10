# Task 5: Auto Tagging Support Tickets Using LLM

## 🎯 Objective
Automatically tag customer support tickets into predefined categories using Large Language Models (LLMs). The project compares two approaches — **Zero-Shot classification** using BART-MNLI and **Few-Shot prompting** using Flan-T5 — to identify the top 3 most probable tags per ticket.

---

## 📂 Dataset
- **Type:** Synthetic free-text support ticket dataset
- **Size:** 25 tickets across 5 categories
- **Categories:**
  - `billing` — Payment issues, invoices, refunds, charges
  - `technical_issue` — Bugs, crashes, errors, API problems
  - `account_management` — Password resets, user roles, profile changes
  - `product_feedback` — Feature requests, UI/UX suggestions
  - `shipping_delivery` — Delivery status, wrong items, address changes
- **Format:** Free-text tickets with ground truth labels

---

## 🛠️ Methodology / Approach

### 1. Dataset Creation
- Created 25 realistic support tickets (5 per category)
- Each ticket mimics real-world customer language and issues
- Dataset stored as a pandas DataFrame — no external download needed

### 2. Zero-Shot Classification (BART-MNLI)
- Model: `facebook/bart-large-mnli` (~1.6GB)
- Loaded via Hugging Face `pipeline("zero-shot-classification")`
- No training examples provided — model classifies based on category descriptions alone
- Hypothesis template: `"This support ticket is about {}."`
- Returns top 3 tags with confidence scores per ticket

### 3. Few-Shot Classification (Flan-T5)
- Model: `google/flan-t5-base` (~300MB)
- 5 hand-crafted examples provided in the prompt (one per category)
- Model generates comma-separated tag list
- Output parsed and validated against known category list
- Pads to 3 tags if fewer than 3 are returned

### 4. Evaluation
- Top-1 Accuracy: exact match of predicted tag vs true label
- Top-3 Hit Rate: whether true label appears in top 3 predictions
- Per-class F1, precision, recall via `classification_report`
- Comparison table: Zero-Shot vs Few-Shot across all metrics

### 5. Visualizations
- Bar chart: Zero-Shot vs Few-Shot performance comparison
- Confusion matrix: Few-Shot predictions
- Confidence score distribution: Zero-Shot model

---

## 📊 Key Results

| Metric          | Zero-Shot (BART) | Few-Shot (Flan-T5) |
|-----------------|-----------------|-------------------|
| Top-1 Accuracy  | **80%** (20/25) | 72% (18/25)       |
| Top-3 Hit Rate  | ~96%            | ~92%              |

### Per-Category Performance (Zero-Shot)
| Category           | Result        |
|--------------------|---------------|
| billing            | 4/5 correct ✅ |
| technical_issue    | 5/5 correct ✅ |
| account_management | 4/5 correct ✅ |
| product_feedback   | 1/5 correct ⚠️ |
| shipping_delivery  | 5/5 correct ✅ |

---

## 🔍 Key Observations
- **Zero-Shot BART outperformed Few-Shot Flan-T5** (80% vs 72%) — larger model size compensates for lack of examples
- `product_feedback` was the hardest category — frequently confused with `technical_issue` due to overlapping vocabulary (e.g. "slow", "not working", "add feature")
- Top-3 hit rate is very high for both models — the true label almost always appears in the top 3 predictions
- Few-shot examples improved billing and shipping accuracy but hurt product_feedback
- **No model training required** — both approaches work entirely at inference time via prompt engineering
- Zero-Shot returns calibrated confidence scores; Few-Shot does not

---

## 🧰 Tools & Libraries
| Tool | Purpose |
|------|---------|
| `transformers` | BART-MNLI and Flan-T5 models |
| `torch` | Deep learning backend |
| `scikit-learn` | Evaluation metrics |
| `pandas` | Data manipulation |
| `matplotlib` | Visualizations |
| `seaborn` | Confusion matrix heatmap |
| `numpy` | Numerical operations |

---

## 📁 Repository Structure
```
task5-auto-tagging-support-tickets/
│
├── task5_auto_tagging.ipynb          # Main notebook
├── zero_shot_results.csv             # Zero-shot predictions output
├── few_shot_results.csv              # Few-shot predictions output
├── eda_plots.png                     # Category distribution charts
├── evaluation_plots.png              # Performance comparison + confusion matrix
└── README.md
```

---

## ▶️ How to Run
```bash
# 1. Install dependencies
pip install transformers torch scikit-learn pandas matplotlib seaborn numpy

# 2. Launch Jupyter and run notebook top to bottom
jupyter notebook task5_auto_tagging.ipynb
```

> **Note:** Models download automatically from Hugging Face on first run.
> - `facebook/bart-large-mnli` — ~1.6 GB (Cell 5)
> - `google/flan-t5-base` — ~300 MB (Cell 8)
> Ensure at least 2.5 GB free disk space before running.

---

## 🚀 Skills Demonstrated
- Prompt engineering for LLM-based classification
- Zero-shot learning with BART-MNLI
- Few-shot learning with Flan-T5
- Multi-label prediction and ranking (Top-3 tags)
- LLM evaluation and comparison

By Asra Khalid
DHC-ID: 1017
