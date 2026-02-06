# MGSIR: Multi-Granular Structural Invariant Representation 🛡️

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://www.tensorflow.org/) [![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-green)](https://xgboost.readthedocs.io/) [![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

> **MGSIR** is a robust SQL injection detection framework based on Multi-Granular Structural Invariant Features (MGSIF).

[English](README.md) | [中文文档](README_CN.md)

---

## 📖 Introduction

**MGSIR** addresses the critical challenge of adversarial robustness in machine learning-based Web Application Firewalls (WAFs).

Our solution introduces:

- **Multi-Granular Structural Invariant Features (MGSIF)** – a 23-dimensional feature set designed to capture invariant attack patterns across four abstraction levels.
- **XGBoost** as the core classifier for high throughput and industrial-grade efficiency.
- **A broad set of baselines** including shallow NLP (TF-IDF, Word2Vec) and deep learning models (BERT, TextCNN, LSTM).
- **Robustness evaluation** on 8 types of obfuscated SQLi attacks.

This framework achieves state-of-the-art detection performance and superior robustness against obfuscated attacks.

---

## 🏗️ Project Structure

The project follows a modular architecture for reproducibility and scalability:

```text
MGSIR-SQLi/
├── data/                         # Data Management
│   └── dataset1/                 # Default Dataset
│       ├── raw/                  # Raw SQLi Dataset
│       │   └── All_SQL_Dataset.csv
│       ├── processed/            # Cleaned & Split Data (Train/Val/Test)
│       └── adversarial/          # 8 Sets of Adversarial / Obfuscated Samples
│           ├── test_adv_randomcase.csv
│           ├── test_adv_space2comment.csv
│           └── ... (and 6 others)
│
├── results/                      # Experiment Results
│   ├── checkpoints/              # Model Artifacts (Weights, Scalers, Tokenizers)
│   │   ├── mgsir_xgb/  # Core Method Artifacts (MGSIR)
│   │   ├── bert_xgb/             # Baseline Artifacts
│   │   └── ...
│   ├── logs/                     # Detailed Execution Logs
│   └── metrics/                  # Aggregated Results CSV
│       └── all_results.csv       # Final Metrics Summary Table
│
├── scripts/                      # Execution Scripts (Entry Points)
│   ├── process_dataset.py             # Dataset Cleaning & Splitting
│   ├── run_all_baselines.py           # Run Standard Training & Testing
│   ├── run_all_adversarial.py         # Run Robustness Evaluation
│   ├── run_mgsir_ablation.py # Run Ablation Studies (L1–L4)
│   ├── run_full_experiment_cycle.py   # One-click Reproduction
│   ├── run_fair_cpu_single_thread.sh  # One-click fair benchmark (CPU 1-thread)
│   └── generate_adversarial_test.py   # Adversarial Sample Generation
│
├── src/                           # Source Code
│   ├── config/                    # Configuration (Paths, Hyperparams)
│   ├── features/                  # Feature Engineering Core
│   │   ├── mgsir/                # MGSIR invariant features (MGSIF)
│   │   ├── deep/                  # Tokenizers for DL Models
│   │   └── ... (bert, tfidf, w2v extractors)
│   ├── models/                    # Model Architectures
│   │   ├── deep/                  # TextCNN, CharCNN, LSTM+Attn, etc.
│   │   └── traditional/           # XGBoost Trainer Wrapper
│   ├── pipelines/                 # Pipeline Orchestration (Train/Test/Adv)
│   │   ├── mgsir/                # MGSIR train/test/adv/ablation
│   │   └── ... (Model-specific pipelines)
│   └── utils/                     # Utilities (Logger, Data Loaders)
│
└── requirements.txt               # Dependencies
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and set up a Python 3.10+ virtual environment:

```bash
git clone https://github.com/your-username/MGSIR-SQLi.git
cd MGSIR-SQLi

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt          # Linux/Windows (default)
# pip install -r requirements.macos.txt    # macOS (Apple Silicon GPU)
```

---

### 2. Data Preparation (Optional)

```bash
# Automatically detects and split 'data/dataset1/raw/All_SQL_Dataset.csv'
python scripts/process_dataset.py

# Or specify a custom raw file to create a new dataset (e.g., dataset2)
python scripts/process_dataset.py --file /path/to/your/data.csv

```

### 2.1 Generate Adversarial Test Set

Adversarial samples are generated from `data/dataset1/processed/test.csv` by default and output to `data/dataset1/adversarial/` (filenames like `test_adv_randomcase.csv`).

```bash
python scripts/generate_adversarial_test.py
```

---

### 3. Run Experiments

#### A. Full Experiment Cycle (Recommended)

To reproduce all experiments (Training -\> Standard Test -\> Ablation -\> Adversarial) in one go:

```bash
python scripts/run_full_experiment_cycle.py
```

> Note: This runs in the default mode without enforcing a fixed CPU/thread/device setting. Latency/QPS may vary across machines and local runtime configurations. Use it to quickly validate the full pipeline.

**Paper evaluation setting (fair and comparable)**: keep training unconstrained (faster), but enforce CPU single-thread constraints for all testing/benchmarks (batch=1 end-to-end inference; excluding model loading).

```bash
python scripts/run_full_experiment_cycle.py --dataset dataset1 --threads 1 --device cpu --apply-to test

# Or use the one-click script (recommended for open-source reproduction)
bash scripts/run_fair_cpu_single_thread.sh dataset1
```

> Warning: This may take several hours depending on your hardware (especially deep model training).

#### Modular Execution

```bash
# 1) Baselines (recommended paper evaluation setting: CPU single-thread during testing)
python scripts/run_all_baselines.py --dataset dataset1 --threads 1 --device cpu --apply-to test

# 2) Ablation (recommended paper evaluation setting: CPU single-thread during testing)
python scripts/run_mgsir_ablation.py --dataset dataset1 --threads 1 --device cpu --apply-to test

# 3) Adversarial robustness (testing only; apply-to is not needed)
python scripts/run_all_adversarial.py --dataset dataset1 --threads 1 --device cpu
```

---

## ✅ Fair Benchmark Notes (Paper Evaluation Setting)

Different methods naturally include different preprocessing (e.g., canonicalization vs. tokenizer/padding). This is part of each method and is not considered unfair as long as we measure the same end-to-end online inference scope and enforce identical resource constraints.

We support enforcing identical test-time constraints for latency/QPS comparisons:

- `--threads`: force inference threads (recommended: `1`)
- `--device`: force inference device (recommended: `cpu`)
- `--apply-to`:
  - `test` (default): apply constraints to testing/benchmark only (training remains fast; engineering-realistic)
  - `all`: apply constraints to both training and testing (most rigorous; slower training)

Recommended (engineering-realistic + fair): `--threads 1 --device cpu --apply-to test`, or run `bash scripts/run_fair_cpu_single_thread.sh dataset1`.

---

## 🧠 MGSIF (Four Levels) XXXX

We design a **23-dimensional**, four-level feature hierarchy.

> **Core design philosophy**: Attackers can easily manipulate surface text (e.g., casing, whitespace), but structural invariants are harder to hide. MGSIF aims to capture these intrinsic structural properties.

### Level definitions

- **Level 1 (H1): Distributional Statistics**
  - **Definition**: Captures global distribution patterns (length and character density) to detect entropy shifts and encoding artifacts without relying on semantics.

- **Level 2 (H2): Syntactic Patterns**
  - **Definition**: Focuses on the usage of special characters (quotes, parentheses, punctuation) that define SQL syntax boundaries.

- **Level 3 (H3): Contextual Disruption**
  - **Definition**: Captures semantic anomalies through SQL keywords, functions, and logic operators that reveal malicious intent.

- **Level 4 (H4): Integrity Check**
  - **Definition**: Detects structural inconsistencies such as mismatched quotes or parentheses often caused by injection payloads.


### Feature table

| Level | Key | Description | Adversarial Rationale |
| --- | --- | --- | --- |
| **H1**<br>Distributional Statistics | `qlen` | Payload length | **Physical scale**: injection payloads often show length shifts when closing strings or chaining expressions. |
|  | `wcount` | Token count | **Complexity**: injected payloads introduce extra keywords/symbols and increase token count. |
|  | `hexnum` | Hex density | **Anti-encoding**: detects hex-encoded obfuscation (`0x..`). |
|  | `alpha` | Alphabetic density | **Entropy shift**: abnormal letter ratios indicate encoding/noise. |
|  | `digit` | Numeric density | **Numeric payloads**: blind or ASCII-based injections often contain many digits. |
|  | `coef_k` | Inter-token sparsity | **Whitespace obfuscation**: captures no-space or comment-as-space injections. |
| **H2**<br>Syntactic Patterns | `sq` | Single-quote count | **Boundary probe**: quote usage signals parameter escape attempts. |
|  | `dq` | Double-quote count | **Boundary probe**: alternative quoting for truncation/escape. |
|  | `lparen` | Left‑parenthesis count | **Nesting structure**: subqueries/functions/logic grouping. |
|  | `rparen` | Right‑parenthesis count | **Closure**: checks structural balance in expressions. |
|  | `puncts` | Punctuation count | **Syntax noise**: unnatural symbols indicate constructed payloads. |
|  | `comments` | Comment density | **Signal amplification**: comments are strong evasion signals (`--`, `#`, `/*`). |
|  | `arith` | Arithmetic operators | **Expression building**: common in blind/error-based injections. |
|  | `spaces` | Space count | **Format anomaly**: abnormal spacing reveals obfuscation. |
|  | `sym_logic` | Symbolic logic ops | **Non-text evasion**: detects `&&`, `||` to bypass keyword filters. |
| **H3**<br>Contextual Disruption | `logic` | Textual logic ops | **Conditional structure**: `AND`/`OR`/`XOR` drive SQL conditions. |
|  | `sqlkw` | SQL keyword freq | **Intent capture**: `UNION`, `SELECT`, `DROP` reflect DB action intent. |
|  | `sqlfunc` | SQL function freq | **Function abuse**: `SLEEP`, `BENCHMARK`, `UPDATEXML` indicate probes/abuse. |
|  | `tok_switch` | Token-type switch rate | **Obfuscation measure**: mixed types increase structure fragmentation. |
|  | `op_ratio` | Operator ratio | **Computation intensity**: blind injections often have dense operators. |
|  | `lit_ratio` | Literal ratio | **Data density**: injected payloads often contain excessive literals. |
| **H4**<br>Integrity Check | `qmismatch` | Quote mismatch | **Strong anomaly**: unbalanced quotes indicate truncation/escape. |
|  | `paren_mismatch` | Parenthesis mismatch | **Strong anomaly**: unbalanced parentheses reveal syntactic interference. |

---

## 🧰 Model comparison library

| Category          | Model Name     | Description             | Key Technology              |
| :---------------- | :------------- | :---------------------- | :-------------------------- |
| **Ours**          | `mgsir_xgb`    | MGSIF Features + XGBoost (MGSIR) | H1–H4 Features     |
| **Shallow NLP**   | `bow_xgb`      | Bag of Words            | CountVectorizer + XGBoost   |
|                   | `tfidf_xgb`    | TF-IDF                  | TfidfVectorizer + XGBoost   |
| **Embeddings**    | `w2v_xgb`      | Word2Vec                | Gensim W2V Avg Pooling      |
|                   | `fasttext_xgb` | FastText                | Subword Embeddings          |
| **Deep Features** | `bert_xgb`     | BERT [CLS] + XGBoost    | Pretrained Transformer      |
| **Deep E2E**      | `textcnn`      | TextCNN                 | 1D Conv                     |
|                   | `char_cnn`     | Char-level CNN          | Robust against typos        |
|                   | `cnn_bilstm`   | CNN-BiLSTM              | Spatial + Temporal modeling |
|                   | `lstm_attn`    | BiLSTM + Attention      | Sequence modeling           |

---

## ⚔️ Adversarial Testing

The framework evaluates model robustness against **8 types of obfuscation strategies**:

- **Set A (`randomcase`)**: Case manipulation (e.g., `SeLeCt`).
- **Set B (`space2comment`)**: Inline comment injection (e.g., `UNION/**/SELECT`).
- **Set C (`charencode`)**: URL encoding (e.g., `UN%49ON`).
- **Set D (`whitespace`)**: Whitespace variation (e.g., `SELECT\t\n*`).
- **Set E (`versioned`)**: Versioned comments (e.g., `/*!50000SELECT*/`).
- **Set F (`symbolic`)**: Symbolic replacement (e.g., `||` instead of `OR`).
- **Set G (`equaltolike`)**: Operator substitution (e.g., `LIKE` instead of `=`).
- **Set H (`mix`)**: Mixed attacks randomly combine 2–3 of the above strategies.

-----

## 📊 Metrics & Visualization

All results: `results/metrics/all_results.csv`

### Key Metrics

1. **Effectiveness**: Accuracy, Precision, Recall, F1-Score, AUC.
2. **WAF Robustness** (Critical for Security):
      - `Recall @ FPR=0.1%`: Detection rate at very low false alarm rates.
3. **Efficiency**:
      - `Latency Avg`: Average inference time (ms).
      - `Latency P99`: 99th percentile latency (tail latency).
      - `QPS`: Throughput (Queries Per Second).

### Visualization

High-quality plots are generated in `results/checkpoints/<model>/figures/`:

- `_cm.png` – Confusion Matrix
- `_roc.png` – ROC
- `_roc_log.png` – Log-scale ROC (critical for low-FPR performance)

---

## 📝 Disclaimer

This repository contains patented technologies under active patent application.
**Commercial use is strictly prohibited without authorization.**
Academic research and internal evaluation are welcome. Please cite this repository if used.

---

*Maintainer: XXX | 2025*
