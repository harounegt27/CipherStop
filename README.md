# CipherStop
### Early Ransomware Detection Using Behavioral Sequence Analysis & Deep Learning

> TEK-UP University · Engineering School · Data Science & AI  
> Module: Deep Learning & Big Data · Academic Year 2026/2027

---

## Table of Contents
- [Overview](#overview)
- [Research Questions](#research-questions)
- [Architecture](#architecture)
- [Team](#team)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Pipeline](#pipeline)
- [Sprint Plan](#sprint-plan)
- [Expected Results](#expected-results)
- [Supervisors](#supervisors)

---

## Overview

Ransomware attacks have become one of the most destructive cyber threats of the decade — WannaCry infected 300,000 machines across 150 countries in 72 hours, and the Colonial Pipeline attack forced a 6-day shutdown of the largest fuel pipeline on the US East Coast.

Traditional antivirus tools rely on **static signature analysis** — they are blind to new, obfuscated, or zero-day ransomware variants and often trigger alerts only after encryption has already begun.

**CipherStop** takes a fundamentally different approach: instead of examining *what a program is*, it observes *what a program does* — monitoring low-level memory and storage access patterns in real time. By combining **Apache Spark** for large-scale behavioral trace processing and **LSTM-based deep learning** for sequential pattern recognition, CipherStop aims to detect ransomware **before significant damage occurs** — even against previously unseen families.

---

## Research Questions

| # | Question |
|---|----------|
| **RQ1** | Can ransomware be reliably detected by observing only a fraction of its behavioral sequence (10%, 25%, 50%, or 75%)? |
| **RQ2** | What is the tradeoff between how early the system detects the attack and how accurate that detection is? |
| **RQ3** | Can the model generalize to a ransomware family it has never seen during training? |

---

## Architecture

```
RanSMAP Dataset (Kaggle)
        │
        ▼
┌───────────────────┐
│  Apache Spark     │  ← Large-scale behavioral trace processing
│  Preprocessing    │    Sequence cuts: 10% / 25% / 50% / 75%
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Great Expectations│  ← Automated data validation
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  LSTM + Attention │  ← Sequential pattern recognition
│  Model (PyTorch)  │    Tracked with MLflow
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  FastAPI          │  ← REST prediction endpoint
│  Endpoint         │    Containerized with Docker
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  React Dashboard  │  ← Live Monitor / Alert Detail / History
│  + WebSocket      │    Email notifications via SMTP
└───────────────────┘
```

---

## Team

| Member | Role | Responsibilities |
|--------|------|-----------------|
| **Mohamed Haroun Mezned** | Infrastructure Engineer | GitHub, Docker, CI/CD, FastAPI |
| **Roua Tbarki** | Data Engineer | RanSMAP, Spark Pipeline, DVC |
| **Ayett Mansouri** | ML Engineer | LSTM Model, MLflow, Evaluation |

---

## Project Structure

```
CipherStop/
├── data/
│   ├── processed/
│   │   ├── 10pct/          # 10% sequence cut
│   │   ├── 25pct/          # 25% sequence cut
│   │   ├── 50pct/          # 50% sequence cut
│   │   ├── 75pct/          # 75% sequence cut
│   │   └── zero_day/       # Held-out zero-day test set
│   └── validated/          # Great Expectations validated data
├── docker/
│   └── Dockerfile
├── notebooks/
│   └── kaggle_setup.md     # Guide for running on Kaggle Notebooks
├── scripts/
│   └── download_data.py
├── src/
│   ├── pipeline/
│   │   ├── preprocess.py   # Spark preprocessing pipeline
│   │   └── validate.py     # Great Expectations validation
│   ├── model/
│   │   └── train.py        # LSTM training loop with MLflow
│   └── api/
│       └── main.py         # FastAPI prediction endpoint
├── .dvc/                   # DVC configuration
├── .github/
│   └── workflows/          # GitHub Actions CI/CD
├── docker-compose.yml
├── dvc.yaml                # DVC pipeline definition
├── params.yaml             # Pipeline parameters
├── requirements.txt
└── README.md
```

---

## Getting Started

### Prerequisites
- Python 3.11.9
- Docker & Docker Compose
- Git + DVC
- Kaggle account (for dataset access)

### 1. Clone the repository
```bash
git clone https://github.com/harounegt27/CipherStop.git
cd CipherStop
```

### 2. Create a virtual environment
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Run with Docker
```bash
docker-compose up --build
```

### 4. Run the pipeline (on Kaggle Notebooks)
> ⚠️ The RanSMAP dataset (100GB+) is never downloaded locally.  
> All data processing runs on Kaggle Notebooks where the dataset is already available.

See [`notebooks/kaggle_setup.md`](notebooks/kaggle_setup.md) for step-by-step instructions.

```bash
# On Kaggle Notebook — after cloning the repo
dvc repro
```

---

## Pipeline

Defined in `dvc.yaml` and parameterized via `params.yaml`:

```
preprocess → validate → train
```

| Stage | Tool | Output |
|-------|------|--------|
| `preprocess` | Apache Spark | Parquet files per sequence cut |
| `validate` | Great Expectations | Validated dataset |
| `train` | PyTorch + MLflow | Trained LSTM model |

### Key parameters (`params.yaml`)
```yaml
preprocess:
  cuts: [10, 25, 50, 75]
  zero_day: true
  input_path: /kaggle/input/ransmap-2024-ransomware-behavioral-features/

train:
  epochs: 50
  batch_size: 64
  learning_rate: 0.001
  model: lstm
```

---

## Sprint Plan

| Sprint | Weeks | Focus | Deliverable |
|--------|-------|-------|-------------|
| **Sprint 1** | 1–2 | Setup & Data Pipeline | Clean processed sequences |
| **Sprint 2** | 3–4 | Model Development | Trained LSTM in MLflow |
| **Sprint 3** | 5–6 | Evaluation & MLOps | Full CI/CD pipeline |
| **Sprint 4** | 7–8 | Simulation & Delivery | Dashboard, report, live demo |

---

## Expected Results

| Metric | Target |
|--------|--------|
| F1-score at ≤50% sequence | > 0.85 |
| False Positive Rate | < 5% |
| Zero-day F1-score | > 0.80 |

---

## Dataset

**RanSMAP (2024)** — Ransomware Storage and Memory Access Patterns  
- Author: Manabu Hirano  
- Source: [Kaggle](https://www.kaggle.com/datasets/hiranomanabu/ransmap-2024-ransomware-behavioral-features)  
- Contains behavioral traces of ransomware and benign software samples

---

## Supervisors

- **Mr. Mohamed Nadjib Ben Daoud**
- **Mrs. Sawsen Jalel**

---

> TEK-UP University · Engineering School · Data Science & AI · 2026/2027
