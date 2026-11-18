# Sprint 1 — Setup Summary
**Project:** CipherStop — Early Ransomware Detection  
**Period:** Sprint 1 (Weeks 1–2)  
**Author:** Mohamed Haroun Mezned — Infrastructure Engineer

---

## Task 1 — GitHub Repository & Branch Strategy

### What was done
- Created the GitHub repository: [harounegt27/CipherStop](https://github.com/harounegt27/CipherStop)
- Defined and applied the following branch strategy:

```
main          ← stable, production-ready code only
develop       ← integration branch (all PRs merge here first)
feature/*     ← one branch per feature, created on demand
experiment/*  ← model experiments (Sprint 2+)
hotfix/*      ← urgent fixes on main
```

- Created and pushed the `develop` branch:
```bash
git checkout -b develop
git push -u origin develop
```

- Set remote origin:
```bash
git remote add origin https://github.com/harounegt27/CipherStop.git
```

### Branch protection rules (GitHub Settings → Branches)
- `main` → require PR + review before merge, no direct push
- `develop` → require PR before merge

### Workflow (per feature)
```bash
git checkout develop
git pull origin develop
git checkout -b feature/<name>
# ... do the work ...
git add .
git commit -m "feat: <description>"
git push -u origin feature/<name>
# Open PR on GitHub → feature/<name> → develop → merge → delete branch
```

### Problems & Solutions

| Problem | Solution |
|---------|----------|
| Confusion about when to create `feature/*` branches | Feature branches are **never created upfront** — only when starting a specific task |
| Confusion about merging `develop` → `main` | `main` only receives a merge when **all Sprint tasks are complete and stable** |

---

## Task 2 — Docker Environment

### What was done
- Created `feature/docker-setup` branch
- Added the following files:

#### `docker/Dockerfile`
```dockerfile
FROM python:3.11.9-slim

RUN apt-get update && apt-get install -y \
    openjdk-17-jdk-headless \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
```

#### `docker-compose.yml` (Sprint 1 scope — dev only)
```yaml
version: "3.9"

services:
  dev:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: cipherstop_dev
    volumes:
      - .:/app
      - ./data:/app/data
    environment:
      - PYTHONPATH=/app
    ports:
      - "8888:8888"
    command: bash
    stdin_open: true
    tty: true
```

#### `requirements.txt`
```txt
pyspark==3.5.3
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
torch==2.4.1
mlflow==2.16.2
great-expectations==0.18.19
dvc==3.55.2
fastapi==0.115.4
uvicorn==0.32.0
pyarrow==17.0.0
python-dotenv==1.0.1
```

#### `.dockerignore`
```
__pycache__/
*.pyc
*.pyo
.git/
.dvc/
data/
*.egg-info/
.env
mlruns/
```

- Opened PR `feature/docker-setup` → `develop` → merged → branch deleted ✅

### Problems & Solutions

| Problem | Solution |
|---------|----------|
| Forgot to create a virtual environment before installing dependencies | Created `.venv` with `python -m venv .venv` and activated it before `pip install -r requirements.txt` |

---

## Task 3 — DVC Setup (Data Versioning)

### What was done
- Created `feature/dvc-setup` branch
- Initialized DVC and configured the pipeline for **Kaggle Notebooks** as the compute environment

> ⚠️ The RanSMAP dataset exceeds 100GB — it is **never downloaded locally**.  
> All data processing runs on Kaggle Notebooks where the dataset is already available at:  
> `/kaggle/input/ransmap-2024-ransomware-behavioral-features/`

#### Project structure created
```
data/
├── processed/
│   ├── 10pct/
│   ├── 25pct/
│   ├── 50pct/
│   ├── 75pct/
│   └── zero_day/
└── validated/
src/
├── pipeline/
│   ├── preprocess.py
│   └── validate.py
├── model/
└── api/
notebooks/
└── kaggle_setup.md
scripts/
└── download_data.py
```

#### `dvc.yaml`
```yaml
stages:
  preprocess:
    cmd: python src/pipeline/preprocess.py
    deps:
      - src/pipeline/preprocess.py
    params:
      - params.yaml:
          - preprocess.cuts
          - preprocess.zero_day
    outs:
      - data/processed/10pct
      - data/processed/25pct
      - data/processed/50pct
      - data/processed/75pct
      - data/processed/zero_day

  validate:
    cmd: python src/pipeline/validate.py
    deps:
      - src/pipeline/validate.py
      - data/processed
    outs:
      - data/validated

  train:
    cmd: python src/model/train.py
    deps:
      - src/model/train.py
      - data/processed
    params:
      - params.yaml:
          - train.epochs
          - train.batch_size
          - train.learning_rate
    metrics:
      - metrics.json:
          cache: false
```

#### `params.yaml`
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

#### `.gitignore`
```
# Data — never stored locally
data/processed/*
data/validated/*
!data/**/.gitkeep

# DVC cache
.dvc/cache/

# MLflow
mlruns/

# Environment
.env
.venv/
venv/
env/

# Python
__pycache__/
*.pyc
*.pyo
```

- Opened PR `feature/dvc-setup` → `develop` → merged → branch deleted ✅

### Problems & Solutions

| Problem | Solution |
|---------|----------|
| Data initially estimated at 30GB — too large for laptop | After investigation, dataset exceeds 100GB — **never downloaded locally** |
| First approach: Google Drive remote (15GB personal account — not enough) | Rejected — insufficient storage |
| Second approach: Kaggle API download locally | Rejected — data too large for local machine |
| **Final solution** | Data stays on **Kaggle servers** — pipeline runs entirely on **Kaggle Notebooks** — DVC tracks pipeline logic only, not raw data |
| `dvc init` error — `cannot import name '_DIR_MARK' from pathspec` | Fixed with `pip install --upgrade pathspec` then `pip install dvc==3.55.2` |
| `dvc init` error — `.dvc is ignored by your SCM tool` | Fixed by adding `!.dvc` at the top of `.gitignore` |

---

## Task 4 — README

### What was done
- Created `feature/readme` branch
- Wrote a complete `README.md` covering:
  - Project overview and context
  - Research questions (RQ1, RQ2, RQ3)
  - System architecture diagram (ASCII)
  - Team roles and responsibilities
  - Full project folder structure
  - Getting started guide (clone, venv, Docker, Kaggle Notebooks)
  - Pipeline stages and parameters
  - Sprint plan table
  - Expected results and metrics
  - Dataset reference and supervisors

- Opened PR `feature/readme` → `develop` → merged → branch deleted ✅

### Problems & Solutions

| Problem | Solution |
|---------|----------|
| Existing README was a placeholder | Fully rewritten from scratch based on the project specification (`Cypher_Stop_CC.pdf`) |

---

## Sprint 1 — Final Branch State

```
main      ← untouched (no merge yet — Sprint 1 still in progress)
develop   ← contains all 4 merged features ✅
```

### Next step
Once all Sprint 1 tasks are validated → merge `develop` → `main` to close Sprint 1.

---

## Bonus — CipherStop Logo

A logo was designed for the system featuring:
- A geometric blue shield with a padlock
- Binary code pattern in the background
- "Cipher" in blue + "Stop" in neutral typography
- Subtle detection pulse arcs

**Midjourney generation prompt:**
```
A modern minimalist tech logo for a cybersecurity system called "CipherStop".
A bold geometric shield in deep blue (#1d4ed8) with a white padlock icon centered
inside. Subtle binary code digits (0s and 1s) fading in the background behind the
shield. Clean sans-serif typography: the word "Cipher" in blue and "Stop" in dark
charcoal next to the shield. Tagline "EARLY RANSOMWARE DETECTION" in small spaced
caps below. Two faint circular pulse rings around the shield suggesting real-time
detection. Flat design, no gradients, no shadows, transparent background,
professional cybersecurity branding style.
--ar 3:1 --style raw
```