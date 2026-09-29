# SAST IQ — Incremental Learning SAST Platform

<div align="center">

![SAST IQ Banner](https://img.shields.io/badge/SAST-Intelligence%20Platform-6c63ff?style=for-the-badge&logo=shield&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

**A self-improving Static Application Security Testing (SAST) platform that learns from developer feedback and improves detection accuracy over time.**

</div>

---

## Overview

SAST IQ combines traditional rule-based vulnerability detection with an incremental ML pipeline. Developers review findings and label them — the system uses this feedback to retrain its detection model weekly, reducing false positives and improving accuracy over time.

```
Scan Repo → Findings → Developer Labels → Retrain Model → Better Detection
     ↑______________________________________________|
```

---

## Architecture

```
backend/
├── api/
│   └── routes.py          # REST API endpoints
├── ml/
│   ├── retraining.py      # TF-IDF + Logistic Regression retraining pipeline
│   └── model_registry.py  # Versioned model storage (.joblib + JSON registry)
├── scanner/
│   ├── git_utils.py       # Differential scanning via GitPython (git diff)
│   └── sast_core.py       # 3-stage scan pipeline
├── database.py            # SQLite engine & session management
├── models.py              # SQLAlchemy ORM: Feedback + SmartMemory tables
└── main.py                # FastAPI entrypoint
frontend/
├── index.html             # Single-page analytics dashboard
├── styles.css             # Dark glassmorphism design (vanilla CSS)
└── app.js                 # Dashboard JS + API integration
seed_training_data.py      # Bootstrap script to seed initial training data
requirements.txt
```

---

## Features

| Feature | Description |
|---------|-------------|
| 🔍 **Repository Scanning** | Full or differential (`git diff`) scan of local repos |
| 🧠 **3-Stage Detection Pipeline** | SmartMemory override → ML Model → Regex Baseline rules |
| 📝 **Developer Feedback API** | Label findings as Valid / False Positive / Needs Review |
| 🔁 **Incremental Retraining** | Automatic model improvement from accumulated feedback |
| 📦 **Model Versioning** | Every retrained model is versioned and tracked |
| 🛡️ **Smart Memory** | Regex/string patterns for safe libraries that override ML predictions |
| 📊 **Analytics Dashboard** | Live metrics: FP reduction %, acceptance rate, risk trends |

---

## Detection Capabilities (Baseline Rules)

| Vulnerability | CWE | Severity |
|--------------|-----|----------|
| SQL Injection | CWE-89 | HIGH |
| Command Injection | CWE-78 | HIGH |
| Code Injection (eval) | CWE-95 | HIGH |
| Insecure Deserialization | CWE-502 | HIGH |
| Hardcoded Credentials | CWE-798 | HIGH |
| Hardcoded Secrets | CWE-798 | HIGH |
| XSS (innerHTML / document.write) | CWE-79 | MEDIUM |
| Weak Cryptography (MD5) | CWE-327 | MEDIUM |
| OS Command Execution | CWE-78 | MEDIUM |

---

## Quick Start

### 1. Clone & Install
```bash
git clone <your-repo-url>
cd SAST
pip install -r requirements.txt
```

### 2. Start the Server
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

### 3. Open the Dashboard
```
http://localhost:8000/dashboard
```

### 4. API Documentation
```
http://localhost:8000/docs
```

---

## Usage

### Scan a Repository
```bash
# Via dashboard UI: click "⚡ Scan Repo" and enter the path
# Or via API:
curl -X POST http://localhost:8000/api/scan \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/path/to/your/repo", "mode": "full"}'

# Differential scan (only changed files):
-d '{"repo_path": "/path/to/your/repo", "mode": "diff"}'
```

### Submit Developer Feedback
```bash
curl -X POST http://localhost:8000/api/feedback \
  -H "Content-Type: application/json" \
  -d '{"code_hash": "<hash>", "developer_label": "false_positive"}'
# Labels: "valid_vulnerability" | "false_positive" | "needs_review"
```

### Add a Smart Memory Pattern (Safe Override)
```bash
curl -X POST http://localhost:8000/api/smart-memory \
  -H "Content-Type: application/json" \
  -d '{"pattern": "internal_auth\\.verify", "pattern_type": "safe_library", "description": "Internal auth is always safe"}'
```

### Retrain the Model
```bash
# Via API:
curl -X POST http://localhost:8000/api/model/retrain

# Via CLI (recommended for scheduled jobs):
python -m backend.ml.retraining
```

### Bootstrap Training Data (First Run)
```bash
python seed_training_data.py
python -m backend.ml.retraining
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/scan` | Scan a local git repository |
| `GET` | `/api/feedback` | List unlabelled (pending) findings |
| `POST` | `/api/feedback` | Submit a developer label for a finding |
| `GET` | `/api/dashboard/metrics` | Analytics: FP rate, acceptance rate, trends |
| `POST` | `/api/smart-memory` | Add a safe pattern override |
| `GET` | `/api/smart-memory` | List all smart memory patterns |
| `POST` | `/api/model/retrain` | Trigger model retraining |
| `GET` | `/api/model/versions` | List all model versions |

---

## ML Pipeline

The learning pipeline uses:
- **Feature Extraction:** TF-IDF with bigram support (`TfidfVectorizer`)
- **Classifier:** Logistic Regression with class balancing
- **Threshold:** Dynamically calibrated per model version based on FP precision
- **Storage:** Versioned `.joblib` files tracked in `models/registry.json`

Minimum **10 labelled findings** are required to trigger a retraining run.

---

## Dashboard

The dashboard (`http://localhost:8000/dashboard`) provides:

- **KPI Cards:** Total Scans, False Positive Rate, Developer Acceptance Rate, Active Model version
- **Risk Trend Chart:** Weekly vulnerability detection count
- **Vulnerability Recurrence:** Bar chart of most frequent finding types
- **Findings Queue:** Pending findings awaiting developer review with inline labelling buttons
- **Smart Memory Manager:** Add/view safe pattern overrides
- **Model History:** All versioned models with threshold and sample count info

---

## Project Structure — Key Files

| File | Purpose |
|------|---------|
| `backend/scanner/sast_core.py` | 3-stage detection: SmartMemory → ML → Regex |
| `backend/scanner/git_utils.py` | `git diff` extraction for differential scanning |
| `backend/ml/retraining.py` | Weekly retraining script |
| `backend/ml/model_registry.py` | Version-tracked model load/save |
| `backend/api/routes.py` | All REST endpoints |
| `seed_training_data.py` | Initial training data bootstrap |

---

## Tech Stack

- **Backend:** Python 3.11, FastAPI, SQLAlchemy, SQLite
- **ML:** scikit-learn (TF-IDF + Logistic Regression), joblib
- **Git Integration:** GitPython
- **Frontend:** Vanilla HTML/CSS/JS (no frameworks), Inter font

---

## Future Improvements

- [ ] Scheduled weekly retraining via cron/Celery
- [ ] Support for more languages (Java, Go, C++)
- [ ] Integration with GitHub/GitLab CI/CD webhooks
- [ ] PostgreSQL support for production deployments
- [ ] Export findings as SARIF format
- [ ] Slack/email notifications for High severity findings

---

<div align="center">
Built as an Incremental Learning SAST proof-of-concept.<br/>
<i>The more you use it, the smarter it gets.</i>
</div>
