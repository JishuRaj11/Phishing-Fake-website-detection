# Phishing Fake Website Detection

An AI-powered phishing URL detection system that uses Machine Learning to analyze URL-based features and identify potentially malicious websites. Built with Python, Scikit-learn, XGBoost, FastAPI, HTML, CSS, and JavaScript.

No network requests are made at inference time: the model only analyzes the text of the submitted URL.

## Features

- URL-only feature extraction (44 features: lengths, character counts and ratios, domain structure, entropy, encoding, suspicious keywords, shorteners, and more)
- Leakage-safe preprocessing fitted on the training split only
- Trained XGBoost classifier with probability scores returned as confidence
- FastAPI REST endpoint `POST /predict` with validated input and CORS for the local frontend
- Frontend UI that displays the prediction, confidence and every extracted feature

## Tech stack

Python, Pandas, NumPy, Scikit-learn, XGBoost, Joblib, FastAPI, Uvicorn, HTML, CSS, JavaScript.

## Project structure

```
phishing_pipeline/   feature extraction, cleaning, analysis and preprocessing
backend/             FastAPI application, detector, schemas and config
frontend/            HTML, CSS and JavaScript UI
outputs/models/      trained model artifacts used by the API
outputs/             preprocessor and evaluation artifacts
run_pipeline.py      dataset inspection, cleaning and preprocessing
train_models.py      model training and comparison
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

The committed artifacts (`outputs/models/xgboost.joblib` and `outputs/preprocessor.joblib`) are enough to run the API. To regenerate them, download the PhiUSIIL Phishing URL Dataset and place it at `dataset/PhiUSIIL_Phishing_URL_Dataset.csv`, then run:

```bash
python run_pipeline.py
python train_models.py
```

The raw dataset and the large processed arrays are intentionally excluded from this repository because of their size.

## Run locally

Backend (port 8000):

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Frontend (port 5500):

```bash
cd frontend
python -m http.server 5500
```

Open `http://127.0.0.1:5500` and submit a URL such as `https://www.google.com`.

## API

`POST /predict`

```json
{ "url": "https://www.google.com" }
```

Response contains `prediction`, `is_phishing`, `confidence`, the submitted `url`, and all 44 extracted `features`.
