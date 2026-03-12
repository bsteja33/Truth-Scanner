# Truth Scanner: Production-Grade Fake News Detection

A compound AI system that classifies news articles as real or fake using a soft-voting ensemble of five classifiers, served via a FastAPI backend and a Streamlit web interface with real-time probability visualization.

---

## Core Features

This system is engineered for high accuracy, robustness, and production safety, utilizing a multi-layered architecture:

* **Soft-Voting Ensemble:** Combines MultinomialNB, LogisticRegression, SVC (linear), SGDClassifier, and RandomForestClassifier over TF-IDF and CountVectorized features.
* **REST API Backend:** Built with FastAPI, featuring strict Pydantic payload validation and high-concurrency handling.
* **Interactive Frontend:** Streamlit UI with Plotly gauge charts for real-time confidence visualization.
* **Adversarial Robustness:** Tested against byline ablation, keyword perturbation, and temporal data drift.
* **Production Safety:** Includes input validation, payload truncation, and comprehensive HTTP error propagation.

---

## Architecture

```text
data/ (ISOT True.csv + Fake.csv)
    |
    v
src/train.py
    SpacyPreprocessor  -->  FeatureUnion (TF-IDF bigrams + CountVectorizer)
                        -->  VotingClassifier (soft)
                                  |- MultinomialNB
                                  |- LogisticRegression
                                  |- SVC (linear kernel)
                                  |- SGDClassifier (log-loss)
                                  |- RandomForestClassifier
                        -->  models/ensemble_pipeline.pkl
    |
    v
api/api.py  (FastAPI + Uvicorn)
    POST /predict  <--  Pydantic-validated JSON payload
    |
    v
frontend/app.py  (Streamlit)
    Gauge chart, tabbed report viewer, downloadable documentation
```

---

## Model Performance

Evaluated on a stratified 20 % holdout split of the ISOT dataset (8,980 samples).

| Metric | FAKE | REAL | Overall |
|---|---|---|---|
| Precision | 1.00 | 1.00 | — |
| Recall | 1.00 | 1.00 | — |
| F1-Score | 1.00 | 1.00 | — |
| Accuracy | — | — | **99.64 %** |

Confusion matrix: 15 false positives, 17 false negatives across 8,980 samples.

### Robustness Audit

| Test | Result | Interpretation |
|---|---|---|
| Byline Ablation — strip "CITY (Reuters) -" | -1 % accuracy drop | Model reads article content, not source names |
| Adversarial perturbation | 0 / 100 predictions flipped | Immune to keyword manipulation |
| Cross-domain (2024-2026 articles) | 50 % | Expected temporal drift from 2016-2017 training data |
| Error calibration | 28 / 29 errors had confidence <= 80 % | Uncertainty correlates with incorrectness |

---

## Safety and Robustness Features

| Patch | Implementation |
|---|---|
| Empty / whitespace input rejection | Pydantic `@field_validator` returns HTTP 422 |
| Payload truncation | Inputs silently capped at 50,000 characters before inference |
| API unavailability | `requests.exceptions.ConnectionError` caught; user shown polite error |
| HTTP error surfacing | `HTTPError` handler exposes status code and API error body |
| File handle safety | All report reads use `with open(...) as f:` context managers |
| Client-side guard | Streamlit blocks submission if input exceeds 50,000 characters |

---

## Dataset

**ISOT Fake News Dataset** — University of Victoria

| File | Label | Articles |
|---|---|---|
| `True.csv` | 1 (Real) | 21,417 |
| `Fake.csv` | 0 (Fake) | 23,481 |

The dataset files are excluded from this repository (see `.gitignore`). Place `True.csv` and `Fake.csv` inside the `data/` directory before training.

---

## Local Deployment

### Prerequisites

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Train the model

```bash
python src/train.py
```

Serialised pipeline is written to `models/ensemble_pipeline.pkl`.

### Start the API server

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload
```

### Start the frontend

```bash
streamlit run frontend/app.py
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| API interactive docs | http://localhost:8000/docs |

To start both services together:

```bash
python main.py
```

---

## API Reference

**`POST /predict`**

Request body:

```json
{
  "text": "The Senate passed a bipartisan infrastructure bill worth 1.2 trillion dollars."
}
```

Response:

```json
{
  "prediction": "TRUE",
  "confidence_score": 0.9741
}
```

`prediction` is `"TRUE"` (real) or `"FALSE"` (fake).  
`confidence_score` is the ensemble's probability for the predicted class.  
Empty or whitespace-only payloads return HTTP 422.

---

## Project Structure

```
.
├── api/
│   └── api.py              # FastAPI application and /predict endpoint
├── data/                   # ISOT dataset CSVs (not committed)
├── frontend/
│   └── app.py              # Streamlit web interface
├── models/                 # Trained pipeline binary (not committed)
├── src/
│   ├── train.py            # Pipeline definition, training, serialisation
│   ├── evaluate.py         # Holdout-set evaluation and report generation
│   └── robustness_test.py  # Byline ablation, adversarial, and drift tests
├── .gitignore
├── LICENSE
├── README.md
├── main.py
├── requirements.txt
├── run.bat
└── run.sh
```

---

## Deployment via Docker

This is the recommended method for running the production environment.

### Pre-requisite

The trained model must be present before building the images. If you have not already trained it:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python src/train.py
```

### Build and start both services

```bash
docker-compose up --build
```

This command builds the backend and frontend images and starts both containers on a shared internal network. The frontend resolves the backend via Docker's internal DNS (`http://backend:8000`).

| Service | Exposed port |
|---|---|
| Backend (FastAPI) | http://localhost:8000 |
| Frontend (Streamlit) | http://localhost:8501 |

### Stop all services

```bash
docker-compose down
```

Both services are configured with `restart: unless-stopped` and will recover automatically from container crashes.

---

## License

MIT License — Copyright (c) 2026 bsteja33 (B. Sai Teja). See [LICENSE](LICENSE) for details.
