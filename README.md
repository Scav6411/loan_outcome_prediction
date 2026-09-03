# Loan Credit Risk Analysis and Prediction

Predicts the probability that a loan applicant will repay, from their in-app behaviour, GPS
mobility, masked profile attributes and application timing. Ships as a trained XGBoost model
behind a FastAPI service that builds features on demand from a PostgreSQL database.

Author: Soham Pandit (IIT Indore). Full methodology and results write-up: [final_report.pdf](final_report.pdf).

---

## What it does

Given a `user_id` and an `application_at` timestamp, the service:

1. Pulls that user's rows from the `features`, `events` and `gps` tables, keeping only records
   **strictly before** `application_at` (leakage prevention is enforced in the feature layer, not
   just in training).
2. Builds 28 features across four families (temporal, GPS, behavioural, masked profile).
3. Scores them with a pre-trained XGBoost classifier and returns a calibrated repayment
   probability in `[0, 1]`.

No thresholding is applied — the output is a probability, intended as a top-of-funnel risk filter
rather than a standalone accept/reject decision.

## Repository layout

| Path | Purpose |
| --- | --- |
| [app.py](app.py) | FastAPI app: `/api/v1/predict` and `/health` |
| [inference.py](inference.py) | End-to-end prediction pipeline — DB fetch → features → model |
| [config.py](config.py) | Env-based settings (DB credentials, model path) |
| [schema.py](schema.py) | Pydantic request/response models and typed error classes |
| [db2csv.py](db2csv.py) | One-off dump of the source tables to `dataset/*.csv` for offline work |
| [feature_generators/](feature_generators/) | Feature engineering classes (see below) |
| [weights/best_weights.json](weights/best_weights.json) | Trained XGBoost model (150 trees, `binary:logistic`) |
| [metrics.txt](metrics.txt) | Validation metrics for all five candidate models |
| [predictions.csv](predictions.csv) | Scored output for the holdout applicant set |
| `*.ipynb` | EDA, feature importance (SHAP), training, stacking, batch inference |

### Feature generators

All inherit [BaseFeatures](feature_generators/base_features.py), which owns the shared
pre-application filtering and merge helpers.

- **[TemporalFeatures](feature_generators/temporal_features.py)** — hour, day of week, day of
  month, year, weekend flag, late-night flag (23:00–05:00, a proxy for financial urgency).
- **[GPSFeatures](feature_generators/gps_features.py)** — max land speed, pre-application GPS
  point count, unique locations, dominant-location ratio, location entropy, radius of gyration,
  plus log transforms of the skewed counts.
- **[BehavioralFeatures](feature_generators/behavioral_features.py)** — 14-day window over app
  events: percentile ranks for session count, active days, session duration and recency; a
  `diligence_stability` interaction (engagement depth × location stability); and screen-to-screen
  **bigram default-rate** features fitted out-of-fold with K-fold smoothing to avoid target leakage
  (`mode='train'` fits, `mode='test'` applies the fitted map).

## Model

XGBoost was selected over Decision Tree, Random Forest, CatBoost and LightGBM — best ROC-AUC
*and* the best-calibrated probabilities (lowest log loss and Brier score).

| Model | ROC-AUC | PR-AUC | Log Loss | Brier |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.5296 | 0.3555 | 0.6705 | 0.2389 |
| Random Forest | 0.5668 | 0.3940 | 0.6177 | 0.2127 |
| LightGBM | 0.5829 | 0.4283 | 0.6407 | 0.2248 |
| CatBoost | 0.5999 | 0.4361 | 0.6456 | 0.2265 |
| **XGBoost** | **0.6008** | 0.4231 | **0.6065** | **0.2083** |

Hyperparameters: `n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8, random_state=42`. Data was sorted chronologically before a 90/10
train/validation split. Ranking metrics were chosen deliberately over accuracy/F1 because the
deliverable is a probability, not a class. A stacked CatBoost + LightGBM + XGBoost ensemble
([stacked_ensemble.ipynb](stacked_ensemble.ipynb)) gave only marginal gains and is not deployed.

## Setup

Requires Python 3.13 (Docker image pins 3.11) and a reachable PostgreSQL instance with the
`features`, `events` and `gps` tables.

```bash
pip install -r requirements.txt      # or: uv sync
cp .env.example .env                 # then fill in the values
```

`.env` variables — all read in [config.py](config.py), none have defaults except `DB_PORT` (5432)
and `MODEL_PATH` (`weights/best_weights.json`):

```
DB_HOST=      DB_PORT=      DB_USER=      DB_PASSWORD=      DB_NAME=      MODEL_PATH=
```

## Running

```bash
uvicorn app:app --reload --port 8000
```

Or with Docker (the image copies only the serving code, model weights and feature generators —
notebooks and datasets are excluded):

```bash
docker build -t loan-outcome .
docker run --env-file .env -p 8000:8000 loan-outcome
```

### API

`GET /health` → `{"status": "healthy"}`

`GET /api/v1/predict` — takes a JSON body:

```json
{ "user_id": 1, "application_at": "2022-06-29T08:08:38.092903" }
```

```json
{ "user_id": 1, "application_at": "2022-06-29T08:08:38.092903", "prediction": 0.2696989 }
```

Failures are mapped to distinct status codes by cause: `404` no data for the user, `503` database
unreachable, `422` feature pipeline failure, `500` model inference or unexpected error.

> The predict route is registered as `GET` but declares a Pydantic body model. Most HTTP clients
> will not send a body on GET — switch the decorator to `@router.post` if you hit that.

### Offline / batch use

[db2csv.py](db2csv.py) dumps `loan_outcomes_train`, `loan_outcomes_predict`, `features`, `events`
and `gps` into `dataset/`; the notebooks read from there.
[inference.ipynb](inference.ipynb) trains on the full set and writes
[predictions.csv](predictions.csv). Note that the served API defaults the two bigram features to
`0.0` unless `bigram_fitted_objects` from training is passed in — the batch path does pass them, so
API and batch scores are not strictly identical.

## Notes

- `inference.py` monkey-patches `pandas.reset_index` for a `name=` compatibility shim under
  pandas 2.x; it must be imported before the feature generators run.
- The XGBoost model is loaded once and cached on the `ModelService` class.
- Geographic coverage in the training data is concentrated in Kenya (Nairobi and the eastern coast).
