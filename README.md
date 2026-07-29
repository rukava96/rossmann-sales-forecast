# Rossmann Store Sales Forecasting

Forecast daily sales for 1,115 Rossmann stores ([Kaggle competition](https://www.kaggle.com/competitions/rossmann-store-sales)).
Reliable forecasts help store managers build staff schedules and plan promotions.

## Business problem

- Predict daily `Sales` for each store on each day.
- Forecast horizon: up to 6 weeks, matching the competition test period.
- Evaluation metric: RMSPE (Root Mean Squared Percentage Error).

RMSPE is computed as:

\[
\text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2 }
\]

Days and stores with `Sales == 0` are excluded from scoring.

## Key lesson: avoiding target leakage

Earlier notebook iterations used `Sales` lag and rolling-mean/std features computed with
`rolling()` *without* an explicit `shift(1)`. That included the current day's `Sales` inside its
own rolling window, so the model implicitly learned `Sales ≈ f(Sales)`. Validation RMSPE dropped
to ~0.03, but the real Kaggle score was 0.23-0.26 because those exact features could not be
computed the same way on the test set (all future sales are unknown there).

**Fix applied in this codebase:**

- All lag/rolling `Sales` features were removed entirely.
- The only historical signal is `store_mean_sales` / `store_dow_mean_sales`, computed strictly
  from the training partition (see `src/features.py::add_historical_store_aggregates`) and joined
  onto validation/test without ever using future or current-day sales.
- Validation is a strict time-based split; the training aggregate is computed only on data that
  precedes the validation window.

See `src/features.py` module docstring for details.

## Data

- Source: [Kaggle - Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- See `data/README.md` for field descriptions and download instructions.

## Repository structure

```text
.
├── data/            # datasets and data description (not tracked in git)
├── models/          # persisted model artefacts (not tracked in git)
├── notebooks/        # exploratory notebooks (EDA, baseline, model selection)
├── src/              # reusable pipeline modules
│   ├── config.py      # paths and hyperparameters
│   ├── features.py     # leakage-free feature engineering
│   ├── models.py       # RMSPE metric, LightGBM ensemble training/inference
│   ├── train.py        # training entry point
│   └── predict.py      # batch inference / Kaggle submission entry point
├── app/               # FastAPI serving application
│   └── main.py
├── tests/             # pytest unit tests
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Approach

1. **EDA** (`notebooks/01_eda.ipynb`) - sales dynamics, seasonality, holiday/promo effects.
2. **Baseline** (`notebooks/02_baseline.ipynb`) - simple LightGBM model, no leakage-prone features.
3. **Feature engineering & model selection** (`notebooks/03_model_selection.ipynb`) - calendar,
   promo interaction, and leakage-free historical aggregate features; LightGBM vs CatBoost vs
   XGBoost comparison; Optuna tuning; bagging ensemble.
4. **Production pipeline** (`src/`) - the finalized, leakage-free version of the above, callable
   from the command line or served through FastAPI.

## Results

- Baseline LightGBM (no leakage): validation RMSPE ≈ 0.164.
- Tuned LightGBM ensemble (5-seed bagging): validation RMSPE ≈ 0.164, **private Kaggle score:
  0.14864**.
- Top features: `store_dow_mean_sales`, `store_mean_sales`, `CompetitionDistance`, `PromoDayOfWeek`,
  calendar features (`Quarter`, `DayOfWeek`).

## How to run

### Install dependencies

```bash
pip install -e ".[dev]"
```

### Download data

Place `train.csv`, `test.csv`, `store.csv`, `sample_submission.csv` into `data/`.
See `data/README.md` for details, or use the Kaggle CLI:

```bash
kaggle competitions download -c rossmann-store-sales -p data/
```

### Train the model

```bash
python -m src.train
```

This writes the fitted ensemble to `models/lgbm_ensemble.joblib` and the feature importance
table to `data/feature_importance.csv`.

### Generate a Kaggle submission

```bash
python -m src.predict
```

This writes `data/submission.csv` in the format expected by Kaggle.

### Run tests

```bash
pytest
```

### Lint

```bash
ruff check .
```

### Run the API locally

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

## API usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "store": 1,
        "day_of_week": 4,
        "date": "2015-09-17",
        "open": 1,
        "promo": 1,
        "state_holiday": "0",
        "school_holiday": 0,
        "store_type": "c",
        "assortment": "a",
        "competition_distance": 1270.0
      }'
```

Response:

```json
{
  "store": 1,
  "predicted_sales": 6123.45
}
```

## Links

- Kaggle competition: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)