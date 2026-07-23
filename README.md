# Rossmann Store Sales Forecasting

Forecast 6 weeks of daily sales for 1,115 Rossmann stores (Kaggle). Reliable sales forecasts help store managers create effective staff schedules and focus on customers and teams.

## Business problem

- Predict daily sales (`Sales` column) for each Rossmann store.
- Forecast horizon: up to 6 weeks of daily sales.
- Evaluation metric: RMSPE (Root Mean Square Percentage Error).

RMSPE is calculated as:

\[
\text{RMSPE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \frac{y_i - \hat{y}_i}{y_i} \right)^2 }
\]

where \(y_i\) denotes the sales of a single store on a single day and \(\hat{y}_i\) denotes the corresponding prediction. Days and stores with 0 sales are ignored in scoring.

## Data

- Source: [Kaggle – Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- Main files:
  - `train.csv` — historical sales data including the `Sales` column.
  - `test.csv` — historical data excluding the `Sales` column.
  - `store.csv` — supplemental information about the stores.
  - `sample_submission.csv` — a sample submission file in the correct format.

Key fields:

- `Store` — unique Id for each store.
- `Sales` — turnover for a given day (target).
- `Customers` — number of customers on a given day.
- `Open` — 0 = closed, 1 = open.
- `StateHoliday` — state holiday indicator (`a`, `b`, `c`, `0`).
- `SchoolHoliday` — public school closure indicator.
- `StoreType` — store model (`a`, `b`, `c`, `d`).
- `Assortment` — assortment level (`a`, `b`, `c`).
- `CompetitionDistance`, `CompetitionOpenSince[Month/Year]`.
- `Promo`, `Promo2`, `Promo2Since[Year/Week]`, `PromoInterval`.

See `data/README.md` for details.

## Approach

1. **EDA**
   - Analyze sales dynamics, seasonality, and the effect of holidays and promotions.
   - Compare stores by `StoreType`, `Assortment`, and competition proximity.

2. **Baseline**
   - Simple heuristics (e.g., mean sales per store for open days).
   - Basic model without advanced features.

3. **Feature engineering**
   - Calendar features: day of week, month, year, state/school holidays.
   - Store features from `store.csv`: `StoreType`, `Assortment`, competition info.
   - Promo features: `Promo`, `Promo2`, `Promo2Since`, `PromoInterval`.
   - Lag features and rolling statistics (mean, std over last 7/14/30 days).

4. **Models**
   - Gradient boosting: LightGBM / CatBoost / XGBoost.
   - Time-series cross-validation (train on past, validate on future).

5. **Evaluation**
   - Compare baseline vs final model using RMSPE.
   - Analyze feature importance.

## Results

- Baseline RMSPE: …
- Best model RMSPE: …
- Top important features: …

(To be filled after experiments.)

## Submission format

The submission file should contain a header and have the following format:

```text
Id,Sales
1,0
2,0
3,0
...
```

Any day and store with 0 sales is ignored in scoring.

## How to run

### Install dependencies

```bash
pip install ".[dev]"
```

or:

```bash
pip install -e ".[dev]"
```

### Download data

Place `train.csv`, `test.csv`, `store.csv`, `sample_submission.csv` into the `data/` folder.  
See `data/README.md` for details.

Optionally, use Kaggle API:

```bash
kaggle competitions download -c rossmann-store-sales
```

### Train model

```bash
python src/train.py
```

### Run API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker

```bash
docker-compose up --build
```

API will be available at `http://localhost:8000`.

## Demo

Example request:

```bash
curl http://localhost:8000/predict
```

Response:

```json
{
  "message": "Predict endpoint is under construction"
}
```

(Later replace with a real prediction example.)

## Notebooks

- `notebooks/01_eda.ipynb` — exploratory data analysis.
- `notebooks/02_baseline_model.ipynb` — baseline models.
- `notebooks/03_feature_engineering_and_models.ipynb` — feature engineering and final models.

## Repository structure

```text
.
├── data/           # datasets and data description
├── notebooks/      # Jupyter notebooks
├── src/            # reusable Python modules
├── tests/          # tests
├── app/            # FastAPI application
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Links

- Kaggle competition: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)
- My Kaggle profile: [link, if available]