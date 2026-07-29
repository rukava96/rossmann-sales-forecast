# Data

## Source

- Kaggle competition: [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales)

## Files

- `train.csv` — historical sales data including the `Sales` column (target).
- `test.csv` — historical data excluding the `Sales` column (to be predicted).
- `store.csv` — supplemental information about the stores.
- `sample_submission.csv` — a sample submission file in the correct format.

## Data fields

Most fields are self-explanatory. Key fields:

- `Store` — a unique Id for each store.
- `Sales` — the turnover for any given day (target variable).
- `Customers` — the number of customers on a given day.
- `Open` — indicator whether the store was open: 0 = closed, 1 = open.
- `StateHoliday` — indicates a state holiday:  
  - `a` = public holiday  
  - `b` = Easter holiday  
  - `c` = Christmas  
  - `0` = None
- `SchoolHoliday` — indicates if public schools were closed.
- `StoreType` — 4 different store models: `a`, `b`, `c`, `d`.
- `Assortment` — assortment level:  
  - `a` = basic  
  - `b` = extra  
  - `c` = extended
- `CompetitionDistance` — distance in meters to the nearest competitor store.
- `CompetitionOpenSince[Month/Year]` — approximate opening time of the nearest competitor.
- `Promo` — indicates whether a store is running a promo on that day.
- `Promo2` — continuing promotion: 0 = not participating, 1 = participating.
- `Promo2Since[Year/Week]` — year and calendar week when the store started Promo2.
- `PromoInterval` — consecutive intervals when Promo2 is started (e.g. "Feb,May,Aug,Nov").

## How to download

1. Create a Kaggle account.
2. Go to the competition page.
3. Download `train.csv`, `test.csv`, `store.csv`, `sample_submission.csv` and place them into this folder.

Optionally, use Kaggle API:

```bash
kaggle competitions download -c rossmann-store-sales
```

Then unzip archives here.