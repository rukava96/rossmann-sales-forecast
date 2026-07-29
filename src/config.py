from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
STORE_PATH = DATA_DIR / "store.csv"
SAMPLE_SUBMISSION_PATH = DATA_DIR / "sample_submission.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_FILE = MODEL_DIR / "rossmann_model.joblib"

# Forecast horizon: up to 6 weeks of daily sales
FORECAST_HORIZON_WEEKS = 6

# Evaluation metric: RMSPE (Root Mean Square Percentage Error)
METRIC = "RMSPE"

KAGGLE_API_TOKEN = "KGAT_61ff0020a9716dd09374d46cec05268d"
