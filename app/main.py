from fastapi import FastAPI

app = FastAPI(
    title="Rossmann Sales Forecast",
    description="API for predicting 6 weeks of daily sales for Rossmann stores (Kaggle).",
)

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/predict")
def predict():
    return {"message": "Predict endpoint is under construction"}
