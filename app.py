from fastapi import FastAPI
import joblib
import pandas as pd
import time

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest
)

from fastapi.responses import Response

app = FastAPI()

# =========================
# Load Model
# =========================

model = joblib.load(
    "training/random_forest.pkl"
)

# =========================
# Metrics
# =========================

REQUEST_COUNT = Counter(
    "request_count",
    "Total Request"
)

PREDICTION_COUNT = Counter(
    "prediction_count",
    "Total Prediction"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction Latency"
)

# =========================
# Routes
# =========================

@app.get("/")
def home():

    REQUEST_COUNT.inc()

    return {
        "message": "Model Telco Churn Running"
    }


@app.post("/predict")
def predict(data: dict):

    REQUEST_COUNT.inc()

    start_time = time.time()

    try:

        df = pd.DataFrame([data])

        # preprocessing sederhana
        df = pd.get_dummies(
            df,
            drop_first=True
        )

        # samakan kolom dengan model
        for col in model.feature_names_in_:

            if col not in df.columns:
                df[col] = 0

        df = df[model.feature_names_in_]

        prediction = model.predict(df)

        PREDICTION_COUNT.inc()

        latency = time.time() - start_time

        PREDICTION_LATENCY.observe(
            latency
        )

        return {
            "prediction": int(prediction[0])
        }

    except Exception as e:

        return {
            "error": str(e)
        }


@app.get("/metrics")
def metrics():

    return Response(
        generate_latest(),
        media_type="text/plain"
    )