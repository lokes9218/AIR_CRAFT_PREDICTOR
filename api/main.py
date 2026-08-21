import sys
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_severity.pipeline import AircraftSeverityPipeline
from aircraft_severity.config import MODEL_PIPELINE_PATH, FEATURE_COLUMNS
from api.schemas import (
    AircraftFlightMetrics,
    SinglePredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthCheckResponse,
)

app = FastAPI(
    title="Aircraft Accident Severity Prediction API",
    description="Production REST API for classifying aircraft accident severity levels using Hybrid Ensemble Machine Learning models.",
    version="1.0.0",
)

# Enable CORS for frontend applications
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_model: AircraftSeverityPipeline = None


@app.on_event("startup")
def load_model():
    global pipeline_model
    try:
        pipeline_model = AircraftSeverityPipeline.load()
        print("[+] FastAPI server successfully loaded model pipeline.")
    except Exception as e:
        print(f"[!] Warning: Could not load model pipeline on startup ({e}). API endpoints will attempt to load on request.")


@app.get("/", response_model=HealthCheckResponse, summary="API Healthcheck & Status")
def healthcheck():
    loaded = pipeline_model is not None and pipeline_model.pipeline is not None
    return HealthCheckResponse(
        status="healthy" if loaded else "degraded - model missing",
        model_loaded=loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=SinglePredictionResponse, summary="Predict Severity for a Single Flight")
def predict_single(metrics: AircraftFlightMetrics):
    global pipeline_model
    if pipeline_model is None or pipeline_model.pipeline is None:
        try:
            pipeline_model = AircraftSeverityPipeline.load()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not trained or not found. Please train model first using train.py. Error: {e}"
            )

    # Convert request dict to DataFrame
    input_data = pd.DataFrame([metrics.model_dump()])
    
    # Predict
    predicted_label = pipeline_model.predict(input_data)[0]
    probabilities_df = pipeline_model.predict_proba(input_data)
    proba_dict = {col: float(probabilities_df[col].iloc[0]) for col in probabilities_df.columns}
    confidence = float(proba_dict[predicted_label])

    return SinglePredictionResponse(
        predicted_severity=predicted_label,
        confidence=round(confidence, 4),
        class_probabilities={k: round(v, 4) for k, v in proba_dict.items()}
    )


@app.post("/batch_predict", response_model=BatchPredictionResponse, summary="Predict Severity for Multiple Flights")
def predict_batch(request: BatchPredictionRequest):
    global pipeline_model
    if pipeline_model is None or pipeline_model.pipeline is None:
        try:
            pipeline_model = AircraftSeverityPipeline.load()
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Model pipeline not trained or available. Error: {e}"
            )

    input_dicts = [item.model_dump() for item in request.flights]
    input_df = pd.DataFrame(input_dicts)

    predictions = pipeline_model.predict(input_df)
    proba_df = pipeline_model.predict_proba(input_df)

    results = []
    for idx, pred_label in enumerate(predictions):
        row_proba = {col: float(proba_df[col].iloc[idx]) for col in proba_df.columns}
        conf = row_proba[pred_label]
        results.append(
            SinglePredictionResponse(
                predicted_severity=pred_label,
                confidence=round(conf, 4),
                class_probabilities={k: round(v, 4) for k, v in row_proba.items()}
            )
        )

    return BatchPredictionResponse(predictions=results)
