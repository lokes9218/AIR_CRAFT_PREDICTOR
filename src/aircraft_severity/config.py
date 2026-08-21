import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Config
DEFAULT_DATA_PATH = DATA_DIR / "AirData.csv"
ROOT_DATA_PATH = BASE_DIR / "AirData.csv"

TARGET_COLUMN = "Severity"
ID_COLUMN = "Accident_ID"

FEATURE_COLUMNS = [
    "Safety_Score",
    "Days_Since_Inspection",
    "Total_Safety_Complaints",
    "Control_Metric",
    "Turbulence_In_gforces",
    "Cabin_Temperature",
    "Accident_Type_Code",
    "Max_Elevation",
    "Violations",
    "Adverse_Weather_Metric",
]

SEVERITY_CLASSES = [
    "Minor_Damage_And_Injuries",
    "Significant_Damage_And_Fatalities",
    "Significant_Damage_And_Injuries",
    "Highly_Fatal_And_Damaging",
]

MODEL_PIPELINE_PATH = MODELS_DIR / "model_pipeline.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
