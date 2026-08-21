from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class AircraftFlightMetrics(BaseModel):
    Safety_Score: float = Field(..., ge=0.0, le=100.0, description="Safety evaluation score (0-100)")
    Days_Since_Inspection: int = Field(..., ge=0, description="Days elapsed since last safety inspection")
    Total_Safety_Complaints: int = Field(..., ge=0, description="Total logged safety complaints")
    Control_Metric: float = Field(..., ge=0.0, le=100.0, description="Flight control metric score")
    Turbulence_In_gforces: float = Field(..., ge=0.0, le=2.0, description="Turbulence intensity in g-forces")
    Cabin_Temperature: float = Field(..., ge=40.0, le=120.0, description="Cabin temperature in Fahrenheit")
    Accident_Type_Code: int = Field(..., ge=1, le=10, description="Categorical code representing type of accident")
    Max_Elevation: float = Field(..., ge=0.0, le=80000.0, description="Maximum altitude reached in feet")
    Violations: int = Field(..., ge=0, description="Number of recorded regulatory violations")
    Adverse_Weather_Metric: float = Field(..., ge=0.0, description="Weather severity index")

    model_config = {
        "json_schema_extra": {
            "example": {
                "Safety_Score": 45.2,
                "Days_Since_Inspection": 14,
                "Total_Safety_Complaints": 4,
                "Control_Metric": 68.5,
                "Turbulence_In_gforces": 0.32,
                "Cabin_Temperature": 78.4,
                "Accident_Type_Code": 3,
                "Max_Elevation": 31200.0,
                "Violations": 1,
                "Adverse_Weather_Metric": 0.45
            }
        }
    }


class SinglePredictionResponse(BaseModel):
    predicted_severity: str
    confidence: float
    class_probabilities: Dict[str, float]


class BatchPredictionRequest(BaseModel):
    flights: List[AircraftFlightMetrics]


class BatchPredictionResponse(BaseModel):
    predictions: List[SinglePredictionResponse]


class HealthCheckResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
