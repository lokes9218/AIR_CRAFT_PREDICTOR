import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from aircraft_severity.config import (
    DEFAULT_DATA_PATH,
    ROOT_DATA_PATH,
    TARGET_COLUMN,
    ID_COLUMN,
    FEATURE_COLUMNS,
    SEVERITY_CLASSES,
)


def generate_synthetic_dataset(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic Aircraft Accident Severity dataset matching the original dataset schema.
    """
    np.random.seed(random_state)

    accident_ids = np.arange(1, n_samples + 1)
    
    # Feature distributions based on original EDA statistics
    safety_score = np.clip(np.random.normal(loc=41.8, scale=16.1, size=n_samples), 0, 100)
    days_since_insp = np.clip(np.random.poisson(lam=12.9, size=n_samples), 1, 30)
    total_complaints = np.clip(np.random.negative_binomial(n=2, p=0.23, size=n_samples), 0, 50)
    control_metric = np.clip(np.random.normal(loc=65.0, scale=12.0, size=n_samples), 10, 100)
    turbulence = np.clip(np.random.normal(loc=0.38, scale=0.12, size=n_samples), 0.1, 1.0)
    cabin_temp = np.clip(np.random.normal(loc=79.5, scale=3.5, size=n_samples), 65.0, 95.0)
    accident_type = np.random.randint(1, 8, size=n_samples)
    max_elevation = np.clip(np.random.normal(loc=30000, scale=8000, size=n_samples), 1000, 60000)
    violations = np.random.poisson(lam=2.0, size=n_samples)
    weather_metric = np.clip(np.random.exponential(scale=0.5, size=n_samples), 0.0, 5.0)

    # Compute a risk score to assign realistic Severity categories
    risk_score = (
        (100 - safety_score) * 0.35 +
        days_since_insp * 0.8 +
        total_complaints * 1.5 -
        control_metric * 0.2 +
        turbulence * 25.0 +
        violations * 4.0 +
        weather_metric * 15.0 +
        np.random.normal(0, 5, size=n_samples)
    )

    percentiles = np.percentile(risk_score, [30, 60, 85])
    
    severity_labels = []
    for r in risk_score:
        if r < percentiles[0]:
            severity_labels.append("Minor_Damage_And_Injuries")
        elif r < percentiles[1]:
            severity_labels.append("Significant_Damage_And_Injuries")
        elif r < percentiles[2]:
            severity_labels.append("Significant_Damage_And_Fatalities")
        else:
            severity_labels.append("Highly_Fatal_And_Damaging")

    df = pd.DataFrame({
        ID_COLUMN: accident_ids,
        TARGET_COLUMN: severity_labels,
        "Safety_Score": np.round(safety_score, 4),
        "Days_Since_Inspection": days_since_insp,
        "Total_Safety_Complaints": total_complaints,
        "Control_Metric": np.round(control_metric, 4),
        "Turbulence_In_gforces": np.round(turbulence, 6),
        "Cabin_Temperature": np.round(cabin_temp, 4),
        "Accident_Type_Code": accident_type,
        "Max_Elevation": np.round(max_elevation, 4),
        "Violations": violations,
        "Adverse_Weather_Metric": np.round(weather_metric, 6),
    })

    return df


def load_dataset(file_path: Path = None) -> pd.DataFrame:
    """
    Loads dataset from given path, root path, or generates synthetic data if none exists.
    """
    path_to_try = file_path or DEFAULT_DATA_PATH
    
    if not path_to_try.exists() and ROOT_DATA_PATH.exists():
        path_to_try = ROOT_DATA_PATH

    if not path_to_try.exists():
        print(f"[*] Dataset not found at {path_to_try}. Generating synthetic dataset...")
        df = generate_synthetic_dataset()
        save_dataset(df, DEFAULT_DATA_PATH)
        save_dataset(df, ROOT_DATA_PATH)
        return df

    df = pd.read_csv(path_to_try)
    return df


def save_dataset(df: pd.DataFrame, file_path: Path):
    """
    Saves dataframe to CSV file ensuring parent directories exist.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(file_path, index=False)
    print(f"[+] Dataset saved to {file_path}")


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares X and y data matrices and splits into train and test sets.
    """
    df = df.copy()
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
