import argparse
import sys
from pathlib import Path
import pandas as pd

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aircraft_severity.pipeline import AircraftSeverityPipeline
from aircraft_severity.config import FEATURE_COLUMNS, DEFAULT_DATA_PATH


def main():
    parser = argparse.ArgumentParser(description="Run Batch Predictions using Trained Aircraft Severity Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV file containing input features for prediction"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/predictions.csv",
        help="Path where prediction output CSV will be saved"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"[!] Error: Input file not found at {input_path}")
        sys.exit(1)

    print(f"[*] Loading input data from {input_path}...")
    df = pd.read_csv(input_path)

    # Ensure feature columns exist
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"[!] Error: Missing required feature columns in input dataset: {missing_cols}")
        sys.exit(1)

    X_input = df[FEATURE_COLUMNS]

    print("[*] Loading trained model pipeline...")
    pipeline = AircraftSeverityPipeline.load()

    print("[*] Generating predictions...")
    predictions = pipeline.predict(X_input)
    proba_df = pipeline.predict_proba(X_input)

    output_df = df.copy()
    output_df["Predicted_Severity"] = predictions
    
    for class_name in proba_df.columns:
        output_df[f"Proba_{class_name}"] = proba_df[class_name]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"[+] Batch predictions saved successfully to {output_path}")
    print("\nSample Output:")
    print(output_df[["Predicted_Severity"] + [c for c in output_df.columns if c.startswith("Proba_")]].head())


if __name__ == "__main__":
    main()
