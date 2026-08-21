import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aircraft_severity.data import load_dataset, prepare_data
from aircraft_severity.pipeline import AircraftSeverityPipeline


def main():
    print("=== Evaluating Trained Model Pipeline on Test Set ===")
    df = load_dataset()
    _, X_test, _, y_test = prepare_data(df)

    pipeline = AircraftSeverityPipeline.load()
    metrics = pipeline.evaluate(X_test, y_test)
    print("\n[+] Evaluation report generated successfully.")


if __name__ == "__main__":
    main()
