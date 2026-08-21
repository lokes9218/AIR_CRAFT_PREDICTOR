import argparse
import sys
from pathlib import Path

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from aircraft_severity.pipeline import train_and_save_pipeline


def main():
    parser = argparse.ArgumentParser(description="Train Aircraft Severity Prediction Hybrid Ensemble Pipeline")
    parser.add_argument(
        "--model-type",
        choices=["stacking", "voting"],
        default="stacking",
        help="Type of hybrid ensemble model to train ('stacking' or 'voting')"
    )
    args = parser.parse_args()

    print(f"=== Starting Training Pipeline (Model: {args.model_type}) ===")
    metrics = train_and_save_pipeline(model_type=args.model_type)
    print(f"=== Training Finished! Accuracy: {metrics['accuracy']:.4f} ===")


if __name__ == "__main__":
    main()
