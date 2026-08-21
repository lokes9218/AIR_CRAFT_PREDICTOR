import json
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from aircraft_severity.config import (
    MODEL_PIPELINE_PATH,
    LABEL_ENCODER_PATH,
    METRICS_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from aircraft_severity.data import load_dataset, prepare_data
from aircraft_severity.features import AircraftFeatureScaler
from aircraft_severity.models import create_hybrid_pipeline_model


class AircraftSeverityPipeline:
    """
    End-to-end Machine Learning pipeline wrapper for training, evaluating, saving, loading, and predicting.
    """
    def __init__(self, model_type: str = "stacking"):
        self.model_type = model_type
        self.pipeline = None
        self.label_encoder = LabelEncoder()
        self.classes_ = None

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Fits label encoder and scikit-learn training pipeline.
        """
        # Fit label encoder on target strings
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_

        # Assemble leak-free Scikit-Learn Pipeline
        classifier = create_hybrid_pipeline_model(self.model_type)
        self.pipeline = Pipeline([
            ('scaler', AircraftFeatureScaler()),
            ('classifier', classifier)
        ])

        print(f"[*] Training {self.model_type.upper()} Hybrid Ensemble Pipeline...")
        self.pipeline.fit(X_train, y_train_encoded)
        print("[+] Training completed successfully.")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluates model on test dataset and returns metrics dictionary.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet.")

        y_test_encoded = self.label_encoder.transform(y_test)
        y_pred_encoded = self.pipeline.predict(X_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred_encoded)

        acc = accuracy_score(y_test, y_pred_labels)
        f1_macro = f1_score(y_test, y_pred_labels, average='macro')
        f1_weighted = f1_score(y_test, y_pred_labels, average='weighted')
        report_dict = classification_report(y_test, y_pred_labels, output_dict=True)
        cm = confusion_matrix(y_test, y_pred_labels, labels=self.classes_).tolist()

        metrics = {
            "model_type": self.model_type,
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "classes": self.classes_.tolist(),
            "confusion_matrix": cm,
            "classification_report": report_dict,
        }

        print(f"\n================ MODEL EVALUATION ({self.model_type.upper()}) ================")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Macro: {f1_macro:.4f}")
        print(f"F1 Weighted: {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_labels))

        return metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts string severity labels for input features.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet. Call fit() or load().")
        
        y_pred_encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts class probability distributions for input features.
        """
        if self.pipeline is None:
            raise RuntimeError("Pipeline is not fitted yet. Call fit() or load().")

        proba = self.pipeline.predict_proba(X)
        return pd.DataFrame(proba, columns=self.classes_)

    def save(self, pipeline_path: Path = MODEL_PIPELINE_PATH, encoder_path: Path = LABEL_ENCODER_PATH):
        """
        Serializes pipeline and label encoder to disk.
        """
        pipeline_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, pipeline_path)
        joblib.dump(self.label_encoder, encoder_path)
        print(f"[+] Saved model pipeline to {pipeline_path}")
        print(f"[+] Saved label encoder to {encoder_path}")

    @classmethod
    def load(cls, pipeline_path: Path = MODEL_PIPELINE_PATH, encoder_path: Path = LABEL_ENCODER_PATH):
        """
        Loads serialized pipeline and label encoder from disk.
        """
        if not pipeline_path.exists() or not encoder_path.exists():
            raise FileNotFoundError(f"Model artifacts not found at {pipeline_path} or {encoder_path}")

        instance = cls()
        instance.pipeline = joblib.load(pipeline_path)
        instance.label_encoder = joblib.load(encoder_path)
        instance.classes_ = instance.label_encoder.classes_
        print(f"[+] Successfully loaded pipeline from {pipeline_path}")
        return instance


def train_and_save_pipeline(model_type: str = "stacking") -> dict:
    """
    High-level entry point to load data, train pipeline, evaluate, and save artifacts.
    """
    df = load_dataset()
    X_train, X_test, y_train, y_test = prepare_data(df)

    pipeline_model = AircraftSeverityPipeline(model_type=model_type)
    pipeline_model.fit(X_train, y_train)
    metrics = pipeline_model.evaluate(X_test, y_test)
    pipeline_model.save()

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[+] Saved evaluation metrics to {METRICS_PATH}")

    return metrics
