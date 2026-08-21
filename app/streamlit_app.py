import sys
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

# Add src to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aircraft_severity.pipeline import AircraftSeverityPipeline
from aircraft_severity.config import FEATURE_COLUMNS, METRICS_PATH, DEFAULT_DATA_PATH
import json

st.set_page_config(
    page_title="Aircraft Severity Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_pipeline():
    try:
        pipeline = AircraftSeverityPipeline.load()
        return pipeline
    except Exception as e:
        return None


def main():
    st.title("✈️ Hybrid Ensemble Aircraft Accident Severity Predictor")
    st.markdown(
        """
        This intelligent aviation safety platform uses a **Hybrid Ensemble Machine Learning Framework** 
        (combining Multi-Layer Perceptron Neural Networks, Random Forest, and Gradient Boosting) 
        to classify the severity of aircraft accidents based on flight telemetry and safety metrics.
        """
    )

    pipeline = load_pipeline()

    if pipeline is None:
        st.error(
            "⚠️ **Model Pipeline Artifacts Not Found!**\n\n"
            "Please train the model first by running `py train.py` in your terminal."
        )
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🎯 Single Flight Predictor", "📁 Batch Prediction (CSV)", "📊 Model Performance"])

    with tab1:
        st.subheader("Interactive Telemetry Input")

        col1, col2, col3 = st.columns(3)

        with col1:
            safety_score = st.slider("Safety Score (0-100)", 0.0, 100.0, 42.0, 0.1)
            days_insp = st.number_input("Days Since Inspection", 0, 365, 12)
            total_complaints = st.number_input("Total Safety Complaints", 0, 100, 4)
            control_metric = st.slider("Control Metric (0-100)", 0.0, 100.0, 65.0, 0.1)

        with col2:
            turbulence = st.slider("Turbulence (g-forces)", 0.0, 2.0, 0.35, 0.01)
            cabin_temp = st.slider("Cabin Temperature (°F)", 50.0, 110.0, 78.0, 0.5)
            accident_type = st.selectbox("Accident Type Code", options=list(range(1, 8)), index=2)

        with col3:
            max_elevation = st.number_input("Max Elevation (ft)", 0, 80000, 31000, 500)
            violations = st.number_input("Recorded Violations", 0, 20, 1)
            weather_metric = st.slider("Adverse Weather Metric", 0.0, 5.0, 0.45, 0.05)

        input_data = pd.DataFrame([{
            "Safety_Score": safety_score,
            "Days_Since_Inspection": days_insp,
            "Total_Safety_Complaints": total_complaints,
            "Control_Metric": control_metric,
            "Turbulence_In_gforces": turbulence,
            "Cabin_Temperature": cabin_temp,
            "Accident_Type_Code": accident_type,
            "Max_Elevation": max_elevation,
            "Violations": violations,
            "Adverse_Weather_Metric": weather_metric,
        }])

        st.markdown("---")
        if st.button("🚀 Predict Accident Severity", type="primary"):
            pred_label = pipeline.predict(input_data)[0]
            proba_df = pipeline.predict_proba(input_data)
            proba_dict = proba_df.iloc[0].to_dict()

            res_col1, res_col2 = st.columns([1, 1])

            with res_col1:
                st.markdown("### Prediction Result")
                if "Minor" in pred_label:
                    st.success(f"🟢 **Severity Level**: `{pred_label}`")
                elif "Injuries" in pred_label:
                    st.warning(f"🟡 **Severity Level**: `{pred_label}`")
                elif "Fatalities" in pred_label:
                    st.error(f"🟠 **Severity Level**: `{pred_label}`")
                else:
                    st.error(f"🚨 **Severity Level**: `{pred_label}`")

                confidence = proba_dict[pred_label]
                st.metric("Model Confidence", f"{confidence * 100:.2f}%")

            with res_col2:
                st.markdown("### Class Probabilities")
                st.bar_chart(pd.Series(proba_dict))

    with tab2:
        st.subheader("Batch Flight Telemetry Processing")
        uploaded_file = st.file_uploader("Upload CSV file with flight metrics", type=["csv"])

        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Uploaded Sample Preview:", batch_df.head())

            missing_cols = [c for c in FEATURE_COLUMNS if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: {missing_cols}")
            else:
                if st.button("Run Batch Prediction"):
                    predictions = pipeline.predict(batch_df[FEATURE_COLUMNS])
                    proba_df = pipeline.predict_proba(batch_df[FEATURE_COLUMNS])

                    out_df = batch_df.copy()
                    out_df["Predicted_Severity"] = predictions
                    for col in proba_df.columns:
                        out_df[f"Proba_{col}"] = proba_df[col]

                    st.success("Batch Prediction Complete!")
                    st.dataframe(out_df)

                    csv = out_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Download Predictions CSV",
                        data=csv,
                        file_name="aircraft_severity_predictions.csv",
                        mime="text/csv",
                    )

    with tab3:
        st.subheader("Model Architecture & Evaluation Metrics")
        if METRICS_PATH.exists():
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)

            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Overall Accuracy", f"{metrics['accuracy']*100:.2f}%")
            m_col2.metric("Macro F1 Score", f"{metrics['f1_macro']:.4f}")
            m_col3.metric("Weighted F1 Score", f"{metrics['f1_weighted']:.4f}")

            st.markdown("#### Classification Report")
            st.json(metrics["classification_report"])
        else:
            st.info("Metrics report file not found. Train model to view detailed evaluation.")


if __name__ == "__main__":
    main()
