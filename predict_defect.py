"""
predict_defect.py
Command-line utility and Python API to perform defect classification on new or single-die metrology measurements.
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from src.physics_simulation import DEFECT_CLASSES, DEFECT_DESCRIPTIONS
from src.preprocessing import MetrologyPreprocessor, FEATURE_COLUMNS


def predict_metrology_sample(
    sample_dict: dict,
    model_path: str = "outputs/models/champion_model_RandomForest.joblib",
    preprocessor_path: str = "outputs/models/metrology_preprocessor.joblib"
) -> dict:
    """
    Given a dictionary of raw metrology measurements, runs preprocessing and outputs
    predicted defect class, probability distribution, and engineering root-cause description.
    """
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or Preprocessor artifact missing. Please run `python run_pipeline.py` first.")

    preprocessor: MetrologyPreprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    df_sample = pd.DataFrame([sample_dict])

    # Ensure all required features are present
    for col in FEATURE_COLUMNS:
        if col not in df_sample.columns:
            df_sample[col] = 0.0

    X_scaled = preprocessor.transform(df_sample)
    pred_class_id = int(model.predict(X_scaled)[0])
    class_name = DEFECT_CLASSES[pred_class_id]

    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_scaled)[0]
        for idx, p in enumerate(probs):
            probabilities[DEFECT_CLASSES[idx]] = float(np.round(p, 4))

    return {
        "predicted_class_id": pred_class_id,
        "predicted_class_name": class_name,
        "description": DEFECT_DESCRIPTIONS[class_name],
        "class_probabilities": probabilities
    }


if __name__ == "__main__":
    # Example demo: Die with elevated roughness and localized spike (Particulate contamination)
    sample_particle_defect = {
        "wafer_x_mm": 24.5,
        "wafer_y_mm": -45.2,
        "normalized_radius": 0.34,
        "film_thickness_nm": 105.2,
        "refractive_index_633nm": 1.488,
        "extinction_coeff_k": 0.021,
        "optical_phase_delta_deg": 158.0,
        "reflectance_vis_pct": 27.5,
        "reflectance_uv_pct": 39.2,
        "roughness_ra_nm": 2.95,
        "roughness_rq_nm": 4.12,
        "roughness_rz_nm": 45.0,  # Prominent spike
        "surface_skewness": 2.45,  # Strong positive skew
        "surface_kurtosis": 8.1,   # Leptokurtic tail
        "sheet_resistance_ohm_sq": 26.1,
        "breakdown_field_mv_cm": 5.4,
        "log10_leakage_current_na": 2.3,
        "v_flatband_shift_v": 0.22,
        "cd_bias_nm": 2.1,
        "defect_area_um2": 12.5,
        "defect_aspect_ratio": 1.15
    }

    result = predict_metrology_sample(sample_particle_defect)
    print("\n--- Metrology Inference Result ---")
    print(f"Prediction: {result['predicted_class_name']} (ID: {result['predicted_class_id']})")
    print(f"Fab Root Cause: {result['description']}")
    print("\nClass Probabilities:")
    for c_name, prob in result["class_probabilities"].items():
        print(f"  {c_name:<28}: {prob * 100:.2f}%")
