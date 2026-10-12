"""
preprocessing.py
Data preprocessing, feature engineering, and stratified splitting pipeline for semiconductor metrology data.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib

FEATURE_COLUMNS = [
    # Spatial
    "wafer_x_mm",
    "wafer_y_mm",
    "normalized_radius",

    # Ellipsometry & Optical
    "film_thickness_nm",
    "refractive_index_633nm",
    "extinction_coeff_k",
    "optical_phase_delta_deg",
    "reflectance_vis_pct",
    "reflectance_uv_pct",

    # AFM & Surface
    "roughness_ra_nm",
    "roughness_rq_nm",
    "roughness_rz_nm",
    "surface_skewness",
    "surface_kurtosis",

    # Electrical
    "sheet_resistance_ohm_sq",
    "breakdown_field_mv_cm",
    "log10_leakage_current_na",
    "v_flatband_shift_v",

    # CD-SEM & Defect
    "cd_bias_nm",
    "defect_area_um2",
    "defect_aspect_ratio"
]

ENGINEERED_COLUMNS = [
    "uv_vis_reflectance_ratio",
    "roughness_peak_ratio",
    "breakdown_margin_metric",
    "thickness_cd_coupling"
]


class MetrologyPreprocessor:
    """
    Transforms raw simulated metrology features into scaled and engineered representations.
    Ensures zero data leakage by fitting transformers strictly on training splits.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: List[str] = []

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes domain-specific physical ratios from metrology measurements.
        """
        df_out = df.copy()

        # 1. Optical Spectral Ratio (UV to VIS reflectance balance)
        df_out["uv_vis_reflectance_ratio"] = (
            df_out["reflectance_uv_pct"] / (df_out["reflectance_vis_pct"] + 1e-4)
        )

        # 2. AFM Peak-to-Roughness Aspect (detects spike vs plateau topography)
        df_out["roughness_peak_ratio"] = (
            df_out["roughness_rz_nm"] / (df_out["roughness_ra_nm"] + 1e-4)
        )

        # 3. Electrical Reliability Metric (dielectric health relative to leakage)
        leakage_linear = np.power(10.0, df_out["log10_leakage_current_na"])
        df_out["breakdown_margin_metric"] = (
            df_out["breakdown_field_mv_cm"] / (leakage_linear + 1e-2)
        )

        # 4. Critical Dimension to Thickness Coupling (geometry aspect indicator)
        df_out["thickness_cd_coupling"] = (
            df_out["cd_bias_nm"] / (df_out["film_thickness_nm"] + 1e-4)
        )

        return df_out

    def fit_transform(
        self,
        X_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Engineers features and fits scaler on training set.
        """
        engineered_df = self.engineer_features(X_df[FEATURE_COLUMNS])
        self.feature_names = list(engineered_df.columns)
        scaled_features = self.scaler.fit_transform(engineered_df)
        self.is_fitted = True
        return scaled_features

    def transform(
        self,
        X_df: pd.DataFrame
    ) -> np.ndarray:
        """
        Applies fitted feature engineering and scaling to test/validation set.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor has not been fitted yet. Call fit_transform first.")
        engineered_df = self.engineer_features(X_df[FEATURE_COLUMNS])
        return self.scaler.transform(engineered_df)

    def save(self, file_path: str):
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: str) -> "MetrologyPreprocessor":
        return joblib.load(file_path)


def prepare_dataset_splits(
    df: pd.DataFrame,
    test_size: float = 0.20,
    val_size: float = 0.15,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Performs stratified train-validation-test split.
    Returns dictionary with raw subsets and preprocessed arrays.
    """
    # First split off test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["defect_class_id"],
        random_state=random_state
    )

    # Then split train into train and val
    val_ratio_adjusted = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df["defect_class_id"],
        random_state=random_state
    )

    preprocessor = MetrologyPreprocessor(scaler_type="robust")
    X_train_scaled = preprocessor.fit_transform(train_df)
    X_val_scaled = preprocessor.transform(val_df)
    X_test_scaled = preprocessor.transform(test_df)

    y_train = train_df["defect_class_id"].to_numpy()
    y_val = val_df["defect_class_id"].to_numpy()
    y_test = test_df["defect_class_id"].to_numpy()

    return {
        "preprocessor": preprocessor,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": preprocessor.feature_names
    }


if __name__ == "__main__":
    from physics_simulation import SemiconductorMetrologySimulator
    sim = SemiconductorMetrologySimulator()
    raw_df = sim.generate_full_dataset(total_samples=1000)
    splits = prepare_dataset_splits(raw_df)
    print("Preprocessed feature dimension:", splits["X_train"].shape)
    print("Classes in y_train:", np.bincount(splits["y_train"]))
    print("Features count:", len(splits["feature_names"]))
