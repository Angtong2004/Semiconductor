"""
models.py
Defines, trains, and exports multiple ML classifiers for semiconductor defect classification:
- Random Forest
- XGBoost (Extreme Gradient Boosting)
- Multi-Layer Perceptron (Neural Network)
- Support Vector Classifier (RBF Kernel)
- Logistic Regression (Linear Baseline)
"""

import numpy as np
from typing import Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import joblib


def get_model_zoo(random_state: int = 42) -> Dict[str, Any]:
    """
    Returns an ensemble of machine learning classifiers tuned for semiconductor defect metrology.
    """
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=120,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1
        ),
        "MLP_NeuralNet": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=32,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=random_state
        ),
        "SupportVectorMachine": CalibratedClassifierCV(
            SVC(
                C=5.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                random_state=random_state
            ),
            ensemble=False
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        )
    }


class MetrologyClassifierTrainer:
    """
    Orchestrates multi-model training, hyperparameter fit, and evaluation benchmarking.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = get_model_zoo(random_state=random_state)
        self.trained_models: Dict[str, Any] = {}

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ) -> Dict[str, Any]:
        """
        Fits each model in the model zoo.
        """
        for name, model in self.models.items():
            print(f"--> Training {name}...")
            if name == "XGBoost" and X_val is not None and y_val is not None:
                # XGBoost can leverage validation set for early stopping
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)

            self.trained_models[name] = model

        print("--> All models successfully trained.")
        return self.trained_models

    def save_model(self, model_name: str, file_path: str):
        if model_name not in self.trained_models:
            raise KeyError(f"Model {model_name} has not been trained yet.")
        joblib.dump(self.trained_models[model_name], file_path)

    @staticmethod
    def load_model(file_path: str):
        return joblib.load(file_path)


if __name__ == "__main__":
    from physics_simulation import SemiconductorMetrologySimulator
    from preprocessing import prepare_dataset_splits

    sim = SemiconductorMetrologySimulator()
    raw_df = sim.generate_full_dataset(total_samples=1000)
    splits = prepare_dataset_splits(raw_df)

    trainer = MetrologyClassifierTrainer()
    trainer.train_all(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    print(f"Trained {len(trainer.trained_models)} models successfully.")
