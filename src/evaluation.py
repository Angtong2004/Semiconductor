"""
evaluation.py
Comprehensive evaluation metrics, model benchmarking, and diagnostics for semiconductor defect classification.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    log_loss
)
try:
    from src.physics_simulation import DEFECT_CLASSES
except (ImportError, ModuleNotFoundError):
    from physics_simulation import DEFECT_CLASSES


def evaluate_single_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str
) -> Dict[str, Any]:
    """
    Computes rigorous multiclass performance metrics for a trained classifier.
    """
    y_pred = model.predict(X_test)
    y_prob = None
    if hasattr(model, "predict_proba"):
        try:
            y_prob = model.predict_proba(X_test)
        except Exception:
            pass

    acc = accuracy_score(y_test, y_pred)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (OvR)
    roc_auc = np.nan
    loss_val = np.nan
    if y_prob is not None:
        try:
            roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
            loss_val = log_loss(y_test, y_prob)
        except Exception:
            pass

    target_names = [DEFECT_CLASSES[i] for i in range(len(DEFECT_CLASSES))]
    clf_report = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    conf_mat = confusion_matrix(y_test, y_pred)

    return {
        "model_name": model_name,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "roc_auc_macro": roc_auc,
        "log_loss": loss_val,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "confusion_matrix": conf_mat,
        "classification_report": clf_report
    }


def benchmark_all_models(
    trained_models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates all models and compiles a comparative benchmark leaderboard.
    """
    detailed_results = {}
    summary_rows = []

    for name, model in trained_models.items():
        res = evaluate_single_model(model, X_test, y_test, name)
        detailed_results[name] = res
        summary_rows.append({
            "Model": name,
            "Accuracy": np.round(res["accuracy"], 4),
            "Balanced Acc": np.round(res["balanced_accuracy"], 4),
            "Macro Precision": np.round(res["precision_macro"], 4),
            "Macro Recall": np.round(res["recall_macro"], 4),
            "Macro F1": np.round(res["f1_macro"], 4),
            "Weighted F1": np.round(res["f1_weighted"], 4),
            "ROC-AUC (OvR)": np.round(res["roc_auc_macro"], 4) if not np.isnan(res["roc_auc_macro"]) else "N/A"
        })

    summary_df = pd.DataFrame(summary_rows).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return summary_df, detailed_results


if __name__ == "__main__":
    from physics_simulation import SemiconductorMetrologySimulator
    from preprocessing import prepare_dataset_splits
    from models import MetrologyClassifierTrainer

    sim = SemiconductorMetrologySimulator()
    raw_df = sim.generate_full_dataset(total_samples=1000)
    splits = prepare_dataset_splits(raw_df)

    trainer = MetrologyClassifierTrainer()
    trainer.train_all(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])

    leaderboard, _ = benchmark_all_models(trainer.trained_models, splits["X_test"], splits["y_test"])
    print("\n=== Model Benchmark Leaderboard ===")
    print(leaderboard.to_string(index=False))
