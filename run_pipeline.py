"""
run_pipeline.py
End-to-end execution pipeline for Semiconductor Defect Metrology & ML Classification:
1. Simulates multi-modal metrology inspection data (Ellipsometry, AFM, 4PP, C-V, CD-SEM).
2. Performs robust feature engineering and stratified splitting.
3. Trains multiple ML algorithms (Random Forest, XGBoost, MLP, SVM, Logistic Regression).
4. Generates benchmark leaderboards, metrics, and error analysis.
5. Produces publication-grade visualizations (Wafer map, confusion matrix, ROC curves, feature importances).
"""

import os
import argparse
import numpy as np
import pandas as pd
from src.physics_simulation import SemiconductorMetrologySimulator, DEFECT_CLASSES
from src.preprocessing import prepare_dataset_splits
from src.models import MetrologyClassifierTrainer
from src.evaluation import benchmark_all_models
from src.visualization import (
    plot_wafer_map,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_multiclass_roc,
    plot_metrology_distribution_matrix
)


def run_defect_classification_pipeline(
    n_samples: int = 5000,
    balanced: bool = False,
    random_state: int = 42
):
    print("=" * 75)
    print("  SEMICONDUCTOR FABRICATION METROLOGY DEFECT CLASSIFICATION PIPELINE")
    print("  Centre for Nano Science and Engineering (CeNSE) Inspired Framework")
    print("=" * 75)

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # -------------------------------------------------------------------------
    # STEP 1: Synthetic Metrology Data Generation
    # -------------------------------------------------------------------------
    print(f"\n[1/5] Simulating {n_samples} inspection dies with multi-modal fab metrology...")
    sim = SemiconductorMetrologySimulator(random_state=random_state)
    raw_df = sim.generate_full_dataset(total_samples=n_samples, balanced=balanced)

    data_path = "data/synthetic_metrology_data.csv"
    raw_df.to_csv(data_path, index=False)
    print(f"      --> Dataset saved to: {data_path}")
    print("      Class Distribution:")
    for class_id, count in raw_df["defect_class_id"].value_counts().sort_index().items():
        pct = (count / len(raw_df)) * 100.0
        print(f"        [{class_id}] {DEFECT_CLASSES[class_id]:<28}: {count:>5} dies ({pct:5.1f}%)")

    # -------------------------------------------------------------------------
    # STEP 2: Preprocessing, Feature Engineering & Stratified Splitting
    # -------------------------------------------------------------------------
    print("\n[2/5] Performing domain feature engineering & stratified train/val/test split...")
    splits = prepare_dataset_splits(
        raw_df,
        test_size=0.20,
        val_size=0.15,
        random_state=random_state
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    feature_names = splits["feature_names"]
    preprocessor = splits["preprocessor"]

    # Save fitted preprocessor
    preprocessor.save("outputs/models/metrology_preprocessor.joblib")
    print(f"      --> Feature dimensions: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    print(f"      --> Total features engineered: {len(feature_names)}")

    # -------------------------------------------------------------------------
    # STEP 3: Multi-Model Zoo Training
    # -------------------------------------------------------------------------
    print("\n[3/5] Training ML Model Zoo (Ensembles, Deep Neural Net, SVM, Baseline)...")
    trainer = MetrologyClassifierTrainer(random_state=random_state)
    trainer.train_all(X_train, y_train, X_val, y_val)

    # -------------------------------------------------------------------------
    # STEP 4: Evaluation Benchmarking
    # -------------------------------------------------------------------------
    print("\n[4/5] Evaluating models on held-out test set...")
    leaderboard, detailed_results = benchmark_all_models(trainer.trained_models, X_test, y_test)

    print("\n" + "=" * 80)
    print("  MODEL BENCHMARK LEADERBOARD (Ranked by Macro F1 Score)")
    print("=" * 80)
    print(leaderboard.to_string(index=False))
    print("=" * 80)

    # Identify champion model
    champion_name = leaderboard.iloc[0]["Model"]
    champion_res = detailed_results[champion_name]
    print(f"\n🏆 Champion Model: {champion_name} (Macro F1 = {champion_res['f1_macro']:.4f}, Accuracy = {champion_res['accuracy']:.4f})")

    # Save champion model
    champion_model_path = f"outputs/models/champion_model_{champion_name}.joblib"
    trainer.save_model(champion_name, champion_model_path)
    print(f"   --> Saved champion model to: {champion_model_path}")

    # Save leaderboard CSV
    leaderboard.to_csv("outputs/model_benchmark_leaderboard.csv", index=False)

    # -------------------------------------------------------------------------
    # STEP 5: Visualizations & Engineering Diagnostics
    # -------------------------------------------------------------------------
    print("\n[5/5] Generating publication-grade metrology & ML diagnostic figures...")

    # 1. 300mm Wafer Inspection Map
    plot_wafer_map(raw_df, save_path="outputs/figures/wafer_defect_map.png")

    # 2. Confusion Matrix for Champion Model
    plot_confusion_matrix(
        champion_res["confusion_matrix"],
        save_path="outputs/figures/confusion_matrix.png",
        model_name=champion_name
    )

    # 3. Feature Importance Plot
    rf_model = trainer.trained_models.get("RandomForest")
    if rf_model is not None:
        plot_feature_importance(
            feature_names,
            rf_model.feature_importances_,
            save_path="outputs/figures/feature_importance.png",
            top_n=15,
            model_name="RandomForest"
        )

    # 4. Multiclass ROC Curves
    if champion_res["y_prob"] is not None:
        plot_multiclass_roc(
            y_test,
            champion_res["y_prob"],
            save_path="outputs/figures/roc_curves.png",
            model_name=champion_name
        )

    # 5. Metrology Feature Distribution Matrix across defect classes
    plot_metrology_distribution_matrix(
        raw_df,
        save_path="outputs/figures/metrology_feature_distributions.png"
    )

    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETE! All artifacts written to outputs/")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semiconductor Metrology Defect Classification Pipeline")
    parser.add_argument("--samples", type=int, default=5000, help="Total dies to simulate (default: 5000)")
    parser.add_argument("--balanced", action="store_true", help="Generate balanced class distribution (default: realistic yield imbalance)")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed (default: 42)")
    args = parser.parse_args()

    run_defect_classification_pipeline(
        n_samples=args.samples,
        balanced=args.balanced,
        random_state=args.seed
    )
