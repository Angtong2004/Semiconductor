"""
visualization.py
Generates publication-quality semiconductor metrology and ML classification visualizations:
1. Wafer Map Defect Spatial Distribution (300mm wafer coordinate projection)
2. Normalized Multiclass Confusion Matrix Heatmap
3. Feature Importance Bar Plot (Random Forest / XGBoost)
4. Metrology Feature Distributions by Defect Class (Violin / Box plots)
5. Multiclass ROC Curves (One-vs-Rest)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List
from sklearn.metrics import roc_curve, auc
try:
    from src.physics_simulation import DEFECT_CLASSES
except (ImportError, ModuleNotFoundError):
    from physics_simulation import DEFECT_CLASSES


# Clean scientific aesthetic
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
CLASS_PALETTE = {
    0: "#2b8a3e",  # Nominal Pass: Green
    1: "#1971c2",  # Under-Etch: Blue
    2: "#e8590c",  # Over-Etch: Orange
    3: "#d6336c",  # Particulate Contamination: Pink/Magenta
    4: "#7048e8",  # Litho Defocus: Purple
    5: "#c92a2a",  # CMP Microscratch: Red
    6: "#f59f00"   # Incomplete Deposition: Amber
}


def plot_wafer_map(
    df: pd.DataFrame,
    save_path: str = "outputs/figures/wafer_defect_map.png",
    wafer_radius_mm: float = 150.0
):
    """
    Plots the spatial defect distribution on a 300mm wafer canvas.
    """
    fig, ax = plt.subplots(figsize=(9, 9), dpi=300)

    # Draw wafer outer edge and notch
    wafer_circle = plt.Circle((0, 0), wafer_radius_mm, color="#dee2e6", fill=True, alpha=0.35, zorder=1)
    ax.add_patch(wafer_circle)
    wafer_outline = plt.Circle((0, 0), wafer_radius_mm, color="#495057", fill=False, linewidth=2, zorder=2)
    ax.add_patch(wafer_outline)

    # Draw wafer notch / flat at bottom (theta = -pi/2)
    ax.plot([0, 0], [-wafer_radius_mm, -wafer_radius_mm + 4.0], color="#212529", lw=3.0, zorder=3)

    # Plot dies
    for class_id, label in DEFECT_CLASSES.items():
        sub = df[df["defect_class_id"] == class_id]
        if len(sub) == 0:
            continue
        alpha = 0.4 if class_id == 0 else 0.85
        size = 18 if class_id == 0 else 45
        marker = "o" if class_id == 0 else "^"
        ax.scatter(
            sub["wafer_x_mm"],
            sub["wafer_y_mm"],
            c=CLASS_PALETTE[class_id],
            label=f"{label} (n={len(sub)})",
            alpha=alpha,
            s=size,
            marker=marker,
            edgecolors="none" if class_id == 0 else "black",
            linewidths=0.5,
            zorder=4 if class_id == 0 else 5
        )

    ax.set_xlim(-wafer_radius_mm - 12, wafer_radius_mm + 12)
    ax.set_ylim(-wafer_radius_mm - 12, wafer_radius_mm + 12)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("300mm Silicon Wafer Metrology Inspection Map", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Wafer X Position (mm)", fontsize=11)
    ax.set_ylabel("Wafer Y Position (mm)", fontsize=11)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.0), frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Saved wafer map: {save_path}")


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    save_path: str = "outputs/figures/confusion_matrix.png",
    model_name: str = "Classifier"
):
    """
    Plots normalized confusion matrix heatmap.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    # Row normalize (Recall per true class)
    row_sums = conf_mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm_mat = conf_mat / row_sums

    class_names = [DEFECT_CLASSES[i] for i in range(len(DEFECT_CLASSES))]

    sns.heatmap(
        norm_mat,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Normalized Recall"},
        ax=ax
    )

    ax.set_title(f"Normalized Confusion Matrix - {model_name}", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted Defect Class", fontsize=11)
    ax.set_ylabel("True Ground Truth Class", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Saved confusion matrix: {save_path}")


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    save_path: str = "outputs/figures/feature_importance.png",
    top_n: int = 15,
    model_name: str = "RandomForest"
):
    """
    Plots top feature importances with domain coloring.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_scores = importances[indices]

    # Domain color coding
    colors = []
    for f in top_features:
        if any(k in f for k in ["film_thickness", "refractive", "extinction", "optical", "reflectance"]):
            colors.append("#1971c2")  # Optical / Ellipsometry
        elif any(k in f for k in ["roughness", "skewness", "kurtosis"]):
            colors.append("#2b8a3e")  # AFM / Surface
        elif any(k in f for k in ["resistance", "breakdown", "leakage", "flatband"]):
            colors.append("#e8590c")  # Electrical
        elif any(k in f for k in ["cd_bias", "defect_area", "aspect_ratio", "coupling"]):
            colors.append("#7048e8")  # CD-SEM / Defect
        else:
            colors.append("#495057")  # Spatial

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_scores, color=colors, edgecolor="black", linewidth=0.5, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Relative Feature Importance (MDI / Gain)", fontsize=11)
    ax.set_title(f"Top {top_n} Metrology Feature Importances ({model_name})", fontsize=13, fontweight="bold", pad=12)

    # Custom legend for metrology domains
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1971c2", label="Ellipsometry & Optical"),
        Patch(facecolor="#2b8a3e", label="AFM & Surface Roughness"),
        Patch(facecolor="#e8590c", label="Electrical & C-V"),
        Patch(facecolor="#7048e8", label="CD-SEM & Defect Geometry"),
        Patch(facecolor="#495057", label="Spatial Coordinates")
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Saved feature importance plot: {save_path}")


def plot_multiclass_roc(
    y_test: np.ndarray,
    y_prob: np.ndarray,
    save_path: str = "outputs/figures/roc_curves.png",
    model_name: str = "Classifier"
):
    """
    Plots multiclass One-vs-Rest ROC curves.
    """
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    n_classes = len(DEFECT_CLASSES)

    for i in range(n_classes):
        # Binary target for class i
        y_binary = (y_test == i).astype(int)
        if np.sum(y_binary) == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        roc_score = auc(fpr, tpr)
        ax.plot(
            fpr,
            tpr,
            label=f"{DEFECT_CLASSES[i]} (AUC = {roc_score:.3f})",
            color=CLASS_PALETTE[i],
            linewidth=2.0
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
    ax.set_title(f"Multiclass One-vs-Rest ROC Curves ({model_name})", fontsize=13, fontweight="bold", pad=12)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Saved ROC curves: {save_path}")


def plot_metrology_distribution_matrix(
    df: pd.DataFrame,
    save_path: str = "outputs/figures/metrology_feature_distributions.png"
):
    """
    Plots box plots comparing core metrology features across defect categories.
    """
    features_to_plot = [
        ("film_thickness_nm", "Film Thickness (nm)\n[Ellipsometry]"),
        ("roughness_rz_nm", "Roughness Rz (nm)\n[AFM Topography]"),
        ("surface_skewness", "Surface Skewness\n[AFM Topography]"),
        ("sheet_resistance_ohm_sq", "Sheet Res. (Ω/□)\n[4-Point Probe]"),
        ("breakdown_field_mv_cm", "Breakdown (MV/cm)\n[C-V Metrology]"),
        ("cd_bias_nm", "CD Bias (nm)\n[CD-SEM]")
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    axes = axes.flatten()

    for idx, (col, label) in enumerate(features_to_plot):
        ax = axes[idx]
        sns.boxplot(
            data=df,
            x="defect_class_name",
            y=col,
            hue="defect_class_name",
            palette=list(CLASS_PALETTE.values()),
            ax=ax,
            legend=False,
            showfliers=False
        )
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
        ax.tick_params(labelsize=8.5)

    plt.suptitle("Multi-Modal Metrology Signatures Across Defect Taxonomies", fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"--> Saved metrology distribution matrix: {save_path}")
