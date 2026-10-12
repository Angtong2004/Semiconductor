# Machine Learning-Based Semiconductor Defect Classification Using Simulated Metrology Data

A physics-informed machine learning pipeline for classifying semiconductor manufacturing defects using multi-modal metrology and characterization data. Inspired by cleanroom fabrication and characterization protocols at the **Centre for Nano Science and Engineering (CeNSE), Indian Institute of Science (IISc)**.

---

## 1. Project Background & Motivation

In advanced semiconductor foundries (e.g., TSMC, Intel, GlobalFoundries) and research cleanrooms (e.g., CeNSE NNFC/MNCF), process excursions during thin-film deposition, photolithography, plasma etching, and chemical mechanical planarization (CMP) generate yield-killing defects.

Rather than relying on destructive or slow manual failure analysis, inline automated metrology tools inspect wafers across multiple physical modalities:
- **Spectroscopic Ellipsometry / Reflectometry**: Measures dielectric/metallic film thickness, refractive index ($n$), extinction coefficient ($k$), and polarization phase shift ($\Delta$).
- **Atomic Force Microscopy (AFM) / Optical Profilometry**: Quantifies surface roughness ($R_a, R_q$), peak-to-valley height ($R_z$), surface skewness ($S_{sk}$), and kurtosis ($S_{ku}$).
- **In-Line Electrical Characterization**: Four-Point Probe sheet resistance ($R_s$), dielectric breakdown field ($E_{bd}$), gate leakage current ($J_{leak}$), and flatband voltage shift ($\Delta V_{fb}$).
- **Critical Dimension SEM (CD-SEM)**: Measures lithographic line-width deviation ($\Delta CD$), defect bounding area, and aspect ratio.

This project simulates physically accurate multi-modal metrology data across a 300 mm wafer and trains machine learning classifiers to diagnose defect root causes with high accuracy and explainability.

---

## 2. Defect Taxonomy & Fab Root Causes

| Class ID | Defect Category | Cleanroom Fab Root Cause | Primary Metrology Signatures |
| :---: | :--- | :--- | :--- |
| **0** | **Nominal Pass** | Baseline process within statistical process control (SPC) | $t \approx 100\text{ nm}$, $R_a < 0.3\text{ nm}$, $E_{bd} > 10\text{ MV/cm}$, $\Delta CD \approx 0$ |
| **1** | **Under-Etch** | Incomplete plasma/wet etch time or chemistry; residual film | Film thickness $+20\text{ to }+45\text{ nm}$, positive CD bias, step height offset |
| **2** | **Over-Etch** | Excessive RF power/etch duration; substrate pitting damage | Film thinned by $-25\text{ to }-55\text{ nm}$, negative $S_{sk}$, low breakdown |
| **3** | **Particulate Contamination** | Cleanroom airborne fallout or chamber wall flaking | AFM spike $R_z \gg 10\times$, kurtosis $S_{ku} \gg 5$, positive skewness $S_{sk} \gg 1.5$ |
| **4** | **Litho Defocus / Bridging** | Stepper lens field curvature or resist residue | Large positive $\Delta CD$, altered optical reflection, annular wafer ring pattern |
| **5** | **CMP Microscratch / Void** | Slurry agglomeration scratch or interfacial void | Extreme negative skewness $S_{sk} < -1$, high aspect ratio streak, dielectric leakage |
| **6** | **Incomplete Deposition** | Precursor starvation / CVD gas distribution non-uniformity | Severe thickness deficit, anomalous refractive index $n$, sheet resistance surge |

---

## 3. Project Architecture

```
Angtong/
├── data/
│   └── synthetic_metrology_data.csv       # 5,000 simulated dies with 23 physical features
├── outputs/
│   ├── figures/
│   │   ├── wafer_defect_map.png           # 300mm wafer spatial inspection visualization
│   │   ├── confusion_matrix.png           # Normalized multiclass confusion matrix
│   │   ├── feature_importance.png         # Domain-colored feature importances (MDI / Gain)
│   │   ├── roc_curves.png                 # Multiclass One-vs-Rest ROC curves
│   │   └── metrology_feature_distributions.png # Physical parameter boxplots across defects
│   ├── models/
│   │   ├── champion_model_RandomForest.joblib  # Trained model weights
│   │   └── metrology_preprocessor.joblib       # Fitted scaler & feature transformer
│   └── model_benchmark_leaderboard.csv
├── src/
│   ├── physics_simulation.py              # Physics-informed synthetic metrology generator
│   ├── preprocessing.py                   # Robust scaling, domain feature engineering, stratified splits
│   ├── models.py                          # ML model zoo (Random Forest, XGBoost, MLP, SVM, Logistic Reg)
│   ├── evaluation.py                      # Metrics, confusion matrix, ROC-AUC, benchmark runner
│   └── visualization.py                   # Matplotlib & Seaborn publication figure generators
├── predict_defect.py                      # Single-sample & batch inference script
├── run_pipeline.py                        # Single-command end-to-end pipeline runner
└── requirements.txt                       # Python dependencies
```

---

## 4. Machine Learning Benchmark Results

On a held-out test set ($N = 1,000$ dies) with realistic fab yield imbalance (64% nominal pass rate):

| Model | Accuracy | Balanced Accuracy | Macro Precision | Macro Recall | Macro F1 | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **XGBoost** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **Support Vector Machine (RBF)** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| **MLP Neural Network** | **0.9990** | **0.9976** | **0.9998** | **0.9976** | **0.9987** | **1.0000** |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

### Key Metrology Discriminators (Feature Importance Ranking)
1. `film_thickness_nm` (10.0%) — Critical for Under-Etch, Over-Etch, Incomplete Deposition.
2. `optical_phase_delta_deg` (8.3%) — Sensitive to optical path length and dielectric thickness.
3. `surface_skewness` (7.7%) — Differentiates CMP scratches (negative skewness) from particulate spikes (positive skewness).
4. `sheet_resistance_ohm_sq` (7.5%) — Electrical confirmation of film continuity and doping uniformity.
5. `v_flatband_shift_v` (6.9%) — Indicates plasma-induced gate damage ($Q_{ox}$).
6. `roughness_rz_nm` (6.8%) — Unambiguously catches particulate protrusion heights.

---

## 5. Quickstart & Usage

### Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the Full Pipeline
Generates synthetic data, trains all models, creates leaderboard, and plots all figures:
```bash
python run_pipeline.py --samples 5000
```

### Test Single-Die Inference
Test a hypothetical metrology reading:
```bash
python predict_defect.py
```
Output:
```text
--- Metrology Inference Result ---
Prediction: Particulate_Contamination (ID: 3)
Fab Root Cause: Airborne or chamber-shed particle; localized AFM height spike, high kurtosis/skewness, optical scatter.

Class Probabilities:
  Nominal_Pass                : 0.00%
  Particulate_Contamination   : 95.89%
  Litho_Defocus_Bridging      : 3.22%
  CMP_Microscratch_Void       : 0.36%
  Incomplete_Deposition       : 0.53%
```

---

## 6. Generated Visualizations
All high-resolution figures are exported to `outputs/figures/`:
- `wafer_defect_map.png`: 300 mm wafer die distribution with defect geometry patterns.
- `confusion_matrix.png`: Normalized confusion matrix showing classification precision and recall.
- `feature_importance.png`: Top 15 metrology parameters colored by physical domain (Optical, AFM, Electrical, CD-SEM).
- `roc_curves.png`: Multiclass receiver operating characteristic curves.
- `metrology_feature_distributions.png`: Boxplots showing how each metrology instrument responds to specific defects.
