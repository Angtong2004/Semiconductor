"""
physics_simulation.py
Simulates realistic semiconductor metrology inspection data across multiple fab characterization domains:
- Spectroscopic Ellipsometry & Reflectometry (Thin-film thickness, refractive index, optical phase shift)
- Atomic Force Microscopy / Profilometry (Surface roughness, peak-to-valley, skewness, kurtosis)
- In-line Electrical Metrology (Sheet resistance 4-point probe, C-V dielectric breakdown, leakage current)
- Critical Dimension SEM (CD bias, defect area, aspect ratio)
- Wafer Spatial Map Coordinates (Wafer radius, angle, spatial defect signatures)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional

# Defect class labels mapping to process excursions
DEFECT_CLASSES = {
    0: "Nominal_Pass",
    1: "Under_Etch",
    2: "Over_Etch",
    3: "Particulate_Contamination",
    4: "Litho_Defocus_Bridging",
    5: "CMP_Microscratch_Void",
    6: "Incomplete_Deposition"
}

DEFECT_DESCRIPTIONS = {
    "Nominal_Pass": "Die within design specs; baseline process uniformity.",
    "Under_Etch": "Insufficient plasma etch time/chemistry; residual dielectric/metal layer causing elevated thickness & positive step.",
    "Over_Etch": "Excessive etch duration/RF power; substrate over-recess, elevated roughness, CD thinning.",
    "Particulate_Contamination": "Airborne or chamber-shed particle; localized AFM height spike, high kurtosis/skewness, optical scatter.",
    "Litho_Defocus_Bridging": "Stepper focal plane drift or resist residue; critical dimension (CD) expansion and micro-bridging.",
    "CMP_Microscratch_Void": "Slurry agglomeration scratch or pinhole void; severe negative surface skewness, localized dielectric leakage.",
    "Incomplete_Deposition": "Precursor starvation / CVD un-uniformity; sub-target thickness, anomalous refractive index, elevated sheet resistance."
}


class SemiconductorMetrologySimulator:
    """
    Physics-informed synthetic generator for semiconductor metrology datasets.
    Simulates physical tool measurements with realistic fab process physics,
    Gaussian measurement noise, instrument limits, and process variation.
    """

    def __init__(self, random_state: int = 42, wafer_radius_mm: float = 150.0):
        self.random_state = random_state
        self.wafer_radius_mm = wafer_radius_mm
        self.rng = np.random.default_rng(seed=random_state)

    def _sample_wafer_coordinates(self, n_samples: int, pattern: str = "random") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples (X, Y) coordinates on a circular wafer of radius wafer_radius_mm.
        Returns x_mm, y_mm, normalized_radius r/R in [0, 1].
        """
        if pattern == "edge":
            # Biased toward wafer edge (e.g., spin-coat edge bead or clamp ring effects)
            r_norm = self.rng.beta(a=4.0, b=1.5, size=n_samples)
        elif pattern == "center":
            # Biased toward center (e.g., gas inlet showerhead center stagnation)
            r_norm = self.rng.beta(a=1.2, b=4.0, size=n_samples)
        elif pattern == "ring":
            # Annular ring defect (e.g., thermal gradient in RTA or plasma ring)
            r_norm = np.clip(self.rng.normal(loc=0.65, scale=0.10, size=n_samples), 0.05, 0.98)
        elif pattern == "scratch":
            # Linear streak across wafer (CMP scratch)
            t = self.rng.uniform(-0.8, 0.8, size=n_samples)
            angle = self.rng.uniform(0, np.pi)
            x = t * np.cos(angle) + self.rng.normal(0, 0.03, size=n_samples)
            y = t * np.sin(angle) + self.rng.normal(0, 0.03, size=n_samples)
            r_norm = np.clip(np.sqrt(x**2 + y**2), 0.0, 0.98)
            theta = np.arctan2(y, x)
            return x * self.wafer_radius_mm, y * self.wafer_radius_mm, r_norm
        else:
            # Uniform area distribution across circular wafer: r = sqrt(uniform(0, 1))
            r_norm = np.sqrt(self.rng.uniform(0.0, 0.96, size=n_samples))

        theta = self.rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
        x_mm = r_norm * self.wafer_radius_mm * np.cos(theta)
        y_mm = r_norm * self.wafer_radius_mm * np.sin(theta)
        return x_mm, y_mm, r_norm

    def generate_class_samples(self, defect_label: int, n_samples: int) -> pd.DataFrame:
        """
        Simulates n_samples of metrology inspection measurements for a specific defect class.
        """
        class_name = DEFECT_CLASSES[defect_label]

        # -------------------------------------------------------------
        # 1. Spatial Wafer Geometry
        # -------------------------------------------------------------
        if defect_label == 0:  # Nominal Pass
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="random")
        elif defect_label == 1:  # Under-Etch (often center-slow or edge-slow RIE)
            pattern = self.rng.choice(["center", "edge", "random"])
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern=pattern)
        elif defect_label == 2:  # Over-Etch (often edge plasma concentration)
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="edge")
        elif defect_label == 3:  # Particulate Contamination (random or showerhead fall-out)
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="random")
        elif defect_label == 4:  # Litho Defocus (field curvature / annular tilt)
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="ring")
        elif defect_label == 5:  # CMP Microscratch / Void (linear polishing tracks)
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="scratch")
        elif defect_label == 6:  # Incomplete Deposition (precursor edge starvation)
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="edge")
        else:
            x_mm, y_mm, r_norm = self._sample_wafer_coordinates(n_samples, pattern="random")

        # -------------------------------------------------------------
        # 2. Spectroscopic Ellipsometry & Reflectometry
        # Baseline Target (e.g. SiO2 or Si3N4 thin film on Si substrate):
        # Target nominal thickness t_0 = 100.0 nm, n_633 = 1.458, k_633 = 0.000
        # -------------------------------------------------------------
        t_base = 100.0  # nm
        n_base = 1.458  # refractive index at 632.8 nm
        k_base = 0.001  # extinction coefficient

        # Process baseline spatial radial bowl effect (~2% across wafer)
        radial_effect_nm = 2.0 * (r_norm**2)

        if defect_label == 0:  # Nominal Pass
            film_thickness = self.rng.normal(t_base, 1.2, size=n_samples) + radial_effect_nm
            refractive_index = self.rng.normal(n_base, 0.005, size=n_samples)
            extinction_k = np.clip(self.rng.normal(k_base, 0.0008, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(168.0, 1.5, size=n_samples)
            reflectance_vis = self.rng.normal(32.5, 0.8, size=n_samples)
            reflectance_uv = self.rng.normal(48.2, 0.9, size=n_samples)

        elif defect_label == 1:  # Under-Etch (Film too thick, step incomplete)
            thickness_excess = self.rng.uniform(18.0, 45.0, size=n_samples)
            film_thickness = t_base + thickness_excess + self.rng.normal(0, 1.5, size=n_samples)
            refractive_index = self.rng.normal(n_base, 0.008, size=n_samples)
            extinction_k = np.clip(self.rng.normal(k_base, 0.001, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(142.0, 3.5, size=n_samples)
            reflectance_vis = self.rng.normal(41.0, 1.5, size=n_samples)
            reflectance_uv = self.rng.normal(55.0, 1.8, size=n_samples)

        elif defect_label == 2:  # Over-Etch (Substrate over-recessed, film thinned)
            thickness_loss = self.rng.uniform(22.0, 55.0, size=n_samples)
            film_thickness = np.clip(t_base - thickness_loss + self.rng.normal(0, 1.5, size=n_samples), 10.0, None)
            refractive_index = self.rng.normal(n_base - 0.02, 0.012, size=n_samples)
            extinction_k = np.clip(self.rng.normal(0.008, 0.003, size=n_samples), 0.0, None)  # slight damage absorption
            optical_phase_delta = self.rng.normal(182.0, 4.0, size=n_samples)
            reflectance_vis = self.rng.normal(24.0, 1.4, size=n_samples)
            reflectance_uv = self.rng.normal(38.0, 1.6, size=n_samples)

        elif defect_label == 3:  # Particulate Contamination
            # Particles cause diffuse scatter, anomalous optical model fit
            film_thickness = self.rng.normal(t_base + 6.0, 4.0, size=n_samples)
            refractive_index = self.rng.normal(n_base + 0.04, 0.025, size=n_samples)  # apparent n shifted
            extinction_k = np.clip(self.rng.normal(0.025, 0.01, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(160.0, 6.0, size=n_samples)
            reflectance_vis = self.rng.normal(27.0, 2.5, size=n_samples)  # scattered light loss
            reflectance_uv = self.rng.normal(39.0, 2.8, size=n_samples)

        elif defect_label == 4:  # Litho Defocus Bridging
            # Film thickness mostly nominal, but optical CD and diffraction changed
            film_thickness = self.rng.normal(t_base, 1.8, size=n_samples)
            refractive_index = self.rng.normal(n_base, 0.007, size=n_samples)
            extinction_k = np.clip(self.rng.normal(k_base, 0.001, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(167.0, 2.2, size=n_samples)
            reflectance_vis = self.rng.normal(35.5, 1.2, size=n_samples)  # modified grating reflectance
            reflectance_uv = self.rng.normal(51.0, 1.3, size=n_samples)

        elif defect_label == 5:  # CMP Microscratch / Void
            # Trench gouged out by abrasive slurry particles
            film_thickness = np.clip(self.rng.normal(t_base - 14.0, 3.5, size=n_samples), 20.0, None)
            refractive_index = self.rng.normal(n_base - 0.035, 0.015, size=n_samples)  # porous effective medium
            extinction_k = np.clip(self.rng.normal(0.006, 0.002, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(175.0, 4.5, size=n_samples)
            reflectance_vis = self.rng.normal(29.0, 2.0, size=n_samples)
            reflectance_uv = self.rng.normal(42.0, 2.2, size=n_samples)

        elif defect_label == 6:  # Incomplete Deposition
            # Severe under-deposition
            film_thickness = self.rng.uniform(35.0, 70.0, size=n_samples) + self.rng.normal(0, 1.5, size=n_samples)
            refractive_index = self.rng.normal(1.39, 0.03, size=n_samples)  # under-dense film
            extinction_k = np.clip(self.rng.normal(0.004, 0.002, size=n_samples), 0.0, None)
            optical_phase_delta = self.rng.normal(130.0, 5.0, size=n_samples)
            reflectance_vis = self.rng.normal(21.0, 1.8, size=n_samples)
            reflectance_uv = self.rng.normal(33.0, 2.0, size=n_samples)

        # -------------------------------------------------------------
        # 3. AFM & Surface Topography Metrology
        # Metrics: Ra (average roughness, nm), Rq (RMS roughness, nm),
        # Rz (peak-to-valley height, nm), Ssk (skewness), Sku (kurtosis)
        # -------------------------------------------------------------
        if defect_label == 0:  # Nominal Pass (ultra-smooth thermal/polished oxide)
            ra_nm = self.rng.normal(0.28, 0.03, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.28, 0.04, size=n_samples)
            rz_nm = rq_nm * self.rng.normal(4.2, 0.3, size=n_samples)
            skewness = self.rng.normal(0.02, 0.12, size=n_samples)
            kurtosis = self.rng.normal(3.01, 0.15, size=n_samples)

        elif defect_label == 1:  # Under-Etch
            ra_nm = self.rng.normal(0.48, 0.06, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.32, 0.05, size=n_samples)
            rz_nm = rq_nm * self.rng.normal(4.8, 0.4, size=n_samples)
            skewness = self.rng.normal(0.35, 0.20, size=n_samples)
            kurtosis = self.rng.normal(3.25, 0.22, size=n_samples)

        elif defect_label == 2:  # Over-Etch (plasma ion bombardment damage / micro-pitting)
            ra_nm = self.rng.normal(1.15, 0.18, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.45, 0.08, size=n_samples)
            rz_nm = rq_nm * self.rng.normal(6.5, 0.6, size=n_samples)
            skewness = self.rng.normal(-0.45, 0.25, size=n_samples)  # pits = negative skewness
            kurtosis = self.rng.normal(3.85, 0.35, size=n_samples)

        elif defect_label == 3:  # Particulate Contamination (huge positive height spikes!)
            ra_nm = self.rng.normal(2.80, 0.65, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.85, 0.20, size=n_samples)
            rz_nm = rq_nm * self.rng.uniform(12.0, 35.0, size=n_samples)  # prominent particle protrusion
            skewness = self.rng.normal(2.60, 0.55, size=n_samples)   # strong positive skewness
            kurtosis = self.rng.normal(8.50, 1.80, size=n_samples)   # extreme heavy tail (leptokurtic)

        elif defect_label == 4:  # Litho Defocus Bridging
            ra_nm = self.rng.normal(0.62, 0.08, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.35, 0.06, size=n_samples)
            rz_nm = rq_nm * self.rng.normal(5.1, 0.45, size=n_samples)
            skewness = self.rng.normal(0.55, 0.22, size=n_samples)
            kurtosis = self.rng.normal(3.40, 0.25, size=n_samples)

        elif defect_label == 5:  # CMP Microscratch / Void (deep trenches/grooves)
            ra_nm = self.rng.normal(1.65, 0.25, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.60, 0.12, size=n_samples)
            rz_nm = rq_nm * self.rng.uniform(8.0, 20.0, size=n_samples)
            skewness = self.rng.normal(-2.10, 0.40, size=n_samples)  # strong negative skewness!
            kurtosis = self.rng.normal(6.20, 1.10, size=n_samples)

        elif defect_label == 6:  # Incomplete Deposition
            ra_nm = self.rng.normal(0.85, 0.14, size=n_samples)
            rq_nm = ra_nm * self.rng.normal(1.38, 0.07, size=n_samples)
            rz_nm = rq_nm * self.rng.normal(5.4, 0.5, size=n_samples)
            skewness = self.rng.normal(-0.15, 0.20, size=n_samples)
            kurtosis = self.rng.normal(3.10, 0.20, size=n_samples)

        # -------------------------------------------------------------
        # 4. Electrical Metrology (4-Point Probe Sheet Resistance & C-V)
        # Metrics:
        # Sheet resistance Rs (Ohm/sq) [Target ~ 25.0 Ohm/sq for gate layer]
        # Dielectric Breakdown Field E_bd (MV/cm) [Nominal SiO2 ~ 10-12 MV/cm]
        # Gate Leakage Current Density J_leak (nA/cm2 at 1V) [Nominal < 0.1 nA/cm2]
        # Flatband voltage shift Delta_Vfb (V) [Nominal ~ 0.0 V]
        # -------------------------------------------------------------
        rs_target = 25.0

        if defect_label == 0:  # Nominal Pass
            rs_ohm_sq = self.rng.normal(rs_target, 0.6, size=n_samples)
            e_bd_mv_cm = self.rng.normal(11.2, 0.4, size=n_samples)
            log_j_leak = self.rng.normal(-1.2, 0.25, size=n_samples)  # log10(nA/cm2) -> ~0.06 nA/cm2
            v_fb_shift = self.rng.normal(0.02, 0.03, size=n_samples)

        elif defect_label == 1:  # Under-Etch (un-cleared residue causes bridging/shorts or high capacitance)
            rs_ohm_sq = self.rng.normal(rs_target - 3.5, 0.9, size=n_samples)
            e_bd_mv_cm = self.rng.normal(10.5, 0.6, size=n_samples)
            log_j_leak = self.rng.normal(0.4, 0.35, size=n_samples)
            v_fb_shift = self.rng.normal(0.18, 0.08, size=n_samples)

        elif defect_label == 2:  # Over-Etch (thinned gate oxide / plasma-induced damage Q_ox)
            rs_ohm_sq = self.rng.normal(rs_target + 5.5, 1.2, size=n_samples)
            e_bd_mv_cm = self.rng.normal(6.8, 0.8, size=n_samples)  # degradation
            log_j_leak = self.rng.normal(1.9, 0.45, size=n_samples)
            v_fb_shift = self.rng.normal(-0.45, 0.12, size=n_samples)  # positive trapped oxide charge

        elif defect_label == 3:  # Particulate Contamination
            rs_ohm_sq = self.rng.normal(rs_target + 1.2, 1.5, size=n_samples)
            e_bd_mv_cm = self.rng.normal(5.2, 1.2, size=n_samples)  # local field enhancement pinhole
            log_j_leak = self.rng.normal(2.4, 0.60, size=n_samples)
            v_fb_shift = self.rng.normal(0.25, 0.15, size=n_samples)

        elif defect_label == 4:  # Litho Defocus Bridging (line-to-line micro-shorts)
            rs_ohm_sq = self.rng.normal(rs_target - 6.2, 1.1, size=n_samples)  # reduced resistance
            e_bd_mv_cm = self.rng.normal(9.8, 0.7, size=n_samples)
            log_j_leak = self.rng.normal(1.1, 0.40, size=n_samples)
            v_fb_shift = self.rng.normal(0.05, 0.05, size=n_samples)

        elif defect_label == 5:  # CMP Microscratch / Void (severed dielectric barrier)
            rs_ohm_sq = self.rng.normal(rs_target + 3.8, 1.4, size=n_samples)
            e_bd_mv_cm = self.rng.normal(3.8, 0.9, size=n_samples)  # severe breakdown collapse
            log_j_leak = self.rng.normal(3.5, 0.55, size=n_samples)  # massive leakage spike
            v_fb_shift = self.rng.normal(-0.60, 0.18, size=n_samples)

        elif defect_label == 6:  # Incomplete Deposition
            rs_ohm_sq = self.rng.normal(rs_target + 14.5, 2.5, size=n_samples)  # high resistance
            e_bd_mv_cm = self.rng.normal(4.5, 0.8, size=n_samples)
            log_j_leak = self.rng.normal(2.8, 0.50, size=n_samples)
            v_fb_shift = self.rng.normal(0.75, 0.20, size=n_samples)

        # -------------------------------------------------------------
        # 5. CD-SEM & Defect Morphology Metrology
        # Metrics:
        # cd_bias_nm: Critical Dimension deviation from target (nominal 45.0 nm feature, bias = CD - 45.0)
        # defect_area_um2: Estimated optical/SEM defect cluster area
        # aspect_ratio: Defect bounding box aspect ratio
        # -------------------------------------------------------------
        if defect_label == 0:  # Nominal Pass
            cd_bias_nm = self.rng.normal(0.05, 0.45, size=n_samples)
            defect_area_um2 = np.clip(self.rng.normal(0.02, 0.01, size=n_samples), 0.001, None)
            aspect_ratio = self.rng.normal(1.05, 0.08, size=n_samples)

        elif defect_label == 1:  # Under-Etch (lines broader than target)
            cd_bias_nm = self.rng.normal(6.5, 1.2, size=n_samples)
            defect_area_um2 = self.rng.uniform(0.8, 3.5, size=n_samples)
            aspect_ratio = self.rng.normal(1.3, 0.2, size=n_samples)

        elif defect_label == 2:  # Over-Etch (lines narrower than target)
            cd_bias_nm = self.rng.normal(-7.2, 1.4, size=n_samples)
            defect_area_um2 = self.rng.uniform(0.6, 2.8, size=n_samples)
            aspect_ratio = self.rng.normal(1.4, 0.25, size=n_samples)

        elif defect_label == 3:  # Particulate Contamination
            cd_bias_nm = self.rng.normal(2.1, 2.0, size=n_samples)
            defect_area_um2 = self.rng.uniform(2.5, 18.0, size=n_samples)  # large cluster
            aspect_ratio = self.rng.normal(1.15, 0.15, size=n_samples)  # roughly circular

        elif defect_label == 4:  # Litho Defocus Bridging (line-width bloat)
            cd_bias_nm = self.rng.normal(9.8, 1.8, size=n_samples)
            defect_area_um2 = self.rng.uniform(1.2, 5.0, size=n_samples)
            aspect_ratio = self.rng.normal(2.2, 0.45, size=n_samples)  # elongated bridging

        elif defect_label == 5:  # CMP Microscratch / Void
            cd_bias_nm = self.rng.normal(-1.5, 1.5, size=n_samples)
            defect_area_um2 = self.rng.uniform(1.5, 8.5, size=n_samples)
            aspect_ratio = self.rng.uniform(4.5, 12.0, size=n_samples)  # elongated high aspect scratch track!

        elif defect_label == 6:  # Incomplete Deposition
            cd_bias_nm = self.rng.normal(-4.8, 1.6, size=n_samples)
            defect_area_um2 = self.rng.uniform(3.0, 15.0, size=n_samples)
            aspect_ratio = self.rng.normal(1.6, 0.35, size=n_samples)

        # Build DataFrame
        df = pd.DataFrame({
            # Spatial coordinates
            "wafer_x_mm": np.round(x_mm, 2),
            "wafer_y_mm": np.round(y_mm, 2),
            "normalized_radius": np.round(r_norm, 4),

            # Spectroscopic Ellipsometry & Reflectometry
            "film_thickness_nm": np.round(film_thickness, 2),
            "refractive_index_633nm": np.round(refractive_index, 4),
            "extinction_coeff_k": np.round(extinction_k, 5),
            "optical_phase_delta_deg": np.round(optical_phase_delta, 2),
            "reflectance_vis_pct": np.round(reflectance_vis, 2),
            "reflectance_uv_pct": np.round(reflectance_uv, 2),

            # AFM & Surface Metrology
            "roughness_ra_nm": np.round(np.clip(ra_nm, 0.05, None), 3),
            "roughness_rq_nm": np.round(np.clip(rq_nm, 0.08, None), 3),
            "roughness_rz_nm": np.round(np.clip(rz_nm, 0.2, None), 2),
            "surface_skewness": np.round(skewness, 3),
            "surface_kurtosis": np.round(np.clip(kurtosis, 1.5, None), 3),

            # Electrical In-line Metrology
            "sheet_resistance_ohm_sq": np.round(np.clip(rs_ohm_sq, 1.0, None), 2),
            "breakdown_field_mv_cm": np.round(np.clip(e_bd_mv_cm, 0.5, None), 2),
            "log10_leakage_current_na": np.round(log_j_leak, 3),
            "v_flatband_shift_v": np.round(v_fb_shift, 3),

            # CD-SEM & Defect Morphology
            "cd_bias_nm": np.round(cd_bias_nm, 2),
            "defect_area_um2": np.round(defect_area_um2, 3),
            "defect_aspect_ratio": np.round(aspect_ratio, 2),

            # Ground Truth Targets
            "defect_class_id": defect_label,
            "defect_class_name": class_name
        })

        return df

    def generate_full_dataset(
        self,
        total_samples: int = 5000,
        balanced: bool = False
    ) -> pd.DataFrame:
        """
        Generates full simulated wafer inspection dataset.
        If balanced=False, follows realistic fab yield distribution:
          - Nominal Pass: ~65%
          - Under-Etch: ~6%
          - Over-Etch: ~6%
          - Particulate Contamination: ~7%
          - Litho Defocus: ~5%
          - CMP Scratch / Void: ~5%
          - Incomplete Deposition: ~6%
        """
        if balanced:
            samples_per_class = total_samples // len(DEFECT_CLASSES)
            dfs = []
            for class_id in DEFECT_CLASSES.keys():
                dfs.append(self.generate_class_samples(class_id, samples_per_class))
            dataset = pd.concat(dfs, ignore_index=True)
        else:
            proportions = {
                0: 0.64,  # Nominal pass (typical baseline yield)
                1: 0.06,  # Under-Etch
                2: 0.06,  # Over-Etch
                3: 0.07,  # Particulate Contamination
                4: 0.05,  # Litho Defocus Bridging
                5: 0.06,  # CMP Microscratch / Void
                6: 0.06   # Incomplete Deposition
            }
            dfs = []
            for class_id, prop in proportions.items():
                count = int(total_samples * prop)
                dfs.append(self.generate_class_samples(class_id, count))
            dataset = pd.concat(dfs, ignore_index=True)

        # Shuffle rows
        dataset = dataset.sample(frac=1.0, random_state=self.random_state).reset_index(drop=True)
        return dataset


if __name__ == "__main__":
    sim = SemiconductorMetrologySimulator(random_state=42)
    sample_df = sim.generate_full_dataset(total_samples=3000, balanced=False)
    print("Simulated dataset summary:")
    print(sample_df["defect_class_name"].value_counts())
    print("\nFeature shape:", sample_df.shape)
    print(sample_df.head(3))
