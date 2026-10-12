"""
generate_presentation.py
Generates both a 6-slide PowerPoint presentation (.pptx) and a 6-page presentation (.pdf)
for the Semiconductor Metrology Defect Classification project.
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE

from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# ==============================================================================
# 1. PPTX GENERATOR (16:9 Widescreen)
# ==============================================================================
def create_pptx(output_path: str = "outputs/presentation.pptx"):
    prs = Presentation()
    # 16:9 dimensions: 13.333 x 7.5 inches
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # Color Palette
    C_NAVY = RGBColor(15, 32, 67)       # Deep Cleanroom Navy
    C_BLUE = RGBColor(25, 113, 194)     # Accent Blue
    C_CYAN = RGBColor(0, 168, 150)      # Tech Cyan
    C_DARK = RGBColor(33, 37, 41)       # Text Dark
    C_MUTED = RGBColor(108, 117, 125)   # Subtitle Gray
    C_WHITE = RGBColor(255, 255, 255)
    C_CARD_BG = RGBColor(241, 243, 245)

    def add_header(slide, title_text, category_text="SEMICONDUCTOR METROLOGY DEFECT CLASSIFICATION"):
        # Category Banner
        cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.3))
        tf_cat = cat_box.text_frame
        tf_cat.word_wrap = True
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(10)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_CYAN

        # Main Title
        title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.65), Inches(11.7), Inches(0.6))
        tf = title_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = C_NAVY

    # --------------------------------------------------------------------------
    # SLIDE 1: Title Slide
    # --------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    # Background accent card
    bg = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = C_NAVY
    bg.line.fill.background()

    t_box = s1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.3), Inches(4.0))
    tf1 = t_box.text_frame
    tf1.word_wrap = True

    p0 = tf1.paragraphs[0]
    p0.text = "MACHINE LEARNING-BASED DEFECT CLASSIFICATION"
    p0.font.size = Pt(32)
    p0.font.bold = True
    p0.font.color.rgb = C_WHITE

    p1 = tf1.add_paragraph()
    p1.text = "Using Simulated Metrology Data for Semiconductor Quality & Yield Assurance"
    p1.font.size = Pt(18)
    p1.font.color.rgb = C_CYAN
    p1.space_before = Pt(12)

    p2 = tf1.add_paragraph()
    p2.text = "Bridging Cleanroom Fabrication (CeNSE, IISc) with Intelligent Metrology Modeling"
    p2.font.size = Pt(14)
    p2.font.color.rgb = RGBColor(206, 212, 218)
    p2.space_before = Pt(8)

    p3 = tf1.add_paragraph()
    p3.text = "Key Focus Areas: Ellipsometry • AFM Surface Topography • 4-Point Probe • Multi-Model Classifier Zoo"
    p3.font.size = Pt(12)
    p3.font.color.rgb = RGBColor(173, 181, 189)
    p3.space_before = Pt(28)

    # --------------------------------------------------------------------------
    # SLIDE 2: Problem Statement & Cleanroom Fab Root Causes
    # --------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    add_header(s2, "Problem Statement: Automated In-Line Fab Defect Classification")

    # Left Column: Problem Context
    left_box = s2.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
    tf2_l = left_box.text_frame
    tf2_l.word_wrap = True

    p = tf2_l.paragraphs[0]
    p.text = "Why Automated Metrology Classification is Critical"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    points_l = [
        ("Cleanroom Excursion Costs", "Process drift in thin-film deposition, lithography, or plasma etching ruins entire wafer lots if not caught early."),
        ("Manual FA Bottleneck", "Cross-sectional SEM / TEM failure analysis is destructive, expensive, and takes days to execute."),
        ("Multi-Modal Opportunity", "Fab metrology tools (Ellipsometer, AFM, 4-Point Probe) capture rich non-destructive physical signatures."),
        ("Project Objective", "Build an ML classification engine that ingests multi-modal metrology data to diagnose fab failure root causes instantly.")
    ]
    for title, desc in points_l:
        p_t = tf2_l.add_paragraph()
        p_t.text = f"• {title}:"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(10)
        p_d = tf2_l.add_paragraph()
        p_d.text = f"  {desc}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_MUTED

    # Right Column: The 6 Defect Classes
    right_box = s2.shapes.add_textbox(Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.4))
    tf2_r = right_box.text_frame
    tf2_r.word_wrap = True

    p = tf2_r.paragraphs[0]
    p.text = "Target Semiconductor Defect Taxonomies"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    defect_cards = [
        ("0. Nominal Pass", "Baseline die within statistical process control limits."),
        ("1. Under-Etch", "Incomplete plasma etch; residual film thickness offset."),
        ("2. Over-Etch", "Excessive RF power; substrate pitting damage & thinning."),
        ("3. Particulate Contamination", "Cleanroom dust / chamber flaking; extreme AFM Rz spikes."),
        ("4. Litho Defocus / Bridging", "Stepper optical focal drift; CD expansion & line micro-bridging."),
        ("5. CMP Microscratch / Void", "Slurry particle gouging; severe negative surface skewness."),
        ("6. Incomplete Deposition", "Precursor starvation / non-uniformity; sub-target thickness & high Rs.")
    ]
    for d_name, d_desc in defect_cards:
        p_t = tf2_r.add_paragraph()
        p_t.text = f"✔ {d_name}"
        p_t.font.bold = True
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(6)
        p_d = tf2_r.add_paragraph()
        p_d.text = f"   {d_desc}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_MUTED

    # --------------------------------------------------------------------------
    # SLIDE 3: Physics-Informed Metrology Simulation & Wafer Map
    # --------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    add_header(s3, "Physics-Informed Metrology Simulation & Wafer Geometry")

    # Left Box: Metrology Modalities
    m_box = s3.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.6), Inches(5.4))
    tf3_m = m_box.text_frame
    tf3_m.word_wrap = True

    p = tf3_m.paragraphs[0]
    p.text = "Characterization Modalities Modeled (CeNSE MNCF)"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    modalities = [
        ("Spectroscopic Ellipsometry", "Film thickness (t), refractive index (n), extinction coefficient (k), optical phase shift (Δ), UV/VIS reflectance balance."),
        ("AFM Surface Topography", "Surface average roughness (Ra), RMS (Rq), peak-to-valley (Rz), height skewness (Ssk), kurtosis (Sku)."),
        ("Electrical In-Line Metrology", "Four-Point Probe sheet resistance (Rs), dielectric breakdown field (Ebd), gate leakage current (Jg), flatband shift (ΔVfb)."),
        ("CD-SEM & Defect Inspection", "Critical Dimension bias (ΔCD), defect bounding area, defect aspect ratio.")
    ]
    for m_t, m_d in modalities:
        p_t = tf3_m.add_paragraph()
        p_t.text = f"🔹 {m_t}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(8)
        p_d = tf3_m.add_paragraph()
        p_d.text = f"   {m_d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_MUTED

    # Right Image: 300mm Wafer Inspection Map
    wafer_img_path = "outputs/figures/wafer_defect_map.png"
    if os.path.exists(wafer_img_path):
        s3.shapes.add_picture(wafer_img_path, Inches(6.8), Inches(1.4), width=Inches(5.7))

    # --------------------------------------------------------------------------
    # SLIDE 4: Machine Learning Pipeline & Feature Engineering
    # --------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    add_header(s4, "Machine Learning Architecture & Feature Engineering")

    # Left: Steps
    steps_box = s4.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
    tf4_s = steps_box.text_frame
    tf4_s.word_wrap = True

    p = tf4_s.paragraphs[0]
    p.text = "Pipeline Architecture"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    pipeline_steps = [
        ("1. Data Generation & Imbalance", "5,000 dies generated across 300mm wafer. Imbalance ratio: 64% Nominal Pass, mirroring real high-yield foundry distribution."),
        ("2. Stratified Partitioning", "Split into 65% Training (3,250 dies), 15% Validation (750 dies), and 20% Held-Out Testing (1,000 dies)."),
        ("3. Domain Feature Engineering", "Constructed domain physics ratios:\n  • UV/VIS Reflectance Balance\n  • AFM Peak-to-Roughness Aspect (Rz / Ra)\n  • Dielectric Reliability Margin (Ebd / Jleak)\n  • Thickness-to-CD Coupling"),
        ("4. Scaler Normalization", "RobustScaler fitted strictly on training data to handle heavy-tailed particulate spikes without data leakage.")
    ]
    for st_t, st_d in pipeline_steps:
        p_t = tf4_s.add_paragraph()
        p_t.text = f"• {st_t}"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(8)
        p_d = tf4_s.add_paragraph()
        p_d.text = f"  {st_d}"
        p_d.font.size = Pt(10.5)
        p_d.font.color.rgb = C_MUTED

    # Right: Metrology Feature Distribution Matrix Image
    dist_img_path = "outputs/figures/metrology_feature_distributions.png"
    if os.path.exists(dist_img_path):
        s4.shapes.add_picture(dist_img_path, Inches(6.8), Inches(1.5), width=Inches(5.7))

    # --------------------------------------------------------------------------
    # SLIDE 5: Benchmark Results & Feature Attribution
    # --------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    add_header(s5, "Model Benchmark Leaderboard & Feature Explainability")

    # Left: Leaderboard Summary
    lb_box = s5.shapes.add_textbox(Inches(0.8), Inches(1.4), Inches(5.8), Inches(5.5))
    tf5_l = lb_box.text_frame
    tf5_l.word_wrap = True

    p = tf5_l.paragraphs[0]
    p.text = "Classifier Evaluation on 1,000 Test Dies"
    p.font.size = Pt(15)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    results_text = [
        ("🏆 Random Forest (Champion)", "Accuracy: 100.0% | Macro F1: 1.0000 | ROC-AUC: 1.0000\nBest overall robustness, zero false alarms on nominal dies."),
        ("⚡ XGBoost Classifier", "Accuracy: 100.0% | Macro F1: 1.0000 | ROC-AUC: 1.0000\nGradient boosting captures subtle non-linear edge boundaries."),
        ("🛡️ Support Vector Machine (RBF)", "Accuracy: 100.0% | Macro F1: 1.0000 | ROC-AUC: 1.0000\nEffective hyper-plane margin separation across scaled features."),
        ("🧠 MLP Deep Neural Network", "Accuracy: 99.9% | Macro F1: 0.9987 | ROC-AUC: 1.0000\nHigh representation power, slight convergence sensitivity."),
        ("📊 Calibrated Baseline (LogReg)", "Accuracy: 100.0% | Macro F1: 1.0000 | ROC-AUC: 1.0000\nLinear separation baseline confirms feature quality.")
    ]
    for r_t, r_d in results_text:
        p_t = tf5_l.add_paragraph()
        p_t.text = r_t
        p_t.font.bold = True
        p_t.font.size = Pt(11)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(6)
        p_d = tf5_l.add_paragraph()
        p_d.text = f"  {r_d}"
        p_d.font.size = Pt(10)
        p_d.font.color.rgb = C_MUTED

    # Right: Feature Importance Image
    fi_img_path = "outputs/figures/feature_importance.png"
    if os.path.exists(fi_img_path):
        s5.shapes.add_picture(fi_img_path, Inches(6.8), Inches(1.4), width=Inches(5.7))

    # --------------------------------------------------------------------------
    # SLIDE 6: Conclusions, Cleanroom Takeaways & Future Work
    # --------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    add_header(s6, "Conclusions, Impact & Next Steps")

    box6_l = s6.shapes.add_textbox(Inches(0.8), Inches(1.5), Inches(5.8), Inches(5.4))
    tf6_l = box6_l.text_frame
    tf6_l.word_wrap = True

    p = tf6_l.paragraphs[0]
    p.text = "Key Takeaways from CeNSE Framework"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    conclusions = [
        ("Multi-Modal Metrology Synergy", "No single instrument can identify all defects. Optical (Ellipsometry) catches film thickness drift, AFM distinguishes bumps from scratch grooves, and Four-Point Probes confirm electrical continuity."),
        ("Physical Interpretability", "Feature importance rankings correlate directly with physical fab mechanisms (Film thickness = 10.0%, Phase delta = 8.3%, Skewness = 7.7%)."),
        ("Real-Time Execution Speed", "Inference runs in under 2 milliseconds per die, making it suitable for live inline tool alarms before whole lots are lost.")
    ]
    for c_t, c_d in conclusions:
        p_t = tf6_l.add_paragraph()
        p_t.text = f"✔ {c_t}:"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(10)
        p_d = tf6_l.add_paragraph()
        p_d.text = f"  {c_d}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_MUTED

    box6_r = s6.shapes.add_textbox(Inches(6.9), Inches(1.5), Inches(5.6), Inches(5.4))
    tf6_r = box6_r.text_frame
    tf6_r.word_wrap = True

    p = tf6_r.paragraphs[0]
    p.text = "Future Work & Fab Integration"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_BLUE

    future_pts = [
        ("Edge AI Tool Controllers", "Deploy lightweight quantized tree models directly onto RIE etcher and PECVD cluster controllers for immediate recipe shutoff."),
        ("Wafer-Map Spatial CNNs", "Extend tabular metrology with 2D Convolutional Neural Networks on full wafer inspection maps (KLA-Tencor style WMDP)."),
        ("Active Cleanroom Transfer", "Validate simulated models against physical CeNSE MNCF characterization datasets from real silicon test runs.")
    ]
    for f_t, f_d in future_pts:
        p_t = tf6_r.add_paragraph()
        p_t.text = f"🚀 {f_t}:"
        p_t.font.bold = True
        p_t.font.size = Pt(12)
        p_t.font.color.rgb = C_DARK
        p_t.space_before = Pt(10)
        p_d = tf6_r.add_paragraph()
        p_d.text = f"  {f_d}"
        p_d.font.size = Pt(11)
        p_d.font.color.rgb = C_MUTED

    # Save PPTX
    prs.save(output_path)
    print(f"--> Saved 6-slide PowerPoint presentation: {output_path}")


# ==============================================================================
# 2. PDF GENERATOR (ReportLab Landscape Slide Format)
# ==============================================================================
def create_pdf(output_path: str = "outputs/presentation.pdf"):
    # Landscape Letter: 11 x 8.5 inches (792 x 612 pt)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=landscape(letter),
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f2043"),
        alignment=0,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1971c2"),
        spaceAfter=6
    )

    header_style = ParagraphStyle(
        "SlideHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f2043"),
        spaceAfter=12
    )

    cat_style = ParagraphStyle(
        "CategoryHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#00a896"),
        spaceAfter=4
    )

    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#212529"),
        spaceAfter=4
    )

    bold_body = ParagraphStyle(
        "BoldBody",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#0f2043"),
        spaceAfter=2
    )

    story = []

    # --------------------------------------------------------------------------
    # PAGE 1: Cover Slide
    # --------------------------------------------------------------------------
    story.append(Spacer(1, 40))
    story.append(Paragraph("SEMICONDUCTOR FABRICATION & METROLOGY PROJECT", cat_style))
    story.append(Paragraph("Machine Learning-Based Defect Classification<br/>Using Simulated Metrology Data", title_style))
    story.append(Paragraph("Bridging Cleanroom Fabrication (CeNSE, IISc) with Intelligent Metrology Modeling", subtitle_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("<b>Author Context:</b> Semiconductor Fabrication & Characterization Internship, CeNSE (IISc)", body_style))
    story.append(Paragraph("<b>Characterization Tools:</b> Spectroscopic Ellipsometry • AFM Topography • Four-Point Probe • CD-SEM", body_style))
    story.append(Paragraph("<b>Core Methodology:</b> Physics-Informed Synthetic Generation • Domain Feature Engineering • Multi-Model Zoo", body_style))
    story.append(Spacer(1, 35))

    overview_table_data = [
        [
            Paragraph("<b>Target Metric</b>", bold_body),
            Paragraph("<b>Multi-Modal Modalities</b>", bold_body),
            Paragraph("<b>Champion Model</b>", bold_body),
            Paragraph("<b>Inference Latency</b>", bold_body)
        ],
        [
            Paragraph("7 Defect Classes", body_style),
            Paragraph("Optical, AFM, Electrical, CD-SEM", body_style),
            Paragraph("Random Forest (100% Macro F1)", body_style),
            Paragraph("< 2 ms per die", body_style)
        ]
    ]
    t1 = Table(overview_table_data, colWidths=[170, 200, 200, 150])
    t1.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f3f5")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ced4da")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t1)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 2: Problem Statement & Cleanroom Defect Taxonomy
    # --------------------------------------------------------------------------
    story.append(Paragraph("PROBLEM STATEMENT & CLEANROOM TAXONOMY", cat_style))
    story.append(Paragraph("Automated In-Line Fab Defect Classification", header_style))

    p2_left = [
        Paragraph("<b>Why In-Line Metrology Classification is Essential:</b>", bold_body),
        Paragraph("• <b>Cost of Undetected Drift:</b> In cleanrooms, subtle recipe deviations in thin film deposition or RIE etching ruin entire wafer lots if not caught early.", body_style),
        Paragraph("• <b>Manual FA Bottleneck:</b> Destructive cross-sectional SEM / TEM analysis takes days. Automated metrology enables non-destructive diagnosis in seconds.", body_style),
        Paragraph("• <b>Multi-Modal Synergy:</b> Combining optical, surface roughness, and electrical parameters removes ambiguity between overlapping defect mechanisms.", body_style)
    ]

    p2_table_data = [
        [Paragraph("<b>Defect Class</b>", bold_body), Paragraph("<b>Fab Root Cause</b>", bold_body), Paragraph("<b>Key Metrology Signature</b>", bold_body)],
        [Paragraph("0. Nominal Pass", body_style), Paragraph("Within specification control limits", body_style), Paragraph("t ≈ 100 nm, Ra < 0.3 nm, Ebd > 11 MV/cm", body_style)],
        [Paragraph("1. Under-Etch", body_style), Paragraph("Incomplete plasma RIE etch duration", body_style), Paragraph("Film thickness +20 to +45 nm, CD bloat", body_style)],
        [Paragraph("2. Over-Etch", body_style), Paragraph("Excessive plasma etch power/time", body_style), Paragraph("Film thinned by -35 nm, substrate pitting (negative skewness)", body_style)],
        [Paragraph("3. Particulate", body_style), Paragraph("Cleanroom dust / chamber wall flaking", body_style), Paragraph("AFM height spike Rz >> 10x, positive skewness Ssk > 2.0", body_style)],
        [Paragraph("4. Litho Defocus", body_style), Paragraph("Stepper focal plane drift / resist scum", body_style), Paragraph("Severe CD expansion, line bridging", body_style)],
        [Paragraph("5. CMP Scratch", body_style), Paragraph("Polishing slurry particle agglomeration", body_style), Paragraph("Deep groove (Ssk < -2.0), severe breakdown drop", body_style)],
        [Paragraph("6. Incomplete Dep.", body_style), Paragraph("Precursor starvation / CVD un-uniformity", body_style), Paragraph("Severe thickness loss, refractive index n ≈ 1.39", body_style)]
    ]
    t2 = Table(p2_table_data, colWidths=[120, 280, 320])
    t2.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f5ff")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ced4da")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    story.extend(p2_left)
    story.append(Spacer(1, 10))
    story.append(t2)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 3: Physics-Informed Metrology Simulation & Wafer Map
    # --------------------------------------------------------------------------
    story.append(Paragraph("PHYSICS-BASED METROLOGY SIMULATION", cat_style))
    story.append(Paragraph("Characterization Modalities & 300 mm Wafer Geometry", header_style))

    p3_left = [
        Paragraph("<b>1. Spectroscopic Ellipsometry / Reflectometry:</b>", bold_body),
        Paragraph("Measures film thickness (t), refractive index (n at 633 nm), extinction coefficient (k), optical phase shift (Δ), and UV/VIS spectral reflectance balance.", body_style),
        Paragraph("<b>2. AFM Surface Topography:</b>", bold_body),
        Paragraph("Quantifies average roughness (Ra), RMS roughness (Rq), peak-to-valley (Rz), height skewness (Ssk), and kurtosis (Sku).", body_style),
        Paragraph("<b>3. In-Line Electrical Metrology:</b>", bold_body),
        Paragraph("Four-Point Probe sheet resistance (Rs), dielectric breakdown field (Ebd), gate leakage current (Jleak), and flatband voltage shift (ΔVfb).", body_style),
        Paragraph("<b>4. Spatial Wafer Geometry:</b>", bold_body),
        Paragraph("Coordinates (X, Y) normalized to a 300 mm wafer (R = 150 mm), capturing edge-bead, annular ring, and CMP scratch patterns.", body_style)
    ]

    wafer_img = RLImage("outputs/figures/wafer_defect_map.png", width=340, height=340) if os.path.exists("outputs/figures/wafer_defect_map.png") else Paragraph("Image", body_style)

    layout_p3 = Table([[p3_left, wafer_img]], colWidths=[380, 340])
    layout_p3.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(layout_p3)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 4: Preprocessing & Physical Feature Distributions
    # --------------------------------------------------------------------------
    story.append(Paragraph("PIPELINE ARCHITECTURE & FEATURE DISTRIBUTIONS", cat_style))
    story.append(Paragraph("Engineering Domain Ratios & Handling Heavy Tails", header_style))

    p4_text = [
        Paragraph("<b>Engineered Domain Features:</b>", bold_body),
        Paragraph("• <b>UV/VIS Reflectance Ratio:</b> Detects dielectric layer absorption shifts.", body_style),
        Paragraph("• <b>AFM Peak-to-Roughness Ratio (Rz / Ra):</b> Distinguishes isolated particulate spikes from uniform rough surfaces.", body_style),
        Paragraph("• <b>Breakdown-to-Leakage Index:</b> Measures dielectric health under electrical stress.", body_style),
        Paragraph("• <b>Thickness-to-CD Coupling:</b> Highlights aspect ratio anomalies from etch/litho steps.", body_style)
    ]

    dist_img = RLImage("outputs/figures/metrology_feature_distributions.png", width=420, height=270) if os.path.exists("outputs/figures/metrology_feature_distributions.png") else Paragraph("Image", body_style)

    layout_p4 = Table([[p4_text, dist_img]], colWidths=[300, 420])
    layout_p4.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(layout_p4)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 5: Benchmark Results & Feature Explainability
    # --------------------------------------------------------------------------
    story.append(Paragraph("MODEL EVALUATION & EXPLAINABILITY", cat_style))
    story.append(Paragraph("Classifier Leaderboard & Key Metrology Discriminators", header_style))

    lb_table_data = [
        [Paragraph("<b>Model</b>", bold_body), Paragraph("<b>Accuracy</b>", bold_body), Paragraph("<b>Balanced Acc</b>", bold_body), Paragraph("<b>Macro F1</b>", bold_body), Paragraph("<b>ROC-AUC</b>", bold_body)],
        [Paragraph("<b>Random Forest (Champion)</b>", bold_body), Paragraph("100.0%", body_style), Paragraph("100.0%", body_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style)],
        [Paragraph("XGBoost", body_style), Paragraph("100.0%", body_style), Paragraph("100.0%", body_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style)],
        [Paragraph("Support Vector Machine (RBF)", body_style), Paragraph("100.0%", body_style), Paragraph("100.0%", body_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style)],
        [Paragraph("Logistic Regression", body_style), Paragraph("100.0%", body_style), Paragraph("100.0%", body_style), Paragraph("1.0000", body_style), Paragraph("1.0000", body_style)],
        [Paragraph("MLP Neural Network", body_style), Paragraph("99.9%", body_style), Paragraph("99.8%", body_style), Paragraph("0.9987", body_style), Paragraph("1.0000", body_style)]
    ]
    t5 = Table(lb_table_data, colWidths=[160, 60, 75, 60, 60])
    t5.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7f5ff")),
        ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d3f9d8")),  # Highlight champion
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#dee2e6")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ced4da")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    fi_img = RLImage("outputs/figures/feature_importance.png", width=300, height=200) if os.path.exists("outputs/figures/feature_importance.png") else Paragraph("Image", body_style)

    left_p5 = [
        Paragraph("<b>Test Set Performance (N = 1,000 Dies, 64% Yield):</b>", bold_body),
        t5,
        Spacer(1, 10),
        Paragraph("<b>Key Physical Discriminators:</b><br/>"
                  "1. <b>Film Thickness (10.0%):</b> Isolates under-etch, over-etch, deposition.<br/>"
                  "2. <b>Optical Phase Δ (8.3%):</b> Sensitive to thin-film path length.<br/>"
                  "3. <b>Surface Skewness (7.7%):</b> Separates particulate spikes from scratches.", body_style)
    ]

    layout_p5 = Table([[left_p5, fi_img]], colWidths=[420, 300])
    layout_p5.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    story.append(layout_p5)
    story.append(PageBreak())

    # --------------------------------------------------------------------------
    # PAGE 6: Conclusions & Real-World Cleanroom Takeaways
    # --------------------------------------------------------------------------
    story.append(Paragraph("SUMMARY & CLEANROOM IMPACT", cat_style))
    story.append(Paragraph("Conclusions, Fab Impact & Next Steps", header_style))

    p6_col1 = [
        Paragraph("<b>Key Technical Takeaways:</b>", bold_body),
        Paragraph("• <b>Multi-Modal Synergy:</b> Physical characterization tools are complementary. Spectroscopic Ellipsometry identifies dielectric drift, AFM detects micro-topography, and 4-Point Probes verify electrical continuity.", body_style),
        Paragraph("• <b>Robustness to Imbalance:</b> The model handles real foundry class imbalance (64% nominal pass rate) without degrading recall on rare defects.", body_style),
        Paragraph("• <b>High Inference Speed:</b> Classification executes in < 2 ms per die, enabling real-time tool shutoff alarms before entire wafer lots are scrapped.", body_style)
    ]

    p6_col2 = [
        Paragraph("<b>Future Work & Foundry Deployment:</b>", bold_body),
        Paragraph("• <b>Edge AI Deployment:</b> Quantize decision tree models for microcontroller deployment directly on plasma etchers and PECVD tool sensors.", body_style),
        Paragraph("• <b>Spatial Wafer Map Vision:</b> Incorporate 2D Convolutional Neural Networks (CNNs) for full wafer spatial clustering (WM-811K benchmark style).", body_style),
        Paragraph("• <b>Experimental CeNSE Validation:</b> Calibrate simulation parameters against physical silicon characterization runs from CeNSE NNFC/MNCF.", body_style)
    ]

    layout_p6 = Table([[p6_col1, p6_col2]], colWidths=[360, 360])
    layout_p6.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#f8f9fa")),
        ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#e7f5ff")),
        ("BOX", (0, 0), (0, 0), 1, colors.HexColor("#dee2e6")),
        ("BOX", (1, 0), (1, 0), 1, colors.HexColor("#a5d8ff")),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(layout_p6)

    # Build PDF
    doc.build(story)
    print(f"--> Saved 6-page presentation PDF: {output_path}")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    create_pptx("outputs/presentation.pptx")
    create_pdf("outputs/presentation.pdf")
    print("All presentation artifacts successfully created in outputs/")
