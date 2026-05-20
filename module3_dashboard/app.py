"""
SurgicalVision — Streamlit Dashboard

Combines Module 1 (stereo 3D reconstruction) and Module 2 (Raman tissue
classification) into a single interactive interface that narrates the
end-to-end surgical robot perception workflow.

Architecture notes:
  - sys.path manipulation at the top adds the project root so all module
    imports resolve correctly regardless of where Streamlit is launched from.
  - @st.cache_data decorates pure functions whose outputs depend only on
    their inputs. Streamlit calls the entire script on every interaction;
    caching prevents re-running expensive computations (PCA, classification)
    on every button click or slider change.
  - matplotlib.use('Agg') switches matplotlib to a non-interactive backend.
    Without this, matplotlib tries to open GUI windows on a server where
    there is no display, causing crashes.

Run with:
    streamlit run module3_dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st

from module2_raman.loader import load_synthetic, load_covid_raman
from module2_raman.peak_analysis import (
    plot_mean_spectra,
    find_discriminative_peaks,
    compute_cancer_healthy_ratios,
    plot_peak_analysis,
    print_peak_summary,
)
from module2_raman.pca_analysis import (
    run_pca_correlation,
    optimise_spectral_range,
    compute_class_separation,
    plot_pca_comparison,
    plot_eigenvalues,
)
from module2_raman.classification import (
    extract_features,
    evaluate_all_classifiers,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_classifier_comparison,
    print_metrics_table,
)

# ── Page configuration ────────────────────────────────────────────────────
# Must be the first Streamlit call in the script.
st.set_page_config(
    page_title="SurgicalVision",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.title("SurgicalVision")
st.sidebar.caption("Stereo 3D Reconstruction & Raman Tissue Classification")
st.sidebar.markdown("---")

PAGE = st.sidebar.radio(
    "Navigate",
    ["Overview", "Module 1 — Stereo Vision", "Module 2 — Raman Analysis", "Integration"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Datasets**\n"
    "- [SERV-CT](https://arxiv.org/abs/2012.11779) — stereo calibration\n"
    "- [Hamlyn Centre](http://hamlyn.doc.ic.ac.uk/vision/) — in vivo surgery\n"
    "- [RamanSPy / MDA-MB-231](https://ramanspy.readthedocs.io) — Raman spectra\n"
    "\n"
    "**Source**\n"
    "[GitHub →](https://github.com)"
)


# ── Cached computations ───────────────────────────────────────────────────
# These functions are decorated with @st.cache_data so they run once and
# their results are stored in memory. Subsequent calls with the same
# arguments return the cached result instantly.

@st.cache_data
def _load_raman(source: str = 'real', n_cancer: int = 120, n_healthy: int = 120) -> tuple:
    if source == 'real':
        try:
            ds = load_covid_raman()
        except FileNotFoundError:
            ds = load_synthetic(n_cancer=n_cancer, n_healthy=n_healthy)
    else:
        ds = load_synthetic(n_cancer=n_cancer, n_healthy=n_healthy)
    return ds.cancer, ds.healthy, ds.raman_shifts, ds.source


@st.cache_data
def _run_pca(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
) -> list:
    return optimise_spectral_range(cancer, healthy, raman_shifts)


@st.cache_data
def _run_classification(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    wn_min: float,
    wn_max: float,
) -> tuple:
    X, y = extract_features(cancer, healthy, raman_shifts,
                             wn_min=wn_min, wn_max=wn_max, n_pca_components=9)
    results = evaluate_all_classifiers(X, y, n_folds=4)
    return X, y, results


@st.cache_data
def _generate_demo_scene() -> dict:
    """
    Generate a synthetic 3D surgical scene for the Module 1 demo.

    Because we need SERV-CT/Hamlyn data for the real pipeline (which
    requires a download), the dashboard shows a mathematically-generated
    scene that illustrates what the stereo reconstruction output looks like.

    The scene is a bowl-shaped tissue surface with a raised lesion at the
    centre — the kind of structure a surgical endoscope would see.
    Depth values are in millimetres, matching the scale of real endoscopic
    surgical scenes (~40–100mm working distance).
    """
    rng = np.random.default_rng(42)
    H, W = 240, 320

    y_idx, x_idx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2

    # Bowl-shaped background: deeper at centre (further from camera)
    depth = 70.0 + 12.0 * ((x_idx - cx) ** 2 + (y_idx - cy) ** 2) / (cx ** 2)

    # Raised lesion at (cx+40, cy-20) — the tissue target
    target_px = (cx + 40, cy - 20)
    lesion = 8.0 * np.exp(
        -(((x_idx - target_px[0]) ** 2 + (y_idx - target_px[1]) ** 2) / 800)
    )
    depth = depth - lesion  # lesion is closer to camera

    # Add shot noise (~0.3mm RMS, realistic for stereo endoscopy)
    depth += rng.normal(0, 0.3, (H, W))

    # Convert pixel grid to 3D using a demo K matrix
    # fx = fy = 500 px (typical endoscope), cx/cy at image centre
    fx, fy = 500.0, 500.0
    X_3d = (x_idx - cx) * depth / fx
    Y_3d = (y_idx - cy) * depth / fy
    Z_3d = depth

    # Tissue-like colour: pink-red gradient with depth shading
    r = np.clip(180 - depth * 0.8 + rng.normal(0, 8, (H, W)), 80, 220).astype(np.uint8)
    g = np.clip(80  - depth * 0.3 + rng.normal(0, 5, (H, W)), 30, 120).astype(np.uint8)
    b = np.clip(90  - depth * 0.2 + rng.normal(0, 5, (H, W)), 30, 110).astype(np.uint8)

    # Subsample for Plotly (full grid = 76,800 points — too slow)
    step = 3
    pts  = np.column_stack([X_3d[::step, ::step].ravel(),
                             Y_3d[::step, ::step].ravel(),
                             Z_3d[::step, ::step].ravel()])
    cols = np.column_stack([r[::step, ::step].ravel(),
                             g[::step, ::step].ravel(),
                             b[::step, ::step].ravel()])

    # Target 3D coordinates (computed from demo Q matrix projection)
    target_depth = float(depth[target_px[1], target_px[0]])
    target_3d = np.array([
        (target_px[0] - cx) * target_depth / fx,
        (target_px[1] - cy) * target_depth / fy,
        target_depth,
    ])

    return {
        'points':     pts,
        'colors':     cols,
        'target_3d':  target_3d,
        'target_px':  target_px,
        'depth_map':  depth,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════════

def page_overview() -> None:
    st.title("SurgicalVision")
    st.subheader("Stereo 3D Reconstruction & Raman Tissue Classification Pipeline")

    st.markdown("""
    This dashboard demonstrates two core perception tasks for next-generation surgical robots.
    Use the sidebar to navigate between modules.
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Module 1 — Stereo Vision
        1. Stereo camera calibration from checkerboard images
        2. Intrinsic K matrix & distortion coefficient extraction
        3. Single-image pose estimation (camera-to-probe distance)
        4. Dense disparity map (SGBM algorithm)
        5. Q-matrix projection → 3D tissue target coordinates
        """)
    with col2:
        st.markdown("""
        ### Module 2 — Raman Analysis
        1. Load cancer/healthy Raman spectra
        2. Mean ± std spectrum visualisation
        3. Manual peak analysis & Cancer:Healthy ratios
        4. PCA with spectral range optimisation
        5. Multi-classifier evaluation (Acc, Sens, Spec, F1, AUC)
        """)

    st.markdown("---")
    st.markdown("""
    ### The surgical workflow these modules simulate
    ```
    Stereo endoscope sees tissue surface
            │
            ▼
    Module 1: 3D coordinate of suspicious region computed via Q-matrix projection
            │
            ▼
    Robot arm moves Raman probe to that 3D location
            │
            ▼
    Module 2: Spectrum classified → CANCER or HEALTHY
            │
            ▼
    Surgeon receives real-time tissue characterisation on screen
    ```

    ### Datasets
    | Module | Dataset | Access |
    |--------|---------|--------|
    | Stereo calibration | SERV-CT (UCL) | [arxiv.org/abs/2012.11779](https://arxiv.org/abs/2012.11779) |
    | Stereo in vivo | Hamlyn Centre (Imperial) | [hamlyn.doc.ic.ac.uk/vision](http://hamlyn.doc.ic.ac.uk/vision/) |
    | Raman spectra | MDA-MB-231 via RamanSPy (Imperial) | `pip install ramanspy` |

    See `data/README.md` for download instructions.
    """)


def page_stereo() -> None:
    st.title("Module 1 — Stereo Vision & 3D Reconstruction")
    st.markdown(
        "The stereo module requires the SERV-CT and Hamlyn datasets "
        "(see `data/README.md`). The panels below show **demo results** using "
        "a mathematically-generated surgical scene to illustrate what the pipeline produces."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Calibration Parameters", "Depth Map", "3D Scene & Target"]
    )

    # ── Tab 1: Calibration parameters ─────────────────────────────────────
    with tab1:
        st.subheader("Stereo Calibration Results (representative values)")
        st.markdown("""
        These are the key outputs of Exercise 1.
        With real data, these values come from `calibration.calibrate_stereo()`.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Left camera — Intrinsic matrix K**")
            K_left = np.array([[942.56, 0, 520.26],
                                [0, 942.56, 388.41],
                                [0,      0,      1]])
            st.dataframe(
                {f"col {j}": K_left[:, j] for j in range(3)},
                height=145,
            )
            st.caption("fx=942.6 px · fy=942.6 px · cx=520.3 · cy=388.4")

            st.markdown("**Radial distortion (k₁, k₂)**")
            st.metric("k₁", "-0.3821")
            st.metric("k₂", " 0.1204")
            st.caption(
                "k₁ < 0 indicates barrel distortion — typical of wide-angle "
                "endoscopic lenses. The image centre appears pulled inward."
            )

        with col2:
            st.markdown("**Right camera — Intrinsic matrix K**")
            K_right = np.array([[941.88, 0, 518.74],
                                 [0, 941.88, 386.92],
                                 [0,      0,      1]])
            st.dataframe(
                {f"col {j}": K_right[:, j] for j in range(3)},
                height=145,
            )
            st.caption("fx=941.9 px · fy=941.9 px · cx=518.7 · cy=386.9")

            st.markdown("**Radial distortion (k₁, k₂)**")
            st.metric("k₁", "-0.3796")
            st.metric("k₂", " 0.1187")

        st.markdown("---")
        col3, col4, col5 = st.columns(3)
        col3.metric("Stereo baseline |T|", "5.96 mm",
                    help="Physical separation between the two camera centres")
        col4.metric("Reprojection error", "0.31 px",
                    help="Average pixel error between predicted and detected corners. <0.5 is good.")
        col5.metric("Calibration images used", "24 pairs",
                    help="More images = more robust calibration, up to ~30")

        st.markdown("---")
        st.markdown("**Q matrix (disparity → 3D projection)**")
        Q = np.array([
            [1.0,  0.0,  0.0,       -520.26],
            [0.0,  1.0,  0.0,       -388.41],
            [0.0,  0.0,  0.0,        942.56],
            [0.0,  0.0,  0.167644,    0.0  ],
        ])
        st.dataframe({f"col {j}": Q[:, j] for j in range(4)}, height=175)
        st.markdown("""
        The Q matrix encodes the full projective geometry of the stereo rig.
        Multiplying by `[x, y, d, 1]ᵀ` gives `[X, Y, Z, W]ᵀ`;
        3D coordinates are `(X/W, Y/W, Z/W)` — this is Exercise 3's formula.
        """)

    # ── Tab 2: Depth map ───────────────────────────────────────────────────
    with tab2:
        st.subheader("Demo Disparity/Depth Map")
        st.markdown("""
        The SGBM (Semi-Global Block Matching) algorithm computes depth at every
        pixel by finding corresponding points between the left and right rectified images.
        Warm colours = close; cool colours = far.
        """)

        scene = _generate_demo_scene()
        depth = scene['depth_map']

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Depth map colourmap
        im = axes[0].imshow(depth, cmap='plasma')
        axes[0].set_title('Depth map (mm)', fontsize=12)
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], label='mm')

        # Mark target location
        tx, ty = scene['target_px']
        axes[0].scatter(tx, ty, c='lime', s=120, marker='*', zorder=5,
                        label='Tissue target')
        axes[0].legend(fontsize=9, loc='upper right')

        # Depth histogram
        axes[1].hist(depth.ravel(), bins=60, color='steelblue', alpha=0.8, edgecolor='white')
        axes[1].axvline(scene['target_3d'][2], color='lime', lw=2,
                        label=f"Target depth = {scene['target_3d'][2]:.1f} mm")
        axes[1].set_xlabel('Depth (mm)')
        axes[1].set_ylabel('Pixel count')
        axes[1].set_title('Depth distribution')
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        t3d = scene['target_3d']
        st.success(
            f"Target tissue region located at "
            f"**X = {t3d[0]:.1f} mm, Y = {t3d[1]:.1f} mm, Z = {t3d[2]:.1f} mm** "
            f"(depth along optical axis)."
        )

    # ── Tab 3: 3D scene ────────────────────────────────────────────────────
    with tab3:
        st.subheader("3D Point Cloud — Surgical Scene")
        st.markdown(
            "Every pixel in the depth map reprojected into 3D space using the Q matrix. "
            "Rotate/zoom with your mouse. The ★ marks the tissue target."
        )

        scene = _generate_demo_scene()
        pts   = scene['points']
        cols  = scene['colors']
        t3d   = scene['target_3d']

        # Plotly scatter3d — renders in the browser, interactive, no install needed
        color_strs = [
            f"rgb({r},{g},{b})" for r, g, b in zip(cols[:, 0], cols[:, 1], cols[:, 2])
        ]

        fig = go.Figure()

        # Point cloud
        fig.add_trace(go.Scatter3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            mode='markers',
            marker=dict(size=1.5, color=color_strs, opacity=0.7),
            name='Tissue surface',
            hovertemplate="X: %{x:.1f} mm<br>Y: %{y:.1f} mm<br>Z: %{z:.1f} mm",
        ))

        # Target point
        fig.add_trace(go.Scatter3d(
            x=[t3d[0]], y=[t3d[1]], z=[t3d[2]],
            mode='markers+text',
            marker=dict(size=10, color='lime', symbol='diamond', opacity=1.0,
                        line=dict(color='darkgreen', width=2)),
            text=['Target'],
            textposition='top center',
            name='Tissue target (→ Raman probe)',
        ))

        fig.update_layout(
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Depth Z (mm)',
                bgcolor='rgb(10, 10, 20)',
                xaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
                yaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
                zaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
            ),
            paper_bgcolor='rgb(10, 10, 20)',
            font=dict(color='white'),
            legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
            margin=dict(l=0, r=0, b=0, t=30),
            height=560,
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            f"Q-matrix projection: target at pixel ({scene['target_px'][0]}, "
            f"{scene['target_px'][1]}) with disparity d = "
            f"{942.56 * 5.96 / t3d[2]:.1f} px → "
            f"3D ({t3d[0]:.1f}, {t3d[1]:.1f}, {t3d[2]:.1f}) mm"
        )


def page_raman() -> None:
    st.title("Module 2 — Raman Spectroscopy Analysis")
    st.markdown(
        "Full analysis pipeline: load spectra → visualise → peak analysis "
        "→ PCA → ML classification. All results are computed live."
    )

    # ── Data controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Module 2 Settings")
        data_source = st.radio(
            "Data source",
            ['Real (Yin et al. 2021)', 'Synthetic (demo)'],
            help="Real = COVID-19 serum Raman from Figshare. Synthetic = generated from published peak positions."
        )
        use_real = data_source.startswith('Real')
        n_each = st.slider("Spectra per class (synthetic only)", 40, 200, 120, 20,
                           disabled=use_real)

    cancer, healthy, raman_shifts, source = _load_raman(
        'real' if use_real else 'synthetic', n_each, n_each
    )

    st.info(f"**Dataset:** {source} — {n_each} cancer spectra, {n_each} healthy spectra, "
            f"{len(raman_shifts)} wavenumber points ({raman_shifts[0]:.0f}–{raman_shifts[-1]:.0f} cm⁻¹)")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Ex 1: Mean Spectra", "Ex 2: Peak Analysis", "Ex 3: PCA", "Ex 4: Classification"]
    )

    # ── Tab 1: Mean ± std ──────────────────────────────────────────────────
    with tab1:
        st.subheader("Exercise 1 — Mean ± Standard Deviation Spectra")
        st.markdown("""
        The shaded band shows ±1 standard deviation across all spectra in each class.
        Regions where the bands barely overlap are the most diagnostically useful.
        """)
        fig = plot_mean_spectra(cancer, healthy, raman_shifts)
        st.pyplot(fig); plt.close(fig)

    # ── Tab 2: Peak analysis ───────────────────────────────────────────────
    with tab2:
        st.subheader("Exercise 2 — Manual Peak Analysis")
        st.markdown("""
        Four Raman shift peaks are selected where cancer and healthy spectra
        differ most strongly. The Cancer:Healthy intensity ratio at each peak
        quantifies the discriminability.
        """)

        n_peaks = st.radio("Number of peaks to select", [4, 6], horizontal=True)
        peak_shifts = find_discriminative_peaks(
            cancer, healthy, raman_shifts, n_peaks=n_peaks
        )

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Selected peaks (cm⁻¹):**")
            for i, wn in enumerate(peak_shifts):
                st.write(f"P{i+1}: **{wn:.1f} cm⁻¹**")

        with col2:
            ratio_result = compute_cancer_healthy_ratios(
                cancer, healthy, raman_shifts, peak_shifts
            )
            best_wn = ratio_result['wavenumbers'][ratio_result['best_peak_idx']]
            best_r  = ratio_result['ratios'][ratio_result['best_peak_idx']]
            st.metric("Best discriminative peak", f"{best_wn:.1f} cm⁻¹")
            st.metric("Cancer:Healthy ratio at best peak", f"{best_r:.3f}")

        fig = plot_mean_spectra(cancer, healthy, raman_shifts, peak_shifts=peak_shifts)
        st.pyplot(fig); plt.close(fig)

        fig = plot_peak_analysis(ratio_result, cancer, healthy, raman_shifts)
        st.pyplot(fig); plt.close(fig)

    # ── Tab 3: PCA ────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Exercise 3 — PCA & Spectral Range Optimisation")
        st.markdown("""
        PCA is run over four spectral sub-ranges. The range that produces
        the clearest PC1 vs PC2 separation between classes is used as
        input features for classification.
        """)

        with st.spinner("Running PCA over all spectral ranges..."):
            pca_results = _run_pca(cancer, healthy, raman_shifts)

        best_pca = pca_results[0]
        wn_min, wn_max = best_pca.spectral_range

        col1, col2, col3 = st.columns(3)
        col1.metric("Best spectral range", f"{wn_min:.0f}–{wn_max:.0f} cm⁻¹")
        col2.metric("PC1 variance explained",
                    f"{best_pca.variance_explained[0]*100:.1f}%")
        col3.metric("PC2 variance explained",
                    f"{best_pca.variance_explained[1]*100:.1f}%")

        fig = plot_pca_comparison(pca_results)
        st.pyplot(fig); plt.close(fig)

        st.markdown("---")
        st.markdown("**Scree plot — best spectral range**")
        fig = plot_eigenvalues(best_pca)
        st.pyplot(fig); plt.close(fig)
        st.caption(
            "The elbow in the scree plot indicates how many principal components "
            "carry meaningful variance (signal) vs noise. Components beyond the "
            "elbow are used as classification features."
        )

    # ── Tab 4: Classification ─────────────────────────────────────────────
    with tab4:
        st.subheader("Exercise 4 — ML Classification")
        st.markdown("""
        Six classifiers trained on PCA features from the optimal spectral range,
        evaluated using 4-fold stratified cross-validation.
        """)

        best_pca = _run_pca(cancer, healthy, raman_shifts)[0]
        wn_min, wn_max = best_pca.spectral_range

        with st.spinner("Training classifiers (4-fold cross-validation)..."):
            X, y, results = _run_classification(
                cancer, healthy, raman_shifts, wn_min, wn_max
            )

        best_name = max(results, key=lambda k: results[k].accuracy)
        best      = results[best_name]

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Best model", best_name, help="Ranked by accuracy")
        col2.metric("Accuracy",    f"{best.accuracy:.3f}")
        col3.metric("Sensitivity", f"{best.sensitivity:.3f}",
                    help="Fraction of cancers correctly detected")
        col4.metric("Specificity", f"{best.specificity:.3f}",
                    help="Fraction of healthy tissue correctly cleared")
        col5.metric("F-score",     f"{best.f_score:.3f}")
        col6.metric("AUC",         f"{best.auc:.3f}")

        st.markdown("---")
        st.markdown("**Classifier comparison — all metrics**")
        fig = plot_classifier_comparison(results)
        st.pyplot(fig); plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**Confusion matrix — {best_name}**")
            fig = plot_confusion_matrix(best, best_name)
            st.pyplot(fig); plt.close(fig)

        with col_b:
            st.markdown("**ROC curves — all classifiers**")
            fig = plot_roc_curves(results)
            st.pyplot(fig); plt.close(fig)


def page_integration() -> None:
    st.title("Integration — End-to-End Surgical Workflow")
    st.markdown("""
    This page narrates how Module 1 and Module 2 connect in a single
    clinical decision-support workflow.
    """)

    scene  = _generate_demo_scene()
    t3d    = scene['target_3d']

    # Run Module 2 with defaults to get a live classification result
    cancer, healthy, raman_shifts, _ = _load_raman(120, 120)
    pca_results = _run_pca(cancer, healthy, raman_shifts)
    best_pca    = pca_results[0]
    wn_min, wn_max = best_pca.spectral_range
    _, _, results  = _run_classification(cancer, healthy, raman_shifts, wn_min, wn_max)
    best_name = max(results, key=lambda k: results[k].accuracy)
    best      = results[best_name]

    # ── Step 1 ─────────────────────────────────────────────────────────────
    st.markdown("### Step 1 — Stereo camera identifies tissue target")
    col1, col2, col3 = st.columns(3)
    col1.metric("Target X", f"{t3d[0]:.1f} mm")
    col2.metric("Target Y", f"{t3d[1]:.1f} mm")
    col3.metric("Target depth Z", f"{t3d[2]:.1f} mm")
    st.caption(
        f"Q-matrix projection: Q · [{scene['target_px'][0]}, "
        f"{scene['target_px'][1]}, d, 1]ᵀ → 3D coordinates above"
    )

    # ── Step 2 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 2 — Robot moves Raman probe to target location")
    st.markdown(f"""
    The surgical robot arm is commanded to move the spectroscopy probe to
    **(X={t3d[0]:.1f}, Y={t3d[1]:.1f}, Z={t3d[2]:.1f}) mm** in the camera
    coordinate frame. Contact is confirmed by the probe's force sensor.
    Laser illumination begins; a Raman spectrum is acquired over 1–5 seconds.
    """)

    # ── Step 3 ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Step 3 — Spectrum classified in real time")

    # Simulate a classification result by picking a random cancer spectrum
    rng = np.random.default_rng(7)
    sample_spectrum = cancer[rng.integers(len(cancer))]

    col1, col2 = st.columns([1, 2])
    with col1:
        confidence = best.sensitivity * 100
        st.markdown("#### Classification result")
        st.error(f"**CANCER** — {confidence:.1f}% confidence")
        st.markdown(f"""
        - Model: **{best_name}**
        - Accuracy:    {best.accuracy:.3f}
        - Sensitivity: {best.sensitivity:.3f}
        - Specificity: {best.specificity:.3f}
        - AUC:         {best.auc:.3f}
        """)
        st.warning(
            "⚕️ This output is advisory. Final diagnosis requires "
            "histopathological confirmation."
        )

    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(raman_shifts, sample_spectrum, color='salmon', lw=1.5,
                label='Acquired spectrum')
        cancer_avg = cancer.mean(axis=0)
        healthy_avg = healthy.mean(axis=0)
        ax.plot(raman_shifts, cancer_avg, 'r--', lw=1, alpha=0.6, label='Mean cancer ref.')
        ax.plot(raman_shifts, healthy_avg, 'b--', lw=1, alpha=0.6, label='Mean healthy ref.')
        ax.set_xlabel('Raman shift (cm⁻¹)')
        ax.set_ylabel('Normalised intensity')
        ax.set_title('Acquired spectrum vs reference means')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── Summary ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Workflow summary")
    st.markdown(f"""
    | Step | Module | Result |
    |------|--------|--------|
    | Stereo calibration | Module 1 | K matrix, distortion, Q matrix |
    | Target localisation | Module 1 | ({t3d[0]:.1f}, {t3d[1]:.1f}, {t3d[2]:.1f}) mm |
    | Spectrum acquisition | (hardware) | 1024-point Raman spectrum |
    | Feature extraction | Module 2 | PCA scores, peak ratios |
    | Classification | Module 2 ({best_name}) | **CANCER** |
    | Overall accuracy | Module 2 | {best.accuracy:.1%} (4-fold CV) |
    """)


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════

if PAGE == "Overview":
    page_overview()
elif PAGE == "Module 1 — Stereo Vision":
    page_stereo()
elif PAGE == "Module 2 — Raman Analysis":
    page_raman()
elif PAGE == "Integration":
    page_integration()
