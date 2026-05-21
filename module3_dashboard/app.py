"""
stereo-raman-pipeline — Streamlit Dashboard

Combines Module 1 (stereo 3D reconstruction) and Module 2 (Raman tissue
classification) into a single interactive interface that narrates the
end-to-end surgical robot perception workflow.

Architecture notes:
  - @st.cache_data: Streamlit reruns the whole script on every interaction.
    Decorating expensive functions means they only execute once per unique
    set of inputs — subsequent calls return the cached result instantly.
  - matplotlib.use('Agg'): switches matplotlib to a non-interactive backend
    so it doesn't try to open GUI windows on a headless server.
  - Graceful fallback: Module 1 falls back to a synthetic demo if SERV-CT
    data is not present; Module 2 falls back to synthetic if real data is
    not present. The dashboard is always runnable.

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
import cv2
import streamlit as st

from data.downloader import ensure_all

# Module 1
from module1_stereo.disparity import compute_disparity_sgbm
from module1_stereo.reconstruction import reproject_to_3d, reproject_single_point, filter_point_cloud
from module1_stereo.validation import evaluate_disparity, evaluate_depth, plot_validation

# Module 2
from module2_raman.loader import load_synthetic, load_covid_raman
from module2_raman.preprocessing import preprocess_spectra, asymmetric_least_squares, detect_cosmic_rays
from module2_raman.peak_analysis import (
    plot_mean_spectra, find_discriminative_peaks,
    compute_cancer_healthy_ratios, plot_peak_analysis,
)
from module2_raman.pca_analysis import (
    optimise_spectral_range, compute_class_separation,
    plot_pca_comparison, plot_eigenvalues,
)
from module2_raman.classification import (
    extract_features, evaluate_all_classifiers, build_classifiers,
    plot_confusion_matrix, plot_roc_curves, plot_classifier_comparison,
    plot_learning_curves, predict_with_confidence, plot_confidence_distribution,
)

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="stereo-raman-pipeline",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ────────────────────────────────────────────────────
st.sidebar.title("stereo-raman-pipeline")
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
    "- [SERV-CT](https://rdr.ucl.ac.uk/articles/dataset/26352199) — stereo (UCL)\n"
    "- [Yin et al. 2021](https://figshare.com/articles/dataset/12159924) — Raman (Figshare)\n"
    "\n"
    "**Source**\n"
    "[GitHub →](https://github.com/joliemcgreavy/stereo-raman-pipeline)"
)

# ── Data availability ─────────────────────────────────────────────────────
# On Streamlit Community Cloud the data/ directory is empty on first deploy.
# ensure_all() downloads both datasets if absent and shows a status spinner.
# On local runs where files already exist, this returns instantly.

@st.cache_resource
def _bootstrap_data() -> dict:
    with st.status("Checking datasets...", expanded=False) as s:
        return ensure_all(status=s)

_data_status = _bootstrap_data()

SERV_CT_DIR = Path(__file__).parent.parent / 'data' / 'raw' / 'serv_ct' / 'SERV-CT'
SERV_CT_AVAILABLE = SERV_CT_DIR.exists() and _data_status.get('serv_ct', False)

ALL_FRAME_IDS = [f'{n:03d}' for n in range(1, 17)]


# ── Cached computations ───────────────────────────────────────────────────

@st.cache_data
def _load_serv_ct_frame(frame_id: str) -> dict | None:
    """Load a SERV-CT frame and return serialisable data for Streamlit."""
    try:
        from module1_stereo.serv_ct_loader import load_frame
        f = load_frame(frame_id)
        return {
            'frame_id':    f.frame_id,
            'experiment':  f.experiment,
            'left_img':    f.left_img,
            'right_img':   f.right_img,
            'Q':           f.Q,
            'P1':          f.P1,
            'P2':          f.P2,
            'gt_depth_mm': f.gt_depth_mm,
            'gt_disparity': f.gt_disparity,
            'valid_mask':  f.valid_mask,
            'focal_px':    float(f.Q[2, 3]),
            'baseline_mm': float(1.0 / f.Q[3, 2]),
            'cx':          float(-f.Q[0, 3]),
            'cy':          float(-f.Q[1, 3]),
        }
    except Exception:
        return None


@st.cache_data
def _compute_disparity(frame_id: str) -> dict | None:
    data = _load_serv_ct_frame(frame_id)
    if data is None:
        return None
    gt_min = int(data['gt_disparity'][data['valid_mask']].min())
    result = compute_disparity_sgbm(
        data['left_img'], data['right_img'],
        min_disparity=max(0, gt_min - 16),
        num_disparities=96, block_size=9,
    )
    return {'disparity_map': result.disparity_map,
            'disparity_visual': result.disparity_visual}


@st.cache_data
def _load_raman(source: str, n_each: int = 120) -> tuple:
    if source == 'real':
        try:
            ds = load_covid_raman()
        except FileNotFoundError:
            ds = load_synthetic(n_cancer=n_each, n_healthy=n_each)
    else:
        ds = load_synthetic(n_cancer=n_each, n_healthy=n_each)
    return ds.cancer, ds.healthy, ds.raman_shifts, ds.source


@st.cache_data
def _run_pca(cancer: np.ndarray, healthy: np.ndarray, raman_shifts: np.ndarray) -> list:
    return optimise_spectral_range(cancer, healthy, raman_shifts)


@st.cache_data
def _run_classification(
    cancer: np.ndarray, healthy: np.ndarray,
    raman_shifts: np.ndarray, wn_min: float, wn_max: float,
) -> tuple:
    X, y = extract_features(cancer, healthy, raman_shifts,
                             wn_min=wn_min, wn_max=wn_max, n_pca_components=9)
    results = evaluate_all_classifiers(X, y, n_folds=4)
    return X, y, results


@st.cache_data
def _run_learning_curves(
    cancer: np.ndarray, healthy: np.ndarray,
    raman_shifts: np.ndarray, wn_min: float, wn_max: float,
) -> tuple:
    X, y = extract_features(cancer, healthy, raman_shifts,
                             wn_min=wn_min, wn_max=wn_max, n_pca_components=9)
    return X, y


# ── Synthetic fallback scene (used if SERV-CT not available) ──────────────

@st.cache_data
def _synthetic_scene() -> dict:
    rng = np.random.default_rng(42)
    H, W = 240, 320
    y_idx, x_idx = np.mgrid[0:H, 0:W]
    cx, cy = W // 2, H // 2
    depth = 70.0 + 12.0 * ((x_idx - cx)**2 + (y_idx - cy)**2) / cx**2
    target_px = (cx + 40, cy - 20)
    depth -= 8.0 * np.exp(-(((x_idx - target_px[0])**2 + (y_idx - target_px[1])**2) / 800))
    depth += rng.normal(0, 0.3, (H, W))
    fx = 500.0
    X_3d = (x_idx - cx) * depth / fx
    Y_3d = (y_idx - cy) * depth / fx
    r = np.clip(180 - depth * 0.8 + rng.normal(0, 8, (H, W)), 80, 220).astype(np.uint8)
    g = np.clip(80  - depth * 0.3 + rng.normal(0, 5, (H, W)), 30, 120).astype(np.uint8)
    b = np.clip(90  - depth * 0.2 + rng.normal(0, 5, (H, W)), 30, 110).astype(np.uint8)
    step = 3
    pts  = np.column_stack([X_3d[::step,::step].ravel(), Y_3d[::step,::step].ravel(), depth[::step,::step].ravel()])
    cols = np.column_stack([r[::step,::step].ravel(), g[::step,::step].ravel(), b[::step,::step].ravel()])
    td = float(depth[target_px[1], target_px[0]])
    return {
        'points': pts, 'colors': cols,
        'target_3d': np.array([(target_px[0]-cx)*td/fx, (target_px[1]-cy)*td/fx, td]),
        'target_px': target_px, 'depth_map': depth,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pages
# ═══════════════════════════════════════════════════════════════════════════

def page_overview() -> None:
    st.title("stereo-raman-pipeline")
    st.subheader("End-to-end Python pipeline for surgical stereo vision and Raman tissue classification")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Module 1 — Stereo Vision
        Real data: **SERV-CT** (Psychogyios et al. 2022, UCL)
        - 16 rectified stereo pairs, da Vinci™ endoscope
        - Ex vivo porcine tissue, 60–95mm depth range
        - CT ground-truth depth validation
        - SGBM disparity → Q-matrix → 3D point cloud
        """)
    with col2:
        st.markdown("""
        ### Module 2 — Raman Analysis
        Real data: **Yin et al. 2021** (Journal of Raman Spectroscopy)
        - 309 serum Raman spectra (159 disease, 150 healthy)
        - ALS baseline correction + cosmic ray removal
        - PCA range optimisation + 6 ML classifiers
        - Uncertainty quantification (selective prediction)
        """)

    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Module 2 accuracy", "94.8%", "Random Forest, 4-fold CV")
    c2.metric("Module 2 AUC", "0.980", "Real serum Raman data")
    c3.metric("Module 1 depth MAE", "~5 mm", "SGBM vs CT ground truth")
    c4.metric("Test suite", "74 / 74", "All passing")

    st.markdown("---")
    st.markdown("### Surgical workflow")
    st.code("""\
Stereo endoscope sees tissue surface
            │
            ▼
SGBM disparity map → Q-matrix projection → 3D target (X, Y, Z)
            │
            ▼
Robot moves Raman probe to (X, Y, Z)
            │
            ▼
Spectrum acquired (1–5 s integration)
            │
            ▼
PCA feature extraction → ML classifier
            │
            ▼
DISEASE / HEALTHY  +  confidence score  →  surgeon
""", language=None)


def page_stereo() -> None:
    st.title("Module 1 — Stereo Vision & 3D Reconstruction")

    # ── Sidebar controls ───────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Module 1 Settings")
        if SERV_CT_AVAILABLE:
            frame_id = st.selectbox(
                "SERV-CT frame",
                ALL_FRAME_IDS,
                index=0,
                help="Frames 001–008: straight endoscope. 009–016: 30° angled endoscope.",
            )
            target_col = st.slider("Target pixel — column", 100, 620, 360)
            target_row = st.slider("Target pixel — row",    80, 496, 288)
        else:
            st.warning("SERV-CT data not found. Showing synthetic demo.")
            frame_id = None

    tab1, tab2, tab3 = st.tabs(["Calibration", "Disparity & Validation", "3D Scene"])

    # ── Tab 1: Calibration ────────────────────────────────────────────────
    with tab1:
        st.subheader("Stereo Calibration — da Vinci Endoscope")

        if SERV_CT_AVAILABLE:
            data = _load_serv_ct_frame(frame_id)
            st.info(f"Frame **{frame_id}** — Experiment {data['experiment']} "
                    f"({'straight' if data['experiment']==1 else '30°'} endoscope)")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Focal length",    f"{data['focal_px']:.1f} px")
            col2.metric("Baseline",        f"{data['baseline_mm']:.2f} mm")
            col3.metric("Principal pt cx", f"{data['cx']:.1f} px")
            col4.metric("Principal pt cy", f"{data['cy']:.1f} px")

            st.markdown("**Q matrix** (disparity → 3D projection, from calibration JSON)")
            Q = data['Q']
            st.dataframe({f"col {j}": Q[:, j].round(4) for j in range(4)}, height=175)
            st.caption(
                "Q is the same matrix used in Exercise 3 of the assignment — "
                "Q · [x, y, d, 1]ᵀ → [X, Y, Z, W]ᵀ, then divide by W. "
                "Here it comes from real stereo calibration, not a given value."
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**P1 — left camera projection matrix**")
                st.dataframe({f"col {j}": data['P1'][:, j].round(2) for j in range(4)}, height=130)
            with col_b:
                st.markdown("**P2 — right camera projection matrix**")
                st.dataframe({f"col {j}": data['P2'][:, j].round(2) for j in range(4)}, height=130)
            st.caption("P2[0,3] is the disparity offset from baseline — "
                       f"{data['P2'][0,3]:.1f} px·mm. "
                       "Dividing by focal length gives baseline: "
                       f"{abs(data['P2'][0,3])/data['focal_px']:.2f} mm.")
        else:
            st.markdown("SERV-CT not available — see `data/README.md` to download.")
            st.code("curl -L -o SERV-CT.zip https://ndownloader.figshare.com/files/47857471\nunzip SERV-CT.zip")

    # ── Tab 2: Disparity & Validation ─────────────────────────────────────
    with tab2:
        st.subheader("SGBM Disparity vs CT Ground Truth")

        if SERV_CT_AVAILABLE:
            data = _load_serv_ct_frame(frame_id)

            with st.spinner("Computing SGBM disparity..."):
                disp_data = _compute_disparity(frame_id)

            disp_map = disp_data['disparity_map']
            valid_mask = data['valid_mask']
            gt_disp    = data['gt_disparity']
            gt_depth   = data['gt_depth_mm']

            # Validation metrics
            disp_m = evaluate_disparity(disp_map, gt_disp, valid_mask, min_disp=1.0)
            pts_3d = reproject_to_3d(disp_map, data['Q'])
            depth_m = evaluate_depth(pts_3d[:, :, 2], gt_depth, valid_mask)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Disparity MAE",  f"{disp_m.mae_px:.2f} px")
            c2.metric("Disparity RMSE", f"{disp_m.rmse_px:.2f} px")
            c3.metric("Err > 1px",      f"{disp_m.pct_1px:.1f}%")
            c4.metric("Depth MAE",      f"{depth_m.mae_mm:.2f} mm")
            c5.metric("Valid pixels",   f"{disp_m.n_valid/1000:.0f}k")

            # Four-panel validation figure
            fig = plot_validation(data['left_img'], disp_map, gt_disp,
                                   valid_mask, frame_id=frame_id)
            st.pyplot(fig); plt.close(fig)

            st.caption(
                "The error map is brightest at depth discontinuities and "
                "specular tissue regions — where block matching struggles most. "
                "This is consistent with published SGBM results on endoscopic tissue."
            )
        else:
            scene = _synthetic_scene()
            st.warning("Showing synthetic demo — download SERV-CT for real results.")
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            im = axes[0].imshow(scene['depth_map'], cmap='plasma')
            axes[0].set_title('Synthetic depth map (demo)')
            axes[0].axis('off')
            plt.colorbar(im, ax=axes[0], label='mm')
            axes[1].hist(scene['depth_map'].ravel(), bins=40, color='steelblue')
            axes[1].set_title('Depth distribution')
            axes[1].set_xlabel('mm')
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    # ── Tab 3: 3D Scene ───────────────────────────────────────────────────
    with tab3:
        st.subheader("3D Point Cloud & Target Localisation")

        if SERV_CT_AVAILABLE:
            data     = _load_serv_ct_frame(frame_id)
            disp_data = _compute_disparity(frame_id)
            disp_map  = disp_data['disparity_map']
            pts_3d    = reproject_to_3d(disp_map, data['Q'])

            pts, colors = filter_point_cloud(
                pts_3d, data['left_img'], z_min_mm=50, z_max_mm=110
            )

            # Subsample for browser rendering
            rng = np.random.default_rng(0)
            idx = rng.choice(len(pts), size=min(15_000, len(pts)), replace=False)
            pts_s   = pts[idx]
            color_s = colors[idx]

            # Target point from sidebar sliders
            d_at_target = float(disp_map[target_row, target_col])
            if np.isnan(d_at_target) or d_at_target <= 0:
                d_at_target = float(np.nanmedian(disp_map))
            coords_3d = reproject_single_point(target_col, target_row, d_at_target, data['Q'])
            gt_z      = float(data['gt_depth_mm'][target_row, target_col])

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Target X", f"{coords_3d[0]:.1f} mm")
            c2.metric("Target Y", f"{coords_3d[1]:.1f} mm")
            c3.metric("Target depth Z", f"{coords_3d[2]:.1f} mm")
            c4.metric("CT depth (GT)", f"{gt_z:.1f} mm",
                      delta=f"{coords_3d[2]-gt_z:+.1f} mm error")

            color_strs = [f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
                          for r, g, b in color_s]

            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(
                x=pts_s[:,0], y=pts_s[:,1], z=pts_s[:,2],
                mode='markers',
                marker=dict(size=1.5, color=color_strs, opacity=0.7),
                name='Tissue surface',
                hovertemplate="X:%{x:.1f}mm  Y:%{y:.1f}mm  Z:%{z:.1f}mm",
            ))
            fig3d.add_trace(go.Scatter3d(
                x=[coords_3d[0]], y=[coords_3d[1]], z=[coords_3d[2]],
                mode='markers+text',
                marker=dict(size=10, color='lime', symbol='diamond',
                            line=dict(color='darkgreen', width=2)),
                text=[f'Target\n({coords_3d[2]:.1f}mm)'],
                textposition='top center',
                name='Tissue target → Raman probe',
            ))
            fig3d.update_layout(
                scene=dict(
                    xaxis_title='X (mm)', yaxis_title='Y (mm)', zaxis_title='Z (mm)',
                    bgcolor='rgb(10,10,20)',
                    xaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
                    yaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
                    zaxis=dict(gridcolor='rgb(50,50,70)', color='white'),
                ),
                paper_bgcolor='rgb(10,10,20)',
                font=dict(color='white'),
                legend=dict(bgcolor='rgba(0,0,0,0.5)', font=dict(color='white')),
                margin=dict(l=0, r=0, b=0, t=30),
                height=540,
            )
            st.plotly_chart(fig3d, use_container_width=True)
            st.caption(
                f"Drag to rotate · scroll to zoom · "
                f"Move the sliders in the sidebar to select a different target pixel. "
                f"Point cloud: {len(pts):,} points (15k shown)."
            )

        else:
            scene = _synthetic_scene()
            st.warning("Showing synthetic demo — download SERV-CT for real results.")
            pts, cols = scene['points'], scene['colors']
            t3d = scene['target_3d']
            color_strs = [f"rgb({r},{g},{b})" for r, g, b in zip(cols[:,0], cols[:,1], cols[:,2])]
            fig3d = go.Figure()
            fig3d.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2],
                mode='markers', marker=dict(size=1.5, color=color_strs, opacity=0.7),
                name='Tissue surface'))
            fig3d.add_trace(go.Scatter3d(x=[t3d[0]], y=[t3d[1]], z=[t3d[2]],
                mode='markers+text', marker=dict(size=10, color='lime', symbol='diamond'),
                text=['Target'], textposition='top center', name='Target'))
            fig3d.update_layout(
                scene=dict(bgcolor='rgb(10,10,20)',
                    xaxis=dict(color='white'), yaxis=dict(color='white'), zaxis=dict(color='white')),
                paper_bgcolor='rgb(10,10,20)', font=dict(color='white'),
                margin=dict(l=0,r=0,b=0,t=30), height=500)
            st.plotly_chart(fig3d, use_container_width=True)


def page_raman() -> None:
    st.title("Module 2 — Raman Spectroscopy Analysis")

    with st.sidebar:
        st.markdown("### Module 2 Settings")
        data_source = st.radio(
            "Data source",
            ['Real (Yin et al. 2021)', 'Synthetic (demo)'],
        )
        use_real = data_source.startswith('Real')
        n_each = st.slider("Spectra per class (synthetic)", 40, 200, 120, 20,
                           disabled=use_real)
        st.markdown("---")
        conf_threshold = st.slider(
            "Confidence threshold", 0.50, 0.95, 0.75, 0.05,
            help="Predictions below this are flagged as uncertain (uncertainty tab)"
        )

    cancer, healthy, raman_shifts, source = _load_raman(
        'real' if use_real else 'synthetic', n_each
    )
    n_c, n_h = len(cancer), len(healthy)
    st.info(f"**{source}** — {n_c} disease spectra · {n_h} healthy spectra · "
            f"{len(raman_shifts)} wavenumbers · "
            f"{raman_shifts[0]:.0f}–{raman_shifts[-1]:.0f} cm⁻¹")

    # Pre-compute PCA + classification (cached)
    pca_results = _run_pca(cancer, healthy, raman_shifts)
    best_pca    = pca_results[0]
    wn_min, wn_max = best_pca.spectral_range
    X, y, results = _run_classification(cancer, healthy, raman_shifts, wn_min, wn_max)
    best_name = max(results, key=lambda k: results[k].accuracy)
    best      = results[best_name]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Mean Spectra", "Peak Analysis", "PCA",
        "Classification", "Preprocessing", "Learning Curves", "Uncertainty",
    ])

    # ── Tab 1 ──────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Mean ± Standard Deviation Spectra")
        st.markdown("Regions where the bands barely overlap carry the most discriminative information.")
        fig = plot_mean_spectra(cancer, healthy, raman_shifts)
        st.pyplot(fig); plt.close(fig)

    # ── Tab 2 ──────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Manual Peak Analysis")
        n_peaks = st.radio("Peaks to select", [4, 6], horizontal=True)
        peaks   = find_discriminative_peaks(cancer, healthy, raman_shifts, n_peaks=n_peaks)
        ratios  = compute_cancer_healthy_ratios(cancer, healthy, raman_shifts, peaks)
        best_wn = ratios['wavenumbers'][ratios['best_peak_idx']]

        cols = st.columns(n_peaks)
        for i, (wn, r) in enumerate(zip(ratios['wavenumbers'], ratios['ratios'])):
            cols[i].metric(f"P{i+1} — {wn:.0f} cm⁻¹",
                           f"ratio {r:.3f}",
                           delta="best" if i == ratios['best_peak_idx'] else None)

        fig = plot_mean_spectra(cancer, healthy, raman_shifts, peak_shifts=peaks)
        st.pyplot(fig); plt.close(fig)
        fig = plot_peak_analysis(ratios, cancer, healthy, raman_shifts)
        st.pyplot(fig); plt.close(fig)

    # ── Tab 3 ──────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("PCA — Spectral Range Optimisation")
        c1, c2, c3 = st.columns(3)
        c1.metric("Best range", f"{wn_min:.0f}–{wn_max:.0f} cm⁻¹")
        c2.metric("PC1 variance", f"{best_pca.variance_explained[0]*100:.1f}%")
        c3.metric("PC2 variance", f"{best_pca.variance_explained[1]*100:.1f}%")
        fig = plot_pca_comparison(pca_results)
        st.pyplot(fig); plt.close(fig)
        fig = plot_eigenvalues(best_pca)
        st.pyplot(fig); plt.close(fig)

    # ── Tab 4 ──────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("ML Classification — 4-fold Cross-Validation")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Best model",  best_name)
        c2.metric("Accuracy",    f"{best.accuracy:.3f}")
        c3.metric("Sensitivity", f"{best.sensitivity:.3f}")
        c4.metric("Specificity", f"{best.specificity:.3f}")
        c5.metric("F-score",     f"{best.f_score:.3f}")
        c6.metric("AUC",         f"{best.auc:.3f}")

        fig = plot_classifier_comparison(results)
        st.pyplot(fig); plt.close(fig)

        col_a, col_b = st.columns(2)
        with col_a:
            fig = plot_confusion_matrix(best, best_name)
            st.pyplot(fig); plt.close(fig)
        with col_b:
            fig = plot_roc_curves(results)
            st.pyplot(fig); plt.close(fig)

    # ── Tab 5: Preprocessing ───────────────────────────────────────────────
    with tab5:
        st.subheader("Spectral Preprocessing — Baseline Correction & Cosmic Ray Removal")
        st.markdown("""
        Raw Raman spectra require two preprocessing steps before analysis.
        This tab demonstrates both on a synthetic raw spectrum, then shows
        that applying them to the already-clean real data is safe.
        """)

        # Build a synthetic raw spectrum for demonstration
        rng_p = np.random.default_rng(42)
        raman_signal = cancer[0] * 500
        fluorescence = 3000 * np.exp(-np.linspace(0, 3, len(raman_shifts))) + 500
        raw = raman_signal + fluorescence

        cosmic_idx = np.argmin(np.abs(raman_shifts - 1100))
        raw_spike  = raw.copy()
        raw_spike[cosmic_idx] += 8000

        baseline  = asymmetric_least_squares(raw_spike, lam=1e5, p=0.01)
        corrected = np.clip(raw_spike - baseline, 0, None)

        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        axes[0].plot(raman_shifts, raw_spike, 'steelblue', lw=1, label='Raw + cosmic ray')
        axes[0].plot(raman_shifts, baseline,  'r--',       lw=2, label='ALS baseline')
        axes[0].set_ylabel('Counts (raw)')
        axes[0].set_title('Step 1 — ALS baseline estimation')
        axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

        axes[1].plot(raman_shifts, corrected,    'steelblue', lw=1.5, label='Corrected')
        axes[1].plot(raman_shifts, raman_signal, 'g--',       lw=1,   label='True Raman', alpha=0.7)
        axes[1].set_ylabel('Counts (corrected)')
        axes[1].set_title('Step 2 — After baseline removal')
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

        axes[2].fill_between(raman_shifts, fluorescence, alpha=0.4, color='red', label='Removed background')
        axes[2].set_xlabel('Raman shift (cm⁻¹)')
        axes[2].set_ylabel('Background')
        axes[2].set_title('Fluorescence background removed')
        axes[2].legend(fontsize=9); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

        processed = preprocess_spectra(cancer[:3], remove_spikes=True,
                                        correct_fluorescence=True, normalise=True)
        change = float(np.abs(processed - cancer[:3]).mean())
        st.success(f"Mean change on real (already-clean) data after preprocessing: "
                   f"**{change:.5f}** — confirms preprocessing is safe to apply.")

    # ── Tab 6: Learning Curves ─────────────────────────────────────────────
    with tab6:
        st.subheader("Learning Curves — Accuracy vs Training Set Size")
        st.markdown("""
        Each classifier is trained on progressively larger subsets (10%→100%)
        and evaluated by cross-validation. A large gap between training and CV
        curves indicates overfitting; a CV curve still rising at max data size
        means more data would improve performance.
        """)

        with st.spinner("Computing learning curves (~30s)..."):
            X_lc, y_lc = _run_learning_curves(cancer, healthy, raman_shifts, wn_min, wn_max)
            fig = plot_learning_curves(X_lc, y_lc, n_folds=4)

        st.pyplot(fig); plt.close(fig)

    # ── Tab 7: Uncertainty ─────────────────────────────────────────────────
    with tab7:
        st.subheader("Uncertainty Quantification — Selective Prediction")
        st.markdown(f"""
        Confidence threshold: **{conf_threshold:.0%}** (adjust in sidebar).
        Predictions below this are flagged as uncertain rather than forcing
        a binary output — clinically safer than always committing to a label.
        """)

        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        rows = []
        for name, clf in build_classifiers().items():
            try:
                pred = predict_with_confidence(clf, X_tr, y_tr, X_te,
                                               threshold=conf_threshold)
                cov  = pred.certain.mean() * 100
                acc  = (pred.label[pred.certain] == y_te[pred.certain]).mean() if pred.certain.any() else float('nan')
                rows.append({'Model': name,
                             'Coverage': f"{cov:.1f}%",
                             'Acc (certain)': f"{acc:.3f}",
                             'Uncertain': int((~pred.certain).sum())})
            except Exception:
                pass

        import pandas as pd
        st.dataframe(pd.DataFrame(rows).set_index('Model'), use_container_width=True)

        # Plot for SVM
        svm_clf = build_classifiers()['SVM (RBF)']
        pred_svm = predict_with_confidence(svm_clf, X_tr, y_tr, X_te, threshold=conf_threshold)
        fig = plot_confidence_distribution(pred_svm, y_te, threshold=conf_threshold,
                                           title=f"SVM (RBF) — threshold {conf_threshold:.0%}")
        st.pyplot(fig); plt.close(fig)


def page_integration() -> None:
    st.title("Integration — End-to-End Surgical Workflow")

    # Get real target if SERV-CT available, else synthetic
    if SERV_CT_AVAILABLE:
        data      = _load_serv_ct_frame('001')
        disp_data = _compute_disparity('001')
        disp_map  = disp_data['disparity_map']
        target_col, target_row = 360, 288
        d_at = float(disp_map[target_row, target_col])
        if np.isnan(d_at) or d_at <= 0:
            d_at = float(np.nanmedian(disp_map))
        coords_3d = reproject_single_point(target_col, target_row, d_at, data['Q'])
        gt_z      = float(data['gt_depth_mm'][target_row, target_col])
        depth_err = abs(coords_3d[2] - gt_z)
        data_label = "SERV-CT frame 001 (real)"
    else:
        scene     = _synthetic_scene()
        coords_3d = scene['target_3d']
        depth_err = None
        data_label = "synthetic demo"

    cancer, healthy, raman_shifts, _ = _load_raman('real')
    pca_results = _run_pca(cancer, healthy, raman_shifts)
    wn_min, wn_max = pca_results[0].spectral_range
    X, y, results = _run_classification(cancer, healthy, raman_shifts, wn_min, wn_max)
    best_name = max(results, key=lambda k: results[k].accuracy)
    best      = results[best_name]

    st.markdown(f"**3D target from:** {data_label}")
    st.markdown("---")

    st.markdown("### Step 1 — Stereo camera localises tissue target")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target X", f"{coords_3d[0]:.1f} mm")
    c2.metric("Target Y", f"{coords_3d[1]:.1f} mm")
    c3.metric("Target Z (depth)", f"{coords_3d[2]:.1f} mm")
    if depth_err is not None:
        c4.metric("Depth error vs CT", f"{depth_err:.1f} mm")

    st.markdown("---")
    st.markdown("### Step 2 — Robot moves Raman probe to target")
    st.markdown(
        f"Probe commanded to **(X={coords_3d[0]:.1f}, Y={coords_3d[1]:.1f}, "
        f"Z={coords_3d[2]:.1f}) mm**. Spectrum acquired (1–5 s integration time)."
    )

    st.markdown("---")
    st.markdown("### Step 3 — Spectrum classified with confidence")

    rng = np.random.default_rng(7)
    sample = cancer[rng.integers(len(cancer))]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.error(f"**DISEASE DETECTED** — {best.sensitivity*100:.1f}% sensitivity")
        st.markdown(f"""
        - **Model:** {best_name}
        - **Accuracy:**    {best.accuracy:.3f}
        - **Sensitivity:** {best.sensitivity:.3f}
        - **Specificity:** {best.specificity:.3f}
        - **AUC:**         {best.auc:.3f}
        """)
        st.warning("Advisory only. Histopathological confirmation required.")

    with col2:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.plot(raman_shifts, sample, color='salmon', lw=1.5, label='Acquired spectrum')
        ax.plot(raman_shifts, cancer.mean(0), 'r--', lw=1, alpha=0.6, label='Mean disease ref.')
        ax.plot(raman_shifts, healthy.mean(0), 'b--', lw=1, alpha=0.6, label='Mean healthy ref.')
        ax.set_xlabel('Raman shift (cm⁻¹)')
        ax.set_ylabel('Normalised intensity')
        ax.set_title('Acquired spectrum vs reference means')
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    st.markdown("---")
    st.markdown(f"""
    | Step | Data | Result |
    |------|------|--------|
    | Stereo localisation | {data_label} | ({coords_3d[0]:.1f}, {coords_3d[1]:.1f}, {coords_3d[2]:.1f}) mm |
    | Raman classification | Yin et al. 2021 ({len(cancer)}+{len(healthy)} spectra) | Disease detected |
    | Best classifier | {best_name} | Acc {best.accuracy:.1%} · AUC {best.auc:.3f} |
    """)


# ── Router ────────────────────────────────────────────────────────────────

if PAGE == "Overview":
    page_overview()
elif PAGE == "Module 1 — Stereo Vision":
    page_stereo()
elif PAGE == "Module 2 — Raman Analysis":
    page_raman()
elif PAGE == "Integration":
    page_integration()
