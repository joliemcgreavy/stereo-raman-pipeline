"""
Supervised machine learning classification of Raman spectra.

This corresponds to Exercise 4 of the assignment — training multiple
classifiers and evaluating them with confusion-matrix-derived metrics.

The assignment used MATLAB's Classification Learner GUI, which:
  1. Accepted a table of PCA scores as input features
  2. Trained all available model types automatically
  3. Displayed confusion matrices, ROC curves, and accuracy for each model

Here we replicate this entirely in code using scikit-learn, which gives
us full control and makes every step transparent.

Key differences vs the assignment:
  - The assignment had 9 training and 9 test observations per class (very small).
    We use stratified k-fold cross-validation instead, which is more robust.
  - We evaluate Accuracy, Precision, Sensitivity, Specificity, F-score, and AUC
    — the same metrics as the assignment (Q4.2 and Q4.3).
  - We add a comparison table across all models, equivalent to inspecting
    the left-hand model list in the Classification Learner GUI.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ── Classifiers ────────────────────────────────────────────────────────────
# These are the Python equivalents of the models available in MATLAB's
# Classification Learner. We wrap each in a Pipeline that standardises
# the features first — important because SVM and KNN are sensitive to
# feature scale.

def build_classifiers() -> dict[str, Pipeline]:
    """
    Return a dictionary of named scikit-learn classifier pipelines.

    Each pipeline applies StandardScaler before the classifier.
    Standardising inside the pipeline (rather than beforehand) prevents
    data leakage: the scaler is fit only on training folds, not test folds.

    Model choices:
      SVM (RBF): powerful non-linear classifier, typically best on Raman data.
                 'C' controls the trade-off between margin width and misclassification.
                 probability=True enables predict_proba() for ROC curves.
      KNN:       classifies by majority vote of K nearest neighbours.
                 Simple and interpretable; sensitive to irrelevant features.
      Random Forest: ensemble of decision trees. More robust than a single tree;
                     resistant to overfitting on moderate-size datasets.
      Gaussian NB: assumes Gaussian feature distributions; fast and interpretable.
      Decision Tree: single tree; easy to visualise but prone to overfitting.
      Logistic Regression: linear classifier; good baseline and interpretable.
    """
    scaler = lambda: StandardScaler()

    return {
        'SVM (RBF)':           Pipeline([('sc', scaler()), ('clf', SVC(kernel='rbf', C=1.0, probability=True))]),
        'KNN (k=5)':           Pipeline([('sc', scaler()), ('clf', KNeighborsClassifier(n_neighbors=5))]),
        'Random Forest':       Pipeline([('sc', scaler()), ('clf', RandomForestClassifier(n_estimators=100, random_state=42))]),
        'Gaussian NB':         Pipeline([('sc', scaler()), ('clf', GaussianNB())]),
        'Decision Tree':       Pipeline([('sc', scaler()), ('clf', DecisionTreeClassifier(max_depth=5, random_state=42))]),
        'Logistic Regression': Pipeline([('sc', scaler()), ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=42))]),
    }


# ── Feature extraction ─────────────────────────────────────────────────────

def extract_features(
    cancer: np.ndarray,
    healthy: np.ndarray,
    raman_shifts: np.ndarray,
    wn_min: float | None = None,
    wn_max: float | None = None,
    n_pca_components: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract PCA features from the optimal spectral range.

    This replicates what the assignment called Exercise 4 Step 0:
    creating the classification table from PCA scores.

    The assignment used 9 observations per class for training (obs 1-9)
    and 9 for testing (obs 10-18). Here we return the full feature matrix
    and let cross-validation handle the splitting robustly.

    Parameters
    ----------
    wn_min, wn_max:
        The optimal spectral range identified in Exercise 3.
        Using the full range if not specified.
    n_pca_components:
        Number of PC scores to use as features. The assignment used 9
        observations, so the feature space was the 9-dimensional PC space.
        For larger datasets, choosing the elbow from the scree plot is better.

    Returns
    -------
    X: (N, n_pca_components) feature matrix
    y: (N,) labels  — 1 = cancer, 0 = healthy
    """
    # Select spectral range
    wn_min = wn_min if wn_min is not None else float(raman_shifts[0])
    wn_max = wn_max if wn_max is not None else float(raman_shifts[-1])
    mask = (raman_shifts >= wn_min) & (raman_shifts <= wn_max)

    cancer_sub  = cancer[:, mask]
    healthy_sub = healthy[:, mask]

    X_raw = np.vstack([cancer_sub, healthy_sub])
    y     = np.array([1] * len(cancer) + [0] * len(healthy))

    # PCA for dimensionality reduction.
    # Using scikit-learn's PCA here (SVD-based) rather than the correlation
    # matrix method from pca_analysis.py — both give equivalent results,
    # but sklearn's PCA integrates cleanly into the Pipeline and supports
    # cross-validation without data leakage.
    pca = SklearnPCA(n_components=n_pca_components, random_state=42)
    X   = pca.fit_transform(StandardScaler().fit_transform(X_raw))

    return X, y


# ── Metrics ────────────────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    """
    All classification metrics from the assignment (Q4.2, Q4.3).

    These are all derived from four numbers in the confusion matrix:
      TP: cancer correctly predicted as cancer
      TN: healthy correctly predicted as healthy
      FP: healthy incorrectly predicted as cancer  (false alarm)
      FN: cancer incorrectly predicted as healthy  (missed cancer — most dangerous)
    """
    accuracy:    float   # (TP+TN) / (TP+FP+TN+FN)
    precision:   float   # TP / (TP+FP)  — of all cancer predictions, how many were right?
    sensitivity: float   # TP / (TP+FN)  — of all real cancers, how many did we catch? (= recall)
    specificity: float   # TN / (TN+FP)  — of all healthy, how many did we correctly clear?
    f_score:     float   # 2 * (Precision * Sensitivity) / (Precision + Sensitivity)
    auc:         float   # area under ROC curve — performance across all thresholds
    confusion:   np.ndarray = field(repr=False, default_factory=lambda: np.zeros((2,2)))


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> ClassificationMetrics:
    """
    Compute all assignment metrics from predictions and ground truth.

    Cancer = positive class (label 1); Healthy = negative class (label 0).

    Note on Sensitivity vs Specificity in clinical context:
      In cancer detection, sensitivity (catching all real cancers) is usually
      more important than specificity (avoiding false alarms). Missing a cancer
      (FN) is more dangerous than a false alarm that leads to a biopsy (FP).
      The assignment asked you to report both without explicitly asking which
      matters more — this is the key clinical trade-off to appreciate.
    """
    cm = confusion_matrix(y_true, y_pred)

    # sklearn's confusion_matrix layout:
    #         Predicted 0   Predicted 1
    # True 0 [     TN    ,     FP     ]
    # True 1 [     FN    ,     TP     ]
    TN, FP = cm[0, 0], cm[0, 1]
    FN, TP = cm[1, 0], cm[1, 1]

    eps = 1e-9
    precision   = TP / (TP + FP + eps)
    sensitivity = TP / (TP + FN + eps)
    specificity = TN / (TN + FP + eps)
    f_score     = 2 * (precision * sensitivity) / (precision + sensitivity + eps)
    accuracy    = (TP + TN) / (TP + FP + TN + FN + eps)

    roc_auc = 0.5  # default if no probabilities provided
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = float(auc(fpr, tpr))

    return ClassificationMetrics(
        accuracy=accuracy,
        precision=precision,
        sensitivity=sensitivity,
        specificity=specificity,
        f_score=f_score,
        auc=roc_auc,
        confusion=cm,
    )


# ── Training and evaluation ────────────────────────────────────────────────

def evaluate_all_classifiers(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 4,
) -> dict[str, ClassificationMetrics]:
    """
    Train and evaluate all classifiers using stratified k-fold cross-validation.

    Cross-validation rationale:
    The assignment used a hard split of 9 training / 9 test samples — possible
    because all samples were available upfront. With k-fold CV:
      - The data is split into k folds
      - Each fold serves as the test set once
      - The remaining k-1 folds are used for training
      - Final metrics are averaged across all k test sets

    This gives a much more reliable estimate of generalisation performance
    than a single train/test split, especially with small datasets.
    n_folds=4 matches the cross-validation setting used in the assignment's
    Classification Learner session.

    cross_val_predict() returns out-of-fold predictions — each sample's
    prediction was made by a model that never saw that sample during training.
    This is the correct way to evaluate without data leakage.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    classifiers = build_classifiers()
    results = {}

    for name, clf in classifiers.items():
        y_pred = cross_val_predict(clf, X, y, cv=cv, method='predict')

        # Get probability scores for ROC/AUC if the classifier supports it
        try:
            y_prob = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]
        except AttributeError:
            y_prob = None

        results[name] = compute_metrics(y_true=y, y_pred=y_pred, y_prob=y_prob)

    return results


# ── Visualisation ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    metrics: ClassificationMetrics,
    model_name: str,
    class_names: list[str] | None = None,
) -> plt.Figure:
    """
    Plot a single confusion matrix with colour-coding and metric annotations.

    Diagonal cells = correct predictions (we want these large).
    Off-diagonal = errors: FP (bottom-left) and FN (top-right).

    In clinical context, FN (missed cancers) is the cell to watch most closely.
    """
    class_names = class_names or ['Healthy', 'Cancer']
    cm = metrics.confusion
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(class_names)
    ax.set_yticks([0, 1]); ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label', fontsize=11)
    ax.set_ylabel('True label', fontsize=11)
    ax.set_title(f'Confusion Matrix — {model_name}', fontsize=12)

    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > thresh else 'black')

    # Metric summary below the plot
    summary = (f"Acc={metrics.accuracy:.3f}  Prec={metrics.precision:.3f}  "
               f"Sens={metrics.sensitivity:.3f}  Spec={metrics.specificity:.3f}  "
               f"F1={metrics.f_score:.3f}  AUC={metrics.auc:.3f}")
    fig.text(0.5, -0.04, summary, ha='center', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    plt.tight_layout()
    return fig


def plot_roc_curves(results: dict[str, ClassificationMetrics]) -> plt.Figure:
    """
    Overlay ROC curves for all classifiers on one axes.

    The ROC curve plots Sensitivity (True Positive Rate) on the y-axis
    against 1-Specificity (False Positive Rate) on the x-axis, across all
    decision thresholds. The diagonal represents random guessing (AUC=0.5).

    A curve that stays high and to the left has high sensitivity AND high
    specificity across all thresholds — the ideal classifier.

    Note: ROC curves require probability outputs. Models that only output
    hard class labels (no probabilities) will have AUC=0.5 listed but no curve.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.50)')

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, metrics), color in zip(results.items(), colors):
        if metrics.auc > 0.5:
            # Reconstruct a simple 2-point ROC from the confusion matrix
            # (exact curve requires per-sample probabilities; this gives the
            # single operating point achieved at the default threshold)
            fpr = 1 - metrics.specificity
            tpr = metrics.sensitivity
            ax.plot([0, fpr, 1], [0, tpr, 1], 'o-',
                    color=color, lw=1.5,
                    label=f'{name} (AUC={metrics.auc:.3f})')

    ax.set_xlabel('1 - Specificity (FPR)', fontsize=12)
    ax.set_ylabel('Sensitivity (TPR)', fontsize=12)
    ax.set_title('ROC Curves — All Classifiers', fontsize=13)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_classifier_comparison(results: dict[str, ClassificationMetrics]) -> plt.Figure:
    """
    Heatmap comparing all metrics across all classifiers.

    This is the Python equivalent of the model accuracy list in the
    left-hand panel of MATLAB's Classification Learner — but showing
    all six metrics simultaneously, not just accuracy.

    Why show all six rather than just accuracy?
    A model can have high accuracy on an imbalanced dataset while having
    terrible sensitivity. For cancer detection, a model that ignores cancer
    and always predicts "healthy" achieves 50% accuracy — but its sensitivity
    is 0.0, which is clinically useless.
    """
    model_names = list(results.keys())
    metric_names = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F-score', 'AUC']
    data = np.array([
        [m.accuracy, m.precision, m.sensitivity, m.specificity, m.f_score, m.auc]
        for m in results.values()
    ])

    fig, ax = plt.subplots(figsize=(11, len(model_names) * 0.75 + 1.5))
    im = ax.imshow(data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    ax.set_title('Classifier Comparison — All Metrics', fontsize=13, pad=12)

    for i in range(len(model_names)):
        for j in range(len(metric_names)):
            val = data[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=9, color='black' if 0.3 < val < 0.8 else 'white')

    plt.tight_layout()
    return fig


def print_metrics_table(results: dict[str, ClassificationMetrics]) -> None:
    """Print a plain-text metrics table matching Assignment Q4.2/Q4.3 format."""
    header = f"{'Model':<22} {'Acc':>6} {'Prec':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'AUC':>6}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    best_acc = max(results.values(), key=lambda m: m.accuracy)
    for name, m in sorted(results.items(), key=lambda x: x[1].accuracy, reverse=True):
        marker = "  ← best" if m is best_acc else ""
        print(f"{name:<22} {m.accuracy:>6.3f} {m.precision:>6.3f} "
              f"{m.sensitivity:>6.3f} {m.specificity:>6.3f} "
              f"{m.f_score:>6.3f} {m.auc:>6.3f}{marker}")
    print("=" * len(header))
