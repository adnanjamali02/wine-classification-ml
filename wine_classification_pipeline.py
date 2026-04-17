"""
Wine Classification — Multi-Model ML Pipeline
==============================================
A clean, production-inspired classification pipeline built on the
UCI Wine dataset. Trains three distinct models, benchmarks them with
cross-validation, and produces a rich set of diagnostic visualizations.

Author  : Generated Pipeline
Dataset : sklearn built-in Wine dataset (178 samples, 13 features, 3 classes)
Models  : Logistic Regression | Support Vector Machine | Gradient Boosting
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ── Global style ───────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
PALETTE   = ["#2E86AB", "#E84855", "#3BB273"]
RANDOM_ST = 42
TEST_SIZE  = 0.20


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD & INSPECT
# ══════════════════════════════════════════════════════════════════════════════
raw   = load_wine()
X     = pd.DataFrame(raw.data, columns=raw.feature_names)
y     = pd.Series(raw.target, name="class")
NAMES = [str(n) for n in raw.target_names]

print("=" * 58)
print("  Wine Classification  |  ML Pipeline")
print("=" * 58)
print(f"  Samples  : {X.shape[0]}")
print(f"  Features : {X.shape[1]}")
print(f"  Classes  : {NAMES}")
print(f"\n  Class distribution")
for cls, cnt in y.value_counts().sort_index().items():
    print(f"    {NAMES[cls]:<12} → {cnt} samples")
print("=" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  EXPLORATORY DATA ANALYSIS  (3 figures)
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a. Key feature distributions ─────────────────────────────────────────────
highlight = ["alcohol", "flavanoids", "color_intensity", "proline"]

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle("Feature Distributions by Wine Class", fontsize=14, fontweight="bold", y=1.02)

for ax, feat in zip(axes, highlight):
    for idx, name in enumerate(NAMES):
        vals = X.loc[y == idx, feat]
        ax.hist(vals, bins=14, alpha=0.60, label=name, color=PALETTE[idx], edgecolor="white")
    ax.set_title(feat.replace("_", " ").title(), fontsize=11)
    ax.set_xlabel("Value", fontsize=9)
    ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig("01_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  01_feature_distributions.png saved")


# ── 2b. Correlation heatmap ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 10))
corr = X.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))          # upper triangle hidden

sns.heatmap(
    corr, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.4,
    ax=ax, annot_kws={"size": 7.5},
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("02_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  02_correlation_heatmap.png saved")


# ── 2c. PCA 2-D scatter ────────────────────────────────────────────────────────
_scaler_pca = StandardScaler()
_pca        = PCA(n_components=2, random_state=RANDOM_ST)
X_pca       = _pca.fit_transform(_scaler_pca.fit_transform(X))
var_exp     = _pca.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8, 6))
for idx, name in enumerate(NAMES):
    m = y == idx
    ax.scatter(
        X_pca[m, 0], X_pca[m, 1],
        label=name, color=PALETTE[idx],
        s=55, alpha=0.75, edgecolors="white", linewidth=0.6,
    )
ax.set_xlabel(f"PC1  ({var_exp[0]:.1f} % variance)", fontsize=11)
ax.set_ylabel(f"PC2  ({var_exp[1]:.1f} % variance)", fontsize=11)
ax.set_title("PCA — 2-D Projection of Wine Classes", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("03_pca_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  03_pca_scatter.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# 3.  TRAIN / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════════
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_ST,
    stratify=y,                # preserves class proportions in both splits
)
print(f"\n  Train : {X_train.shape[0]} samples")
print(f"  Test  : {X_test.shape[0]} samples")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL PIPELINES
#     Every pipeline owns its own StandardScaler so there is
#     absolutely no data-leakage between train and test.
# ══════════════════════════════════════════════════════════════════════════════
pipelines = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(
            C=1.0, max_iter=5000, solver="lbfgs",
            random_state=RANDOM_ST,
        )),
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  SVC(
            kernel="rbf", C=5.0, gamma="scale",
            probability=True, random_state=RANDOM_ST,
        )),
    ]),
    "Gradient Boosting": Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.08,
            max_depth=3, subsample=0.85,
            random_state=RANDOM_ST,
        )),
    ]),
}


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAIN  +  CROSS-VALIDATE  +  EVALUATE
# ══════════════════════════════════════════════════════════════════════════════
results     = {}
predictions = {}

print("\n" + "=" * 58)
print("  Training & Evaluation")
print("=" * 58)

for name, pipe in pipelines.items():
    # 5-fold CV on the training portion only
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")

    # Final training & test evaluation
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    results[name]     = {"accuracy": acc, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    predictions[name] = y_pred

    print(f"\n  ▸ {name}")
    print(f"    Test Accuracy  : {acc:.4f}")
    print(f"    CV  (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    report = classification_report(y_test, y_pred, target_names=NAMES)
    for line in report.splitlines():
        print(f"    {line}")

print("=" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MODEL COMPARISON  (bar charts)
# ══════════════════════════════════════════════════════════════════════════════
res_df = pd.DataFrame(results).T.reset_index()
res_df.columns = ["Model", "Test Accuracy", "CV Mean", "CV Std"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison", fontsize=14, fontweight="bold")

# — Test accuracy
bars = axes[0].bar(
    res_df["Model"], res_df["Test Accuracy"],
    color=PALETTE, width=0.45, edgecolor="white",
)
axes[0].set_ylim(0.85, 1.03)
axes[0].set_title("Test-Set Accuracy", fontsize=12)
axes[0].set_ylabel("Accuracy")
axes[0].tick_params(axis="x", rotation=14)
for bar, val in zip(bars, res_df["Test Accuracy"]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
        f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold",
    )

# — CV scores with error bars
axes[1].bar(
    res_df["Model"], res_df["CV Mean"],
    yerr=res_df["CV Std"], color=PALETTE, width=0.45,
    edgecolor="white", capsize=7, alpha=0.85,
)
axes[1].set_ylim(0.85, 1.05)
axes[1].set_title("5-Fold CV Accuracy  (mean ± std)", fontsize=12)
axes[1].set_ylabel("CV Accuracy")
axes[1].tick_params(axis="x", rotation=14)

plt.tight_layout()
plt.savefig("04_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  04_model_comparison.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  CONFUSION MATRICES
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

for ax, (name, y_pred) in zip(axes, predictions.items()):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        ax=ax, xticklabels=NAMES, yticklabels=NAMES,
        linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(name, fontsize=10, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    ax.tick_params(axis="x", rotation=20, labelsize=8)
    ax.tick_params(axis="y", rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig("05_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  05_confusion_matrices.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FEATURE IMPORTANCE  (Gradient Boosting)
# ══════════════════════════════════════════════════════════════════════════════
gb_inner    = pipelines["Gradient Boosting"]["model"]
importance  = pd.Series(gb_inner.feature_importances_, index=X.columns)
top_feats   = importance.nlargest(10).sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(top_feats.index, top_feats.values, color="#2E86AB", edgecolor="white")

for bar, val in zip(bars, top_feats.values):
    ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)

ax.set_title("Top 10 Feature Importances  (Gradient Boosting)", fontsize=13, fontweight="bold")
ax.set_xlabel("Importance Score", fontsize=11)
ax.set_xlim(0, top_feats.max() + 0.04)
plt.tight_layout()
plt.savefig("06_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  ✔  06_feature_importance.png saved")


# ══════════════════════════════════════════════════════════════════════════════
# 9.  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
best = max(results, key=lambda k: results[k]["accuracy"])

print("\n" + "=" * 58)
print("  FINAL SUMMARY")
print("=" * 58)
print(f"  {'Model':<28} {'Test Acc':>9}  {'CV Mean':>8}  {'CV Std':>7}")
print(f"  {'-'*54}")
for name, r in results.items():
    marker = "◀ best" if name == best else ""
    print(f"  {name:<28} {r['accuracy']:>9.4f}  {r['cv_mean']:>8.4f}  {r['cv_std']:>7.4f}  {marker}")
print("=" * 58)
print(f"\n  🏆  Best Model  →  {best}")
print(f"      Test Accuracy : {results[best]['accuracy']:.4f}")
print(f"      CV Score      : {results[best]['cv_mean']:.4f} ± {results[best]['cv_std']:.4f}")
print("\n  All 6 plots saved as PNG files.\n")
