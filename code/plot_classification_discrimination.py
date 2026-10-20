"""Plot primary horizon discrimination and model-level mean performance."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from classify_fixed_horizons import OUT_DIR, RANDOM_STATE, make_preprocessor, model_defs


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

HORIZONS = [24, 36, 60]
FEATURE_SET = "Top50"
RESULTS_PATH = OUT_DIR / "horizon_model_results_long.csv"
PLOT_DIR = OUT_DIR / "publication_figures"
PLOT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_PATH = PLOT_DIR / "figure_discrimination_performance_60m.png"
FIGURE_VERTICAL_PATH = PLOT_DIR / "figure_discrimination_performance_60m_vertical.png"
FIGURE_ALL_HORIZONS_PATH = PLOT_DIR / "figure_discrimination_performance_24_36_60m.png"
FIGURE_MEAN_PERFORMANCE_PATH = PLOT_DIR / "figure_mean_model_performance_24_36_60m.png"
MEAN_PERFORMANCE_TABLE_PATH = PLOT_DIR / "mean_model_performance_24_36_60m.csv"

FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8

MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "LightGBM"]
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "CatBoost": "CatBoost",
    "LightGBM": "LightGBM",
}
COLORS = {
    "LogisticRegression": "#2F6FDB",
    "RandomForest": "#F28E2B",
    "XGBoost": "#59A14F",
    "CatBoost": "#E15759",
    "LightGBM": "#8E6BBE",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def dataset_path(horizon: int) -> Path:
    return OUT_DIR / f"top50_{horizon}m_dataset.csv"


def predictions_path(horizon: int) -> Path:
    return PLOT_DIR / f"oof_predictions_{horizon}m_top50.csv"


def load_reported_metrics(horizon: int) -> dict[str, dict[str, float]]:
    results = pd.read_csv(RESULTS_PATH)
    results = results[(results["horizon"] == horizon) & (results["feature_set"] == FEATURE_SET)]
    out = {}
    for _, row in results.iterrows():
        out[row["model"]] = {
            "roc_auc": float(row["roc_auc_mean"]),
            "pr_auc": float(row["pr_auc_mean"]),
        }
    return out


def collect_oof_predictions(horizon: int) -> pd.DataFrame:
    path = dataset_path(horizon)
    data = pd.read_csv(path)
    if "label" not in data.columns:
        raise ValueError(f"{path} must contain a label column.")
    x = data.drop(columns=["label"])
    y = data["label"].astype(int).to_numpy()

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    models = model_defs()
    rows = []

    for model_name in MODEL_ORDER:
        estimator, scale_numeric = models[model_name]
        print(f"Collecting out-of-fold predictions: {horizon}m | {model_name}")
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            pipe = Pipeline(
                [
                    ("preprocess", make_preprocessor(x_train, scale_numeric=scale_numeric)),
                    ("model", clone(estimator)),
                ]
            )
            pipe.fit(x_train, y_train)
            y_score = pipe.predict_proba(x_test)[:, 1]
            for subject_index, true_label, score in zip(test_idx, y_test, y_score):
                rows.append(
                    {
                        "model": model_name,
                        "horizon": horizon,
                        "fold": fold_idx,
                        "subject_index": int(subject_index),
                        "label": int(true_label),
                        "score": float(score),
                    }
                )

    pred = pd.DataFrame(rows)
    pred.to_csv(predictions_path(horizon), index=False)
    return pred


def load_or_collect_oof_predictions(horizon: int) -> pd.DataFrame:
    path = predictions_path(horizon)
    if path.exists():
        pred = pd.read_csv(path)
        pred["horizon"] = horizon
        print(f"Loaded existing out-of-fold predictions: {path}")
        return pred
    return collect_oof_predictions(horizon)


def plot_single_horizon_vertical(pred: pd.DataFrame, horizon: int) -> None:
    reported = load_reported_metrics(horizon)
    set_style()
    fig, axes = plt.subplots(2, 1, figsize=(3.6, 6.2))

    roc_ax, pr_ax = axes
    for model_name in MODEL_ORDER:
        sub = pred[pred["model"] == model_name]
        y_true = sub["label"].to_numpy()
        y_score = sub["score"].to_numpy()

        fpr, tpr, _ = roc_curve(y_true, y_score)
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pooled_roc_auc = roc_auc_score(y_true, y_score)
        pooled_pr_auc = average_precision_score(y_true, y_score)
        legend_roc = reported.get(model_name, {}).get("roc_auc", pooled_roc_auc)
        legend_pr = reported.get(model_name, {}).get("pr_auc", pooled_pr_auc)
        label_roc = f"{MODEL_LABELS[model_name]} ({legend_roc:.3f})"
        label_pr = f"{MODEL_LABELS[model_name]} ({legend_pr:.3f})"

        roc_ax.plot(fpr, tpr, color=COLORS[model_name], linewidth=1.35, label=label_roc)
        pr_ax.plot(recall, precision, color=COLORS[model_name], linewidth=1.35, label=label_pr)

    roc_ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.9)
    roc_ax.set_title(f"ROC curves ({horizon}-month horizon)", fontweight="bold")
    roc_ax.set_xlabel("False positive rate (1 - specificity)")
    roc_ax.set_ylabel("True positive rate (sensitivity)")
    roc_ax.set_xlim(0, 1)
    roc_ax.set_ylim(0, 1)
    roc_ax.legend(title="Model (AUC)", loc="lower right", frameon=True, title_fontsize=FONT_SIZE)

    pr_ax.set_title(f"Precision-recall curves ({horizon}-month horizon)", fontweight="bold")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.set_xlim(0, 1)
    pr_ax.set_ylim(0, 1)
    pr_ax.legend(title="Model (PR-AUC)", loc="lower left", frameon=True, title_fontsize=FONT_SIZE)

    for ax in axes:
        ax.grid(False)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(FONT_FAMILY)
            label.set_fontsize(FONT_SIZE)

    fig.suptitle(
        "Discrimination performance (classification cohort)",
        x=0.01,
        y=1.01,
        ha="left",
        fontsize=FONT_SIZE + 1,
        fontweight="bold",
    )
    fig.tight_layout()
    out_path = PLOT_DIR / f"figure_discrimination_performance_{horizon}m_vertical.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved vertical figure: {out_path}")


def plot_all_horizons(all_predictions: dict[int, pd.DataFrame]) -> None:
    set_style()
    fig, axes = plt.subplots(len(HORIZONS), 2, figsize=(7.2, 8.8))

    for row_idx, horizon in enumerate(HORIZONS):
        pred = all_predictions[horizon]
        reported = load_reported_metrics(horizon)
        roc_ax = axes[row_idx, 0]
        pr_ax = axes[row_idx, 1]

        for model_name in MODEL_ORDER:
            sub = pred[pred["model"] == model_name]
            y_true = sub["label"].to_numpy()
            y_score = sub["score"].to_numpy()

            fpr, tpr, _ = roc_curve(y_true, y_score)
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pooled_roc_auc = roc_auc_score(y_true, y_score)
            pooled_pr_auc = average_precision_score(y_true, y_score)
            legend_roc = reported.get(model_name, {}).get("roc_auc", pooled_roc_auc)
            legend_pr = reported.get(model_name, {}).get("pr_auc", pooled_pr_auc)

            roc_ax.plot(
                fpr,
                tpr,
                color=COLORS[model_name],
                linewidth=1.2,
                label=f"{MODEL_LABELS[model_name]} ({legend_roc:.3f})",
            )
            pr_ax.plot(
                recall,
                precision,
                color=COLORS[model_name],
                linewidth=1.2,
                label=f"{MODEL_LABELS[model_name]} ({legend_pr:.3f})",
            )

        roc_ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=0.8)
        roc_ax.set_title(f"ROC curves ({horizon}-month horizon)", fontweight="bold")
        roc_ax.set_xlabel("False positive rate (1 - specificity)")
        roc_ax.set_ylabel("True positive rate (sensitivity)")
        roc_ax.set_xlim(0, 1)
        roc_ax.set_ylim(0, 1)
        roc_ax.legend(title="Model (AUC)", loc="lower right", frameon=True, title_fontsize=FONT_SIZE - 1)

        pr_ax.set_title(f"Precision-recall curves ({horizon}-month horizon)", fontweight="bold")
        pr_ax.set_xlabel("Recall")
        pr_ax.set_ylabel("Precision")
        pr_ax.set_xlim(0, 1)
        pr_ax.set_ylim(0, 1)
        pr_ax.legend(title="Model (PR-AUC)", loc="lower left", frameon=True, title_fontsize=FONT_SIZE - 1)

    for ax in axes.ravel():
        ax.grid(False)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(FONT_FAMILY)
            label.set_fontsize(FONT_SIZE)

    fig.suptitle(
        "Discrimination performance across fixed prediction horizons",
        x=0.01,
        y=1.005,
        ha="left",
        fontsize=FONT_SIZE + 1,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(FIGURE_ALL_HORIZONS_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved all-horizons figure: {FIGURE_ALL_HORIZONS_PATH}")


def plot_mean_model_performance() -> None:
    results = pd.read_csv(RESULTS_PATH)
    results = results[
        (results["horizon"].isin(HORIZONS))
        & (results["feature_set"] == FEATURE_SET)
        & (results["model"].isin(MODEL_ORDER))
    ].copy()

    summary_rows = []
    for horizon in HORIZONS:
        horizon_rows = results[results["horizon"] == horizon]
        summary_rows.append(
            {
                "horizon": horizon,
                "mean_roc_auc_across_models": horizon_rows["roc_auc_mean"].mean(),
                "std_roc_auc_across_models": horizon_rows["roc_auc_mean"].std(ddof=1),
                "mean_pr_auc_across_models": horizon_rows["pr_auc_mean"].mean(),
                "std_pr_auc_across_models": horizon_rows["pr_auc_mean"].std(ddof=1),
                "mean_f1_across_models": horizon_rows["f1_mean"].mean(),
                "std_f1_across_models": horizon_rows["f1_mean"].std(ddof=1),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(MEAN_PERFORMANCE_TABLE_PATH, index=False)

    set_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    x = np.asarray(HORIZONS)

    for model_name in MODEL_ORDER:
        sub = results[results["model"] == model_name].sort_values("horizon")
        ax.plot(
            sub["horizon"],
            sub["roc_auc_mean"],
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            color=COLORS[model_name],
            alpha=0.75,
            label=MODEL_LABELS[model_name],
        )

    ax.errorbar(
        x,
        summary["mean_roc_auc_across_models"],
        yerr=summary["std_roc_auc_across_models"],
        color="black",
        marker="s",
        markersize=4,
        linewidth=1.8,
        capsize=3,
        label="Mean across models",
        zorder=10,
    )

    ax.set_title("Mean model discrimination across prediction horizons", fontweight="bold")
    ax.set_xlabel("Prediction horizon (months)")
    ax.set_ylabel("Mean ROC-AUC")
    ax.set_xticks(HORIZONS)
    ax.set_ylim(0.80, 0.94)
    ax.legend(loc="lower right", frameon=True, ncol=1)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, color="0.75")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname(FONT_FAMILY)
        label.set_fontsize(FONT_SIZE)
    fig.tight_layout()
    fig.savefig(FIGURE_MEAN_PERFORMANCE_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved mean performance figure: {FIGURE_MEAN_PERFORMANCE_PATH}")
    print(f"Saved mean performance table: {MEAN_PERFORMANCE_TABLE_PATH}")


def main() -> None:
    all_predictions = {horizon: load_or_collect_oof_predictions(horizon) for horizon in HORIZONS}
    plot_all_horizons(all_predictions)
    plot_single_horizon_vertical(all_predictions[60], 60)
    plot_mean_model_performance()


if __name__ == "__main__":
    main()
