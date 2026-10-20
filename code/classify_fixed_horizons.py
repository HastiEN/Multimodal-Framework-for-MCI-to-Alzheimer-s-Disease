"""Evaluate 24/36/60-month outcomes using fixed publication subsets."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import re
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
COHORT_PATH = (
    PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"
    if (PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv").exists()
    else PROCESSED_DIR / "2-strict_mci_to_ad_publication_cohort.csv"
)
DATA_PATH = (
    PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
    if (PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv").exists()
    else PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
)
RANKING_DIR = PROCESSED_DIR / "publication_feature_ranking"
OUT_DIR = PROCESSED_DIR / "prediction_horizon_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [24, 36, 60]
TOP_FILES = {
    "Top20": RANKING_DIR / "top20_features.csv",
    "Top50": RANKING_DIR / "top50_features.csv",
    "Top100": RANKING_DIR / "top100_features.csv",
}

FOLD_RESULTS_PATH = OUT_DIR / "horizon_cv_fold_results.csv"
SUMMARY_RESULTS_PATH = OUT_DIR / "horizon_model_results_long.csv"
COMPARISON_PATH = OUT_DIR / "prediction_horizon_comparison.csv"
TREND_REPORT_PATH = OUT_DIR / "prediction_horizon_trend_report.txt"
FIGURE_PATH = OUT_DIR / "prediction_horizon_auc.png"

RANDOM_STATE = 42
LEAKAGE_PATTERNS = [
    r"^PTID$",
    r"^RID$",
    r"DIAG",
    r"\bDX\b",
    r"CONVERT",
    r"CONVERSION",
    r"FOLLOWUP",
    r"FUTURE",
    r"LABEL",
    r"BASELINE_DATE",
    r"BASELINE_VISIT",
    r"LAST_FOLLOWUP",
    r"MONTHS_TO_CONVERSION",
    r"VISCODE",
    r"EXAMDATE",
    r"SCANDATE",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days / 30.4375


def build_horizon_labels(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = cohort["PTID"].astype(str).str.strip()
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort["conversion_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype(int)
    cohort["months_to_conversion"] = pd.to_numeric(cohort["months_to_conversion"], errors="coerce")
    cohort["followup_months"] = months_between(cohort["last_followup_date"], cohort["baseline_date"])

    rows = []
    for _, row in cohort.iterrows():
        included = False
        label = np.nan
        reason = ""
        notes = ""
        if bool(row.get("missing_conversion_date", False)) and row["conversion_label"] == 1:
            reason = "Excluded: converter with missing conversion date; horizon status uncertain"
        elif row["conversion_label"] == 1 and pd.notna(row["months_to_conversion"]):
            included = True
            label = int(row["months_to_conversion"] <= horizon)
            notes = f"Converted within {horizon} months" if label == 1 else f"Converted after {horizon} months"
        elif row["conversion_label"] == 0:
            if pd.notna(row["followup_months"]) and row["followup_months"] >= horizon:
                included = True
                label = 0
                notes = f"Stable/non-AD with at least {horizon} months follow-up"
            else:
                reason = f"Excluded: stable/non-AD but censored before {horizon} months"
        else:
            reason = "Excluded: insufficient timing information"
        rows.append(
            {
                "PTID": row["PTID"],
                "horizon_months": horizon,
                "baseline_date": row["baseline_date"],
                "conversion_label_any_future": row["conversion_label"],
                "label": label,
                "included": included,
                "exclusion_reason": reason,
                "months_to_conversion": row["months_to_conversion"],
                "followup_months": row["followup_months"],
                "conversion_date": row["conversion_date"],
                "last_followup_date": row["last_followup_date"],
                "notes": notes,
            }
        )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_DIR / f"conversion_{horizon}m_cohort_audit.csv", index=False)
    eligible = audit[audit["included"]].copy()
    eligible["label"] = eligible["label"].astype(int)
    return eligible[["PTID", "label"]], audit


def target_column(df: pd.DataFrame) -> str:
    for col in ["label", "conversion_label", "converter_label", "converter"]:
        if col in df.columns:
            return col
    raise ValueError("No target column found.")


def load_feature_list(feature_set: str) -> list[str]:
    df = pd.read_csv(TOP_FILES[feature_set])
    return df["feature"].dropna().astype(str).tolist()


def is_leakage_feature(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def build_horizon_datasets(horizon: int, labels: pd.DataFrame) -> dict[str, pd.DataFrame]:
    full = pd.read_csv(DATA_PATH)
    original_target = target_column(full)
    full["PTID"] = full["PTID"].astype(str).str.strip()
    full = full.drop(columns=[original_target]).merge(labels, on="PTID", how="inner")
    full.to_csv(OUT_DIR / f"final_mci_to_ad_{horizon}m_dataset.csv", index=False)

    datasets = {}
    for feature_set in ["Top20", "Top50", "Top100"]:
        features = load_feature_list(feature_set)
        missing = [feature for feature in features if feature not in full.columns]
        if missing:
            raise ValueError(f"{feature_set} missing features: {missing[:10]}")
        leakage = [feature for feature in features if is_leakage_feature(feature)]
        if leakage:
            raise ValueError(f"{feature_set} has leakage-like features: {leakage}")
        data = full[["label"] + list(dict.fromkeys(features))].copy()
        data.to_csv(OUT_DIR / f"{feature_set.lower()}_{horizon}m_dataset.csv", index=False)
        datasets[feature_set] = data
    return datasets


def coerce_numeric_like(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = pd.to_numeric(out[col], errors="coerce")
            non_missing = out[col].notna().sum()
            if non_missing and numeric.notna().sum() / non_missing >= 0.8:
                out[col] = numeric
    return out


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            (
                "cat",
                Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def model_defs() -> dict[str, tuple[object, bool]]:
    return {
        "LogisticRegression": (
            LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE),
            True,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "XGBoost": (
            XGBClassifier(
                n_estimators=500,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        ),
        "CatBoost": (
            CatBoostClassifier(
                iterations=1000,
                depth=6,
                learning_rate=0.03,
                loss_function="Logloss",
                eval_metric="AUC",
                verbose=False,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "LightGBM": (
            LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=RANDOM_STATE,
                verbose=-1,
            ),
            False,
        ),
    }


def evaluate_predictions(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def run_horizon_cv(horizon: int, datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for feature_set, data in datasets.items():
        x = coerce_numeric_like(data.drop(columns=["label"]))
        y = data["label"].astype(int).to_numpy()
        for model_name, (model, scale_numeric) in model_defs().items():
            log(f"Evaluating {horizon}m | {model_name} | {feature_set}")
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                pipe = Pipeline(
                    [
                        ("preprocess", make_preprocessor(x_train, scale_numeric)),
                        ("model", clone(model)),
                    ]
                )
                pipe.fit(x_train, y_train)
                y_score = pipe.predict_proba(x_test)[:, 1]
                metrics = evaluate_predictions(y_test, y_score)
                metrics.update(
                    {
                        "horizon": horizon,
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold,
                        "n_features": x.shape[1],
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def summarize_folds(folds: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "recall", "f1"]
    summary = folds.groupby(["horizon", "model", "feature_set", "n_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    return summary.sort_values(["horizon", "roc_auc_mean", "balanced_accuracy_mean"], ascending=[True, False, False]).reset_index(drop=True)


def horizon_comparison(summary: pd.DataFrame, audits: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS:
        best = summary[summary["horizon"] == horizon].sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False).iloc[0]
        audit = audits[horizon]
        eligible = audit[audit["included"]]
        rows.append(
            {
                "Horizon": f"{horizon}m",
                "Subjects": len(eligible),
                "Positives": int((eligible["label"] == 1).sum()),
                "Negatives": int((eligible["label"] == 0).sum()),
                "Best Model": best["model"],
                "Best Features": best["feature_set"],
                "ROC-AUC": best["roc_auc_mean"],
                "ROC-AUC Std": best["roc_auc_std"],
                "PR-AUC": best["pr_auc_mean"],
                "F1": best["f1_mean"],
                "Balanced Accuracy": best["balanced_accuracy_mean"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(COMPARISON_PATH, index=False)
    return out


def top_features(feature_set: str, n: int = 30) -> list[str]:
    return load_feature_list(feature_set)[:n]


def write_trend_report(comparison: pd.DataFrame) -> None:
    lookup = comparison.set_index("Horizon")
    pairs = [("24m", "36m"), ("36m", "60m"), ("24m", "60m")]
    lines = [
        "Prediction Horizon Trend Report",
        "",
        comparison.to_string(index=False),
        "",
        "Pairwise Differences:",
    ]
    for a, b in pairs:
        lines.append(
            f"{a} vs {b}: ROC-AUC delta={lookup.loc[b, 'ROC-AUC'] - lookup.loc[a, 'ROC-AUC']:.4f}, "
            f"PR-AUC delta={lookup.loc[b, 'PR-AUC'] - lookup.loc[a, 'PR-AUC']:.4f}, "
            f"F1 delta={lookup.loc[b, 'F1'] - lookup.loc[a, 'F1']:.4f}"
        )

    best_features = {row["Horizon"]: set(top_features(row["Best Features"], 30)) for _, row in comparison.iterrows()}
    shared = set.intersection(*best_features.values()) if best_features else set()
    early_unique = best_features.get("24m", set()) - best_features.get("36m", set()) - best_features.get("60m", set())
    late_unique = best_features.get("60m", set()) - best_features.get("24m", set()) - best_features.get("36m", set())

    roc_values = comparison["ROC-AUC"].to_numpy()
    declines = all(earlier >= later for earlier, later in zip(roc_values, roc_values[1:]))
    lines.extend(
        [
            "",
            "Clinical Interpretation:",
            "Early converters may be easier to predict because baseline cognitive impairment, functional decline, and neurodegeneration are often already closer to the dementia threshold. Longer horizons include more heterogeneous subjects whose conversion depends on future biological and clinical trajectories not fully observable at baseline.",
            "",
            f"Does ROC-AUC monotonically decrease as horizon increases? {'Yes' if declines else 'No'}",
            "",
            "Features consistently present across all horizon best-feature sets:",
            ", ".join(sorted(shared)) if shared else "None",
            "",
            "Features unique to early conversion best set:",
            ", ".join(sorted(early_unique)) if early_unique else "None",
            "",
            "Features unique to late conversion best set:",
            ", ".join(sorted(late_unique)) if late_unique else "None",
            "",
            "Conclusion:",
            "The hypothesis is supported if 24-month performance exceeds 36-month and 60-month performance. If 36-month or 60-month performance is similar or higher, the data suggest that the endpoint definition and censoring exclusions may produce an easier or more balanced prediction problem at that horizon.",
        ]
    )
    TREND_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def save_figure(comparison: pd.DataFrame) -> None:
    horizons = comparison["Horizon"].str.replace("m", "", regex=False).astype(int)
    auc = comparison["ROC-AUC"]
    err = comparison["ROC-AUC Std"]
    plt.figure(figsize=(6, 4))
    plt.errorbar(horizons, auc, yerr=err, marker="o", capsize=4, linewidth=2)
    plt.xticks(horizons, comparison["Horizon"])
    plt.xlabel("Prediction horizon")
    plt.ylabel("ROC-AUC")
    plt.ylim(max(0.5, auc.min() - 0.08), min(1.0, auc.max() + 0.06))
    plt.title("MCI-to-AD Prediction Performance by Horizon")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    all_folds = []
    audits = {}
    for horizon in HORIZONS:
        labels, audit = build_horizon_labels(horizon)
        audits[horizon] = audit
        eligible = audit[audit["included"]]
        log("====================================================")
        log(f"{horizon}-MONTH COHORT")
        log("====================================================")
        log(f"Eligible subjects: {len(eligible)}")
        log(f"Positives: {int((eligible['label'] == 1).sum())}")
        log(f"Negatives: {int((eligible['label'] == 0).sum())}")
        log(f"Excluded: {int((~audit['included']).sum())}")
        log(f"Class balance: {float((eligible['label'] == 1).mean()):.3f}")
        datasets = build_horizon_datasets(horizon, labels)
        all_folds.append(run_horizon_cv(horizon, datasets))

    folds = pd.concat(all_folds, ignore_index=True)
    folds.to_csv(FOLD_RESULTS_PATH, index=False)
    summary = summarize_folds(folds)
    summary.to_csv(SUMMARY_RESULTS_PATH, index=False)
    comparison = horizon_comparison(summary, audits)
    save_figure(comparison)
    write_trend_report(comparison)

    log("\n====================================================")
    log("PREDICTION HORIZON COMPARISON")
    log("====================================================")
    log(comparison.to_string(index=False))
    log(f"\nSaved comparison: {COMPARISON_PATH}")
    log(f"Saved trend report: {TREND_REPORT_PATH}")
    log(f"Saved figure: {FIGURE_PATH}")


if __name__ == "__main__":
    main()
