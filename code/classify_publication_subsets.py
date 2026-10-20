"""Reproduce the earlier fixed-publication-subset classification comparison."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import re
from pathlib import Path

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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
REQUESTED_DATA_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
FALLBACK_DATA_PATH = PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
DATA_PATH = REQUESTED_DATA_PATH if REQUESTED_DATA_PATH.exists() else FALLBACK_DATA_PATH
RANKING_DIR = PROCESSED_DIR / "publication_feature_ranking"
OUT_DIR = PROCESSED_DIR / "final_modeling_experiments_v2_different_repeat_seeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_FILES = {
    "Top20": RANKING_DIR / "top20_features.csv",
    "Top50": RANKING_DIR / "top50_features.csv",
    "Top100": RANKING_DIR / "top100_features.csv",
}

DATASET_OUTPUTS = {
    "Top20": OUT_DIR / "top20_dataset.csv",
    "Top50": OUT_DIR / "top50_dataset.csv",
    "Top100": OUT_DIR / "top100_dataset.csv",
}

FOLD_RESULTS_PATH = OUT_DIR / "cv_fold_results.csv"
SUMMARY_RESULTS_PATH = OUT_DIR / "model_results_long.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "model_comparison.csv"
BEST_ANALYSIS_PATH = OUT_DIR / "best_model_analysis.json"
REPORT_PATH = OUT_DIR / "final_modeling_report.txt"

RANDOM_STATE = 42
REPEAT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
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


def target_column(df: pd.DataFrame) -> str:
    for col in ["label", "conversion_label", "converter_label", "converter"]:
        if col in df.columns:
            return col
    raise ValueError("No target column found. Expected label or conversion_label.")


def load_feature_list(path: Path) -> list[str]:
    ranking = pd.read_csv(path)
    if "feature" not in ranking.columns:
        raise ValueError(f"{path} must contain a feature column.")
    return ranking["feature"].dropna().astype(str).tolist()


def is_leakage_feature(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def coerce_numeric_like(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = pd.to_numeric(out[col], errors="coerce")
            non_missing = out[col].notna().sum()
            if non_missing and numeric.notna().sum() / non_missing >= 0.8:
                out[col] = numeric
    return out


def build_modeling_datasets(df: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    datasets = {}
    for name, feature_path in TOP_FILES.items():
        features = load_feature_list(feature_path)
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"{name} has features not present in dataset: {missing[:10]}")
        leakage = [feature for feature in features if is_leakage_feature(feature)]
        if leakage:
            raise ValueError(f"{name} contains possible leakage variables: {leakage}")
        selected = [target] + list(dict.fromkeys(features))
        if len(selected) != len(set(selected)):
            raise ValueError(f"{name} has duplicate columns after selection.")
        out = df[selected].copy()
        out = out.loc[:, ~out.columns.duplicated()]
        out[target] = out[target].astype(int)
        if len(out) != out.index.nunique():
            raise ValueError(f"{name} row index is not unique.")
        out.to_csv(DATASET_OUTPUTS[name], index=False)
        datasets[name] = out
    return datasets


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric),
            ("cat", categorical_pipeline, categorical),
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
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def run_cv(datasets: dict[str, pd.DataFrame], target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    models = model_defs()
    for feature_set, data in datasets.items():
        x = coerce_numeric_like(data.drop(columns=[target]))
        y = data[target].astype(int).to_numpy()
        for model_name, (model, scale_numeric) in models.items():
            log(f"Evaluating {model_name} | {feature_set}")
            for repeat_idx, seed in enumerate(REPEAT_SEEDS, start=1):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    pipe = Pipeline(
                        [
                            ("preprocess", make_preprocessor(x_train, scale_numeric=scale_numeric)),
                            ("model", clone(model)),
                        ]
                    )
                    pipe.fit(x_train, y_train)
                    y_score = pipe.predict_proba(x_test)[:, 1]
                    metrics = evaluate_predictions(y_test, y_score)
                    metrics.update(
                        {
                            "feature_set": feature_set,
                            "model": model_name,
                            "repeat": repeat_idx,
                            "repeat_seed": seed,
                            "fold": fold_idx,
                            "global_fold": (repeat_idx - 1) * 5 + fold_idx,
                            "n_features": x.shape[1],
                        }
                    )
                    rows.append(metrics)
    folds = pd.DataFrame(rows)
    metric_cols = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]
    summary = folds.groupby(["model", "feature_set", "n_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False).reset_index(drop=True)
    folds.to_csv(FOLD_RESULTS_PATH, index=False)
    summary.to_csv(SUMMARY_RESULTS_PATH, index=False)
    return folds, summary


def make_model_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in model_defs().keys():
        row = {"Model": model}
        for feature_set in ["Top20", "Top50", "Top100"]:
            sub = summary[(summary["model"] == model) & (summary["feature_set"] == feature_set)]
            if sub.empty:
                row[feature_set] = ""
            else:
                item = sub.iloc[0]
                row[feature_set] = f"{item['roc_auc_mean']:.3f}±{item['roc_auc_std']:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(MODEL_COMPARISON_PATH, index=False)
    return out


def best_model_analysis(summary: pd.DataFrame) -> dict:
    best = summary.iloc[0]
    analysis = {
        "best_model": best["model"],
        "best_feature_set": best["feature_set"],
        "best_n_features": int(best["n_features"]),
        "best_roc_auc_mean": float(best["roc_auc_mean"]),
        "best_roc_auc_std": float(best["roc_auc_std"]),
        "best_pr_auc_mean": float(best["pr_auc_mean"]),
        "best_f1_mean": float(best["f1_mean"]),
        "best_balanced_accuracy_mean": float(best["balanced_accuracy_mean"]),
    }
    BEST_ANALYSIS_PATH.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    return analysis


def compare_feature_sets_for_best(summary: pd.DataFrame, best_model: str) -> pd.DataFrame:
    sub = summary[summary["model"] == best_model].copy()
    sub["feature_set_order"] = sub["feature_set"].map({"Top20": 20, "Top50": 50, "Top100": 100})
    return sub.sort_values("feature_set_order")


def write_report(summary: pd.DataFrame, comparison: pd.DataFrame, analysis: dict) -> None:
    best_sub = compare_feature_sets_for_best(summary, analysis["best_model"])
    top10 = pd.read_csv(TOP_FILES[analysis["best_feature_set"]]).head(10)["feature"].tolist()
    aucs = {row.feature_set: row.roc_auc_mean for row in best_sub.itertuples(index=False)}
    gain_20_50 = aucs.get("Top50", np.nan) - aucs.get("Top20", np.nan)
    gain_50_100 = aucs.get("Top100", np.nan) - aucs.get("Top50", np.nan)
    meaningful = "yes" if pd.notna(gain_20_50) and gain_20_50 > 0.01 else "no"
    overfit_text = (
        "larger feature sets did not provide a meaningful ROC-AUC gain"
        if meaningful == "no"
        else "Top50/Top100 should be checked against external validation because added features may increase overfitting risk"
    )
    lines = [
        "Final Modeling Report",
        "",
        f"Dataset: {DATA_PATH}",
        "Validation: 10 repeated StratifiedKFold runs, 5 folds per repeat.",
        f"Repeat seeds: {REPEAT_SEEDS}",
        "",
        "Model Comparison (ROC-AUC mean +/- std):",
        comparison.to_string(index=False),
        "",
        "Best Model Analysis:",
        f"Best performing model: {analysis['best_model']}",
        f"Best feature set: {analysis['best_feature_set']} ({analysis['best_n_features']} features)",
        f"Best ROC-AUC: {analysis['best_roc_auc_mean']:.4f} +/- {analysis['best_roc_auc_std']:.4f}",
        f"Best PR-AUC: {analysis['best_pr_auc_mean']:.4f}",
        f"Best F1: {analysis['best_f1_mean']:.4f}",
        f"Best Balanced Accuracy: {analysis['best_balanced_accuracy_mean']:.4f}",
        "",
        "Top 10 Predictors From Best Feature Set:",
    ]
    lines.extend([f"{idx}. {feature}" for idx, feature in enumerate(top10, start=1)])
    lines.extend(
        [
            "",
            "Statistical Comparison Across Feature Set Sizes:",
            best_sub[["feature_set", "n_features", "roc_auc_mean", "roc_auc_std", "pr_auc_mean", "f1_mean", "balanced_accuracy_mean"]].to_string(index=False),
            "",
            f"Top20 to Top50 ROC-AUC gain: {gain_20_50:.4f}",
            f"Top50 to Top100 ROC-AUC gain: {gain_50_100:.4f}",
            f"Meaningful gain from additional features: {meaningful}",
            f"Overfitting interpretation: {overfit_text}.",
            "",
            "Final Recommendation:",
            "Prefer the smallest feature set whose ROC-AUC is within approximately 0.01 of the best result, unless a larger feature set gives a clear and reproducible gain. This balances predictive performance, interpretability, and overfitting control for an ADNI MCI-to-AD publication model.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target = target_column(df)
    log("====================================================")
    log("BUILDING MODELING DATASETS")
    log("====================================================")
    log(f"Using dataset: {DATA_PATH}")
    log(f"Subjects: {len(df)}")
    log(f"Target: {target}")
    log(df[target].value_counts().to_string())
    datasets = build_modeling_datasets(df, target)
    for name, data in datasets.items():
        log(f"{name}: {data.shape[0]} subjects, {data.shape[1] - 1} features -> {DATASET_OUTPUTS[name]}")

    log("\n====================================================")
    log("RUNNING 5x10 REPEATED STRATIFIED CV")
    log("====================================================")
    _, summary = run_cv(datasets, target)
    comparison = make_model_comparison(summary)
    analysis = best_model_analysis(summary)
    write_report(summary, comparison, analysis)

    log("\n====================================================")
    log("MODEL COMPARISON")
    log("====================================================")
    log(comparison.to_string(index=False))
    log("\nBest model:")
    log(json.dumps(analysis, indent=2))
    log(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
