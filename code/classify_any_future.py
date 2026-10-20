"""Evaluate Any-Future classification with fold-specific filtering and MI."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import re
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
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


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)

PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
REQUESTED_DATA_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
FALLBACK_DATA_PATH = PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
DATA_PATH = REQUESTED_DATA_PATH if REQUESTED_DATA_PATH.exists() else FALLBACK_DATA_PATH

OUT_DIR = PROCESSED_DIR / "final_modeling_fold_specific_preprocessing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLD_RESULTS_PATH = OUT_DIR / "cv_fold_results.csv"
SUMMARY_RESULTS_PATH = OUT_DIR / "model_results_long.csv"
MODEL_COMPARISON_PATH = OUT_DIR / "model_comparison.csv"
SELECTED_FEATURES_PATH = OUT_DIR / "fold_selected_features.csv"
FILTER_AUDIT_PATH = OUT_DIR / "fold_filter_audit.csv"
BEST_ANALYSIS_PATH = OUT_DIR / "best_model_analysis.json"
REPORT_PATH = OUT_DIR / "fold_specific_preprocessing_report.txt"

RANDOM_STATE = 42
REPEAT_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
TOP_K_VALUES = [20, 50, 100]
MISSING_THRESHOLD = 40.0
VARIANCE_THRESHOLD = 0.001
CORRELATION_THRESHOLD = 0.95

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
    r"TIME_TO_EVENT",
    r"EVENT",
    r"VISCODE",
    r"VISIT",
    r"EXAMDATE",
    r"SCANDATE",
]

BIO_PRIORITY = [
    "CDRSB",
    "ADAS",
    "TOTSCORE",
    "TOTAL13",
    "MMSCORE",
    "MMSE",
    "FAQ",
    "MOCA",
    "NEUROBAT",
    "APOE",
    "GENOTYPE",
    "PTEDUCAT",
    "AGE",
    "HIPPOCAMPUS",
    "ENTORHINAL",
    "VENTRICLE",
    "CENTILOIDS",
    "SUMMARY_SUVR",
    "SUVR",
    "PTAU",
    "TAU",
    "ABETA",
    "BMI",
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


def is_leakage_feature(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def bio_score(feature: str) -> int:
    upper = feature.upper()
    return sum(1 for token in BIO_PRIORITY if token in upper)


def coerce_numeric_like(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = pd.to_numeric(out[col], errors="coerce")
            non_missing = out[col].notna().sum()
            if non_missing and numeric.notna().sum() / non_missing >= 0.8:
                out[col] = numeric
    return out


class FoldSpecificFeatureSelector(BaseEstimator, TransformerMixin):
    """Apply filtering and MI feature selection using only the training fold."""

    def __init__(
        self,
        top_k: int,
        missing_threshold: float = MISSING_THRESHOLD,
        variance_threshold: float = VARIANCE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.top_k = top_k
        self.missing_threshold = missing_threshold
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FoldSpecificFeatureSelector":
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        self.input_features_ = x.columns.tolist()

        keep = [col for col in x.columns if not is_leakage_feature(col)]
        self.removed_leakage_ = [col for col in x.columns if col not in keep]
        x = x[keep].copy()

        missing = x.isna().mean() * 100
        keep = missing[missing <= self.missing_threshold].index.tolist()
        self.removed_missing_ = missing[missing > self.missing_threshold].index.tolist()
        x = x[keep].copy()

        keep = []
        self.removed_variance_ = []
        for col in x.columns:
            if pd.api.types.is_numeric_dtype(x[col]):
                vals = pd.to_numeric(x[col], errors="coerce")
                var = vals.var(skipna=True)
                unique = vals.dropna().nunique()
                if unique <= 1 or (pd.notna(var) and var < self.variance_threshold):
                    self.removed_variance_.append(col)
                else:
                    keep.append(col)
            else:
                if x[col].dropna().nunique() <= 1:
                    self.removed_variance_.append(col)
                else:
                    keep.append(col)
        x = x[keep].copy()

        self.removed_correlation_ = []
        numeric = x.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric) >= 2:
            imputed = pd.DataFrame(
                SimpleImputer(strategy="median").fit_transform(x[numeric]),
                columns=numeric,
                index=x.index,
            )
            corr = imputed.corr().abs()
            missing_numeric = x[numeric].isna().mean() * 100
            removed = set()
            cols = list(corr.columns)
            for i, a in enumerate(cols):
                if a in removed:
                    continue
                for b in cols[i + 1 :]:
                    if b in removed or corr.loc[a, b] <= self.correlation_threshold:
                        continue
                    score_a = (100 - missing_numeric[a], bio_score(a), -missing_numeric[a])
                    score_b = (100 - missing_numeric[b], bio_score(b), -missing_numeric[b])
                    drop = b if score_a >= score_b else a
                    removed.add(drop)
                    self.removed_correlation_.append(drop)
                    if drop == a:
                        break
            x = x.drop(columns=list(removed), errors="ignore")

        self.filtered_features_ = x.columns.tolist()
        self.selected_features_ = self._mutual_information_select(x, y)
        return self

    def _mutual_information_select(self, x: pd.DataFrame, y: np.ndarray) -> list[str]:
        if x.empty:
            return []
        numeric = x.select_dtypes(include=[np.number]).columns.tolist()
        categorical = [col for col in x.columns if col not in numeric]
        preprocessor = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), numeric),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", one_hot_encoder()),
                        ]
                    ),
                    categorical,
                ),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        xt = preprocessor.fit_transform(x)
        names = preprocessor.get_feature_names_out().tolist()
        discrete = [name not in numeric for name in names]
        scores = mutual_info_classif(xt, y, discrete_features=discrete, random_state=self.random_state)

        original_names = []
        for name in names:
            matched = name if name in x.columns else None
            if matched is None:
                for col in categorical:
                    if name.startswith(f"{col}_"):
                        matched = col
                        break
            original_names.append(matched or name)

        ranking = (
            pd.DataFrame({"feature": original_names, "mi_score": scores})
            .groupby("feature", as_index=False)["mi_score"]
            .sum()
        )
        ranking["bio_score"] = ranking["feature"].map(bio_score)
        ranking = ranking.sort_values(["mi_score", "bio_score", "feature"], ascending=[False, False, True])
        return ranking.head(min(self.top_k, len(ranking)))["feature"].tolist()

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        for col in self.selected_features_:
            if col not in x.columns:
                x[col] = np.nan
        return x[self.selected_features_].copy()

    def audit_row(self) -> dict:
        return {
            "top_k": self.top_k,
            "n_input_features": len(getattr(self, "input_features_", [])),
            "n_after_leakage_removal": len(getattr(self, "input_features_", [])) - len(getattr(self, "removed_leakage_", [])),
            "n_removed_leakage": len(getattr(self, "removed_leakage_", [])),
            "n_removed_missingness": len(getattr(self, "removed_missing_", [])),
            "n_removed_low_variance": len(getattr(self, "removed_variance_", [])),
            "n_removed_high_correlation": len(getattr(self, "removed_correlation_", [])),
            "n_after_filters": len(getattr(self, "filtered_features_", [])),
            "n_selected_features": len(getattr(self, "selected_features_", [])),
            "selected_features": ";".join(getattr(self, "selected_features_", [])),
        }


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", one_hot_encoder()),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", Pipeline(categorical_steps), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """Build imputation, encoding, and scaling after fold-specific feature selection."""

    def __init__(self, scale_numeric: bool = False) -> None:
        self.scale_numeric = scale_numeric

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> "DynamicPreprocessor":
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        self.preprocessor_ = make_preprocessor(x, scale_numeric=self.scale_numeric)
        self.preprocessor_.fit(x, y)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        transformed = self.preprocessor_.transform(x)
        try:
            names = self.preprocessor_.get_feature_names_out()
            return pd.DataFrame(transformed, columns=names, index=x.index)
        except Exception:
            return transformed

    def get_feature_names_out(self) -> np.ndarray:
        return self.preprocessor_.get_feature_names_out()


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


def prepare_data() -> tuple[pd.DataFrame, np.ndarray, str]:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    target = target_column(df)
    y = df[target].astype(int).to_numpy()
    feature_cols = [col for col in df.columns if col != target and not is_leakage_feature(col)]
    x = coerce_numeric_like(df[feature_cols].copy())
    if "PTID" in df.columns and df["PTID"].duplicated().any():
        raise ValueError("Dataset contains duplicated PTID values.")
    return x, y, target


def run_cv(x: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    selected_rows = []
    audit_rows = []
    models = model_defs()

    for top_k in TOP_K_VALUES:
        feature_set_name = f"FoldSpecificTop{top_k}"
        for model_name, (model, scale_numeric) in models.items():
            log(f"Evaluating {model_name} | {feature_set_name}")
            for repeat_idx, seed in enumerate(REPEAT_SEEDS, start=1):
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
                for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                    x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    selector = FoldSpecificFeatureSelector(top_k=top_k, random_state=seed)
                    pipe = Pipeline(
                        [
                            ("feature_selector", selector),
                            ("preprocess", DynamicPreprocessor(scale_numeric=scale_numeric)),
                            ("model", clone(model)),
                        ]
                    )
                    pipe.fit(x_train, y_train)
                    y_score = pipe.predict_proba(x_test)[:, 1]
                    metrics = evaluate_predictions(y_test, y_score)
                    selected = pipe.named_steps["feature_selector"].selected_features_
                    metrics.update(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "repeat": repeat_idx,
                            "repeat_seed": seed,
                            "fold": fold_idx,
                            "global_fold": (repeat_idx - 1) * 5 + fold_idx,
                            "n_features": len(selected),
                        }
                    )
                    rows.append(metrics)
                    audit = pipe.named_steps["feature_selector"].audit_row()
                    audit.update(
                        {
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "repeat": repeat_idx,
                            "repeat_seed": seed,
                            "fold": fold_idx,
                            "global_fold": (repeat_idx - 1) * 5 + fold_idx,
                        }
                    )
                    audit_rows.append(audit)
                    for rank, feature in enumerate(selected, start=1):
                        selected_rows.append(
                            {
                                "feature_set": feature_set_name,
                                "model": model_name,
                                "repeat": repeat_idx,
                                "repeat_seed": seed,
                                "fold": fold_idx,
                                "global_fold": (repeat_idx - 1) * 5 + fold_idx,
                                "rank": rank,
                                "feature": feature,
                            }
                        )

    folds = pd.DataFrame(rows)
    selected_features = pd.DataFrame(selected_rows)
    filter_audit = pd.DataFrame(audit_rows)
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
    selected_features.to_csv(SELECTED_FEATURES_PATH, index=False)
    filter_audit.to_csv(FILTER_AUDIT_PATH, index=False)
    return folds, summary, selected_features, filter_audit


def make_model_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in model_defs().keys():
        row = {"Model": model}
        for top_k in TOP_K_VALUES:
            feature_set = f"FoldSpecificTop{top_k}"
            sub = summary[(summary["model"] == model) & (summary["feature_set"] == feature_set)]
            if sub.empty:
                row[f"Top{top_k}"] = ""
            else:
                item = sub.iloc[0]
                row[f"Top{top_k}"] = f"{item['roc_auc_mean']:.3f}+/-{item['roc_auc_std']:.3f}"
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(MODEL_COMPARISON_PATH, index=False)
    return out


def best_model_analysis(summary: pd.DataFrame) -> dict:
    best = summary.iloc[0]
    analysis = {
        "validation_type": "fold_specific_filtering_and_feature_selection",
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


def write_report(x: pd.DataFrame, y: np.ndarray, summary: pd.DataFrame, comparison: pd.DataFrame, analysis: dict) -> None:
    lines = [
        "Fold-Specific Preprocessing and Feature Selection Report",
        "",
        f"Dataset: {DATA_PATH}",
        f"Subjects: {len(y)}",
        f"Converters: {int((y == 1).sum())}",
        f"Stable MCI: {int((y == 0).sum())}",
        f"Candidate baseline features before fold-specific filtering: {x.shape[1]}",
        "",
        "Validation:",
        "10 repeated StratifiedKFold runs, 5 folds per repeat.",
        f"Repeat seeds: {REPEAT_SEEDS}",
        "",
        "Leakage-control statement:",
        "Outcome-related variables, follow-up information, post-baseline measurements, and subject identifiers were excluded before modeling. Missingness filtering, low-variance filtering, correlation filtering, mutual-information feature selection, imputation, encoding, scaling, and model fitting were performed inside each training fold and then applied unchanged to the corresponding validation fold.",
        "",
        "Model Comparison (ROC-AUC mean +/- std):",
        comparison.to_string(index=False),
        "",
        "Best Model Analysis:",
        json.dumps(analysis, indent=2),
        "",
        "Full Summary:",
        summary.to_string(index=False),
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    log("====================================================")
    log("FOLD-SPECIFIC PREPROCESSING MODELING")
    log("====================================================")
    x, y, target = prepare_data()
    log(f"Using dataset: {DATA_PATH}")
    log(f"Target: {target}")
    log(f"Subjects: {len(y)}")
    log(f"Converters: {int((y == 1).sum())}")
    log(f"Stable MCI: {int((y == 0).sum())}")
    log(f"Candidate features before fold-specific filtering: {x.shape[1]}")

    log("\nRunning fold-specific 5x10 repeated stratified CV...")
    _, summary, _, _ = run_cv(x, y)
    comparison = make_model_comparison(summary)
    analysis = best_model_analysis(summary)
    write_report(x, y, summary, comparison, analysis)

    log("\n====================================================")
    log("MODEL COMPARISON")
    log("====================================================")
    log(comparison.to_string(index=False))
    log("\nBest model:")
    log(json.dumps(analysis, indent=2))
    log(f"\nSaved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
