"""Run nested model selection, semantic restriction, and common-cohort checks."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import argparse
import gc
import json
import platform
import re
import sys
import warnings
from pathlib import Path
from typing import Any

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
DATA_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
FALLBACK_DATA_PATH = PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
COHORT_PATH = PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"

OUT_DIR = PROCESSED_DIR / "robustness_analysis"
NESTED_DIR = OUT_DIR / "nested_cv"
SEMANTIC_DIR = OUT_DIR / "semantic_leakage"
COMMON_DIR = OUT_DIR / "common_cohort"
SUMMARY_DIR = OUT_DIR / "summary"

RANDOM_STATE = 42
OUTER_SEEDS = list(range(42, 52))
OUTER_FOLDS = 5
INNER_FOLDS = 4
TOP_K_VALUES = [20, 50, 100]
MODEL_ORDER = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "LightGBM"]
HORIZONS = [24, 36, 60]

MISSING_THRESHOLD = 40.0
VARIANCE_THRESHOLD = 0.001
CORRELATION_THRESHOLD = 0.95
TIE_TOLERANCE = 1e-4

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

SEMANTIC_FAMILIES = {
    "CDR": "diagnosis-adjacent clinical dementia severity measure",
    "ADAS": "diagnosis-adjacent cognitive assessment measure",
    "FAQ": "diagnosis-adjacent functional assessment measure",
}

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


def ensure_dirs() -> None:
    for path in [NESTED_DIR, SEMANTIC_DIR, COMMON_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


def semantic_family(feature: str) -> str | None:
    upper = feature.upper()
    for family in SEMANTIC_FAMILIES:
        if upper == family or upper.startswith(f"{family}_") or f"_{family}_" in upper:
            return family
    return None


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


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days / 30.4375


class FoldSpecificFeatureSelector(BaseEstimator, TransformerMixin):
    """Filter and rank features using only the current training fold."""

    def __init__(
        self,
        top_k: int,
        semantic_exclusions: tuple[str, ...] = (),
        missing_threshold: float = MISSING_THRESHOLD,
        variance_threshold: float = VARIANCE_THRESHOLD,
        correlation_threshold: float = CORRELATION_THRESHOLD,
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.top_k = top_k
        self.semantic_exclusions = semantic_exclusions
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

        exclusions = set(self.semantic_exclusions)
        keep = [col for col in x.columns if col not in exclusions]
        self.removed_semantic_ = [col for col in x.columns if col not in keep]
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
                    score_a = (100 - missing_numeric[a], bio_score(a), -missing_numeric[a], a)
                    score_b = (100 - missing_numeric[b], bio_score(b), -missing_numeric[b], b)
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

    def audit_row(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "n_input_features": len(getattr(self, "input_features_", [])),
            "n_removed_leakage": len(getattr(self, "removed_leakage_", [])),
            "n_removed_semantic": len(getattr(self, "removed_semantic_", [])),
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
    """Fit imputation, encoding, and optional scaling after feature selection."""

    def __init__(self, scale_numeric: bool = False) -> None:
        self.scale_numeric = scale_numeric

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> "DynamicPreprocessor":
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        self.preprocessor_ = make_preprocessor(x, scale_numeric=self.scale_numeric)
        self.preprocessor_.fit(x, y)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame | np.ndarray:
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        transformed = self.preprocessor_.transform(x)
        try:
            names = self.preprocessor_.get_feature_names_out()
            return pd.DataFrame(transformed, columns=names, index=x.index)
        except Exception:
            return transformed


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
                n_jobs=1,
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
                n_jobs=1,
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
                thread_count=1,
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
                n_jobs=1,
                verbose=-1,
            ),
            False,
        ),
    }


def predict_probability(pipe: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(x)[:, 1]
    score = pipe.decision_function(x)
    return 1.0 / (1.0 + np.exp(-score))


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_prauc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": safe_auc(y_true, y_score),
        "pr_auc": safe_prauc(y_true, y_score),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_pipeline(model_name: str, top_k: int, seed: int, semantic_exclusions: tuple[str, ...]) -> Pipeline:
    models = model_defs()
    estimator, scale_numeric = models[model_name]
    estimator = clone(estimator)
    if hasattr(estimator, "set_params"):
        params = estimator.get_params()
        if "random_state" in params:
            estimator.set_params(random_state=seed)
    return Pipeline(
        [
            (
                "feature_selector",
                FoldSpecificFeatureSelector(
                    top_k=top_k,
                    semantic_exclusions=semantic_exclusions,
                    random_state=seed,
                ),
            ),
            ("preprocess", DynamicPreprocessor(scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )


def load_modeling_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series, str]:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    target = target_column(df)
    if "PTID" not in df.columns:
        raise ValueError("Modeling dataset must contain PTID for subject-level split tracking.")
    if df["PTID"].duplicated().any():
        dup = df.loc[df["PTID"].duplicated(), "PTID"].head().tolist()
        raise ValueError(f"Dataset contains duplicated PTID values, examples: {dup}")
    feature_cols = [col for col in df.columns if col != target and col != "PTID" and not is_leakage_feature(col)]
    x = coerce_numeric_like(df[feature_cols].copy())
    y = df[target].astype(int).to_numpy()
    ptid = df["PTID"].astype(str)
    return x, y, ptid, target


def build_semantic_exclusion_list(columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in columns:
        family = semantic_family(feature)
        if family:
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "exclusion_reason": SEMANTIC_FAMILIES[family],
                }
            )
    out = pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)
    out.to_csv(SEMANTIC_DIR / "excluded_semantic_features.csv", index=False)
    return out


def create_outer_split_manifest(ptid: pd.Series, y: np.ndarray, name: str, base_dir: Path) -> pd.DataFrame:
    rows = []
    for seed in OUTER_SEEDS:
        cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y), start=1):
            train_ids = set(ptid.iloc[train_idx])
            test_ids = set(ptid.iloc[test_idx])
            assert train_ids.isdisjoint(test_ids)
            rows.extend(
                {"PTID": sid, "seed": seed, "fold": fold, "split": "train", "split_system": name}
                for sid in sorted(train_ids)
            )
            rows.extend(
                {"PTID": sid, "seed": seed, "fold": fold, "split": "test", "split_system": name}
                for sid in sorted(test_ids)
            )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(base_dir / f"{name}_outer_split_manifest.csv", index=False)
    return manifest


def append_df(path: Path, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def completed_key_set(path: Path, keys: list[str]) -> set[tuple]:
    df = load_csv_or_empty(path)
    if df.empty or not set(keys).issubset(df.columns):
        return set()
    return set(map(tuple, df[keys].drop_duplicates().itertuples(index=False, name=None)))


def remove_rows_matching(path: Path, criteria: dict[str, Any]) -> None:
    df = load_csv_or_empty(path)
    if df.empty or not set(criteria).issubset(df.columns):
        return
    mask = pd.Series(True, index=df.index)
    for col, value in criteria.items():
        mask &= df[col] == value
    if mask.any():
        df.loc[~mask].to_csv(path, index=False)


def clear_partial_nested_fold(paths: dict[str, Path], analysis_type: str, condition: str, seed: int, outer_fold: int) -> None:
    criteria = {
        "analysis_type": analysis_type,
        "condition": condition,
        "seed": seed,
        "outer_fold": outer_fold,
    }
    for path in paths.values():
        remove_rows_matching(path, criteria)


def clear_partial_common_fold(paths: list[Path], horizon: int, seed: int, outer_fold: int) -> None:
    criteria = {
        "horizon_months": horizon,
        "seed": seed,
        "outer_fold": outer_fold,
    }
    for path in paths:
        remove_rows_matching(path, criteria)


def nested_output_prefix(condition: str) -> str:
    if condition == "FULL":
        return "full"
    if condition == "SEMANTIC_RESTRICTED":
        return "restricted"
    return sanitize_filename(condition.lower())


def sanitize_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def nested_output_paths(out_dir: Path, prefix: str = "") -> dict[str, Path]:
    pre = f"{prefix}_" if prefix else ""
    return {
        "outer_predictions": out_dir / f"{pre}outer_predictions.csv",
        "outer_fold_results": out_dir / f"{pre}outer_fold_results.csv",
        "selected_configs": out_dir / f"{pre}selected_configs.csv",
        "selected_features": out_dir / f"{pre}selected_features.csv",
        "filter_audit": out_dir / f"{pre}filter_audit.csv",
        "inner_cv_results": out_dir / f"{pre}inner_cv_results.csv",
        "inner_split_manifest": out_dir / f"{pre}inner_split_manifest.csv",
    }


def read_nested_outputs(out_dir: Path, prefix: str = "") -> dict[str, pd.DataFrame]:
    return {
        name: load_csv_or_empty(path)
        for name, path in nested_output_paths(out_dir, prefix).items()
    }


def inner_cv_select_config(
    x_outer_train: pd.DataFrame,
    y_outer_train: np.ndarray,
    ptid_outer_train: pd.Series,
    seed: int,
    outer_fold: int,
    semantic_exclusions: tuple[str, ...],
    condition: str,
    analysis_type: str,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = []
    split_rows = []
    inner_seed = seed * 100 + outer_fold
    cv = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=inner_seed)
    models = list(model_defs())

    for model_name in models:
        for top_k in TOP_K_VALUES:
            fold_scores = []
            for inner_fold, (tr_idx, val_idx) in enumerate(cv.split(x_outer_train, y_outer_train), start=1):
                train_ids = set(ptid_outer_train.iloc[tr_idx])
                val_ids = set(ptid_outer_train.iloc[val_idx])
                assert train_ids.isdisjoint(val_ids)
                if model_name == models[0] and top_k == TOP_K_VALUES[0]:
                    split_rows.extend(
                        {
                            "analysis_type": analysis_type,
                            "condition": condition,
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "inner_seed": inner_seed,
                            "inner_fold": inner_fold,
                            "PTID": sid,
                            "split": "train",
                        }
                        for sid in sorted(train_ids)
                    )
                    split_rows.extend(
                        {
                            "analysis_type": analysis_type,
                            "condition": condition,
                            "seed": seed,
                            "outer_fold": outer_fold,
                            "inner_seed": inner_seed,
                            "inner_fold": inner_fold,
                            "PTID": sid,
                            "split": "validation",
                        }
                        for sid in sorted(val_ids)
                    )
                pipe = build_pipeline(model_name, top_k, inner_seed + inner_fold, semantic_exclusions)
                pipe.fit(x_outer_train.iloc[tr_idx], y_outer_train[tr_idx])
                score = predict_probability(pipe, x_outer_train.iloc[val_idx])
                metrics = compute_metrics(y_outer_train[val_idx], score)
                fold_scores.append(metrics["roc_auc"])
                row = {
                    "analysis_type": analysis_type,
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "inner_seed": inner_seed,
                    "inner_fold": inner_fold,
                    "model": model_name,
                    "top_k": top_k,
                    **metrics,
                }
                rows.append(row)
                del pipe, score, metrics
                gc.collect()

            mean_auc = float(np.nanmean(fold_scores))
            rows.append(
                {
                    "analysis_type": analysis_type,
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "inner_seed": inner_seed,
                    "inner_fold": 0,
                    "model": model_name,
                    "top_k": top_k,
                    "roc_auc": mean_auc,
                    "is_inner_mean_row": True,
                }
            )

    inner_results = pd.DataFrame(rows)
    means = inner_results[inner_results.get("is_inner_mean_row", False) == True].copy()
    if means.empty:
        means = (
            inner_results.groupby(["model", "top_k"], as_index=False)["roc_auc"]
            .mean()
            .rename(columns={"roc_auc": "mean_inner_roc_auc"})
        )
    else:
        means = means.rename(columns={"roc_auc": "mean_inner_roc_auc"})
    means["model_order"] = means["model"].map({name: i for i, name in enumerate(MODEL_ORDER)})
    means = means.sort_values(
        ["mean_inner_roc_auc", "top_k", "model_order"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    best_auc = means.iloc[0]["mean_inner_roc_auc"]
    tied = means[np.abs(means["mean_inner_roc_auc"] - best_auc) < TIE_TOLERANCE].copy()
    tied = tied.sort_values(["top_k", "model_order"], ascending=[True, True])
    best = tied.iloc[0]
    selected = {
        "analysis_type": analysis_type,
        "condition": condition,
        "seed": seed,
        "outer_fold": outer_fold,
        "inner_seed": inner_seed,
        "selected_model": best["model"],
        "selected_top_k": int(best["top_k"]),
        "mean_inner_roc_auc": float(best["mean_inner_roc_auc"]),
        "tie_tolerance": TIE_TOLERANCE,
    }
    return selected, inner_results, pd.DataFrame(split_rows)


def run_nested_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    ptid: pd.Series,
    analysis_type: str,
    condition: str,
    semantic_exclusions: tuple[str, ...],
    out_dir: Path,
    max_outer_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    log(f"\nRunning nested CV | {analysis_type} | {condition}")
    outer_manifest = create_outer_split_manifest(ptid, y, f"{analysis_type}_{condition}", out_dir)
    prefix = nested_output_prefix(condition)
    paths = nested_output_paths(out_dir, prefix)
    completed = completed_key_set(paths["outer_fold_results"], ["analysis_type", "condition", "seed", "outer_fold"])
    if completed:
        log(f"  Resume mode: {len(completed)} completed outer folds already found for {analysis_type} | {condition}.")

    fold_counter = 0
    for seed in OUTER_SEEDS:
        cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for outer_fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
            fold_counter += 1
            if max_outer_folds is not None and fold_counter > max_outer_folds:
                break
            completed_key = (analysis_type, condition, seed, outer_fold)
            if completed_key in completed:
                log(f"  Skip completed outer seed {seed}, fold {outer_fold} | {condition}")
                continue
            clear_partial_nested_fold(paths, analysis_type, condition, seed, outer_fold)
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            ptid_train, ptid_test = ptid.iloc[train_idx], ptid.iloc[test_idx]
            assert set(ptid_train).isdisjoint(set(ptid_test))

            log(f"  Outer seed {seed}, fold {outer_fold}: inner model selection")
            selected, inner_result, inner_splits = inner_cv_select_config(
                x_train,
                y_train,
                ptid_train,
                seed,
                outer_fold,
                semantic_exclusions,
                condition,
                analysis_type,
            )
            append_df(paths["selected_configs"], pd.DataFrame([selected]))
            append_df(paths["inner_cv_results"], inner_result)
            append_df(paths["inner_split_manifest"], inner_splits)

            log(
                "  Refit outer model: "
                f"{selected['selected_model']} Top-{selected['selected_top_k']}"
            )
            pipe = build_pipeline(
                selected["selected_model"],
                selected["selected_top_k"],
                selected["inner_seed"],
                semantic_exclusions,
            )
            pipe.fit(x_train, y_train)
            y_score = predict_probability(pipe, x_test)
            metrics = compute_metrics(y_test, y_score)
            fold_row = {
                "analysis_type": analysis_type,
                "condition": condition,
                "seed": seed,
                "outer_fold": outer_fold,
                "selected_model": selected["selected_model"],
                "selected_top_k": selected["selected_top_k"],
                **metrics,
            }

            selected_outer_features = pipe.named_steps["feature_selector"].selected_features_
            audit = pipe.named_steps["feature_selector"].audit_row()
            audit.update(
                {
                    "analysis_type": analysis_type,
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "selected_model": selected["selected_model"],
                    "selected_top_k": selected["selected_top_k"],
                }
            )
            selected_feature_rows = [
                {
                    "analysis_type": analysis_type,
                    "condition": condition,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "selected_model": selected["selected_model"],
                    "selected_top_k": selected["selected_top_k"],
                    "rank": rank,
                    "feature": feature,
                }
                for rank, feature in enumerate(selected_outer_features, start=1)
            ]
            prediction_rows = [
                {
                    "PTID": sid,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "y_true": int(yt),
                    "y_probability": float(ys),
                    "selected_model": selected["selected_model"],
                    "selected_top_k": selected["selected_top_k"],
                    "analysis_type": analysis_type,
                    "condition": condition,
                }
                for sid, yt, ys in zip(ptid_test, y_test, y_score)
            ]
            append_df(paths["filter_audit"], pd.DataFrame([audit]))
            append_df(paths["selected_features"], pd.DataFrame(selected_feature_rows))
            append_df(paths["outer_predictions"], pd.DataFrame(prediction_rows))
            # Write fold result last. Its presence is the resume marker for a completed outer fold.
            append_df(paths["outer_fold_results"], pd.DataFrame([fold_row]))
            completed.add(completed_key)
            del pipe, y_score, x_train, x_test, y_train, y_test, ptid_train, ptid_test
            del inner_result, inner_splits, selected_feature_rows, prediction_rows
            gc.collect()
        if max_outer_folds is not None and fold_counter >= max_outer_folds:
            break

    outputs = read_nested_outputs(out_dir, prefix)
    outputs["outer_manifest"] = outer_manifest
    return outputs


def summarize_folds(folds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "recall", "f1"]
    summary = folds.groupby(group_cols)[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    return summary.sort_values([f"{group_cols[-1]}"] if False else ["roc_auc_mean"], ascending=False)


def selected_config_frequency(configs: pd.DataFrame) -> pd.DataFrame:
    if configs.empty:
        return pd.DataFrame()
    out = (
        configs.groupby(["condition", "selected_model", "selected_top_k"], as_index=False)
        .size()
        .rename(columns={"size": "selection_count"})
    )
    totals = configs.groupby("condition").size().to_dict()
    out["selection_percent"] = out.apply(
        lambda row: 100 * row["selection_count"] / totals[row["condition"]],
        axis=1,
    )
    return out.sort_values(["condition", "selection_count"], ascending=[True, False])


def feature_frequency(selected_features: pd.DataFrame, condition: str) -> pd.DataFrame:
    sub = selected_features[selected_features["condition"] == condition].copy()
    if sub.empty:
        return pd.DataFrame(columns=["feature", "selection_count", "selection_frequency"])
    n_folds = sub[["seed", "outer_fold"]].drop_duplicates().shape[0]
    out = (
        sub.groupby("feature", as_index=False)
        .agg(selection_count=("feature", "size"), mean_rank=("rank", "mean"))
        .sort_values(["selection_count", "mean_rank"], ascending=[False, True])
    )
    out["selection_frequency"] = out["selection_count"] / n_folds
    return out


def save_nested_outputs(outputs: dict[str, pd.DataFrame], out_dir: Path, prefix: str = "") -> None:
    pre = f"{prefix}_" if prefix else ""
    for name, df in outputs.items():
        df.to_csv(out_dir / f"{pre}{name}.csv", index=False)


def semantic_paired_outputs(full: dict[str, pd.DataFrame], restricted: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_pred = full["outer_predictions"].rename(
        columns={
            "y_probability": "full_probability",
            "selected_model": "full_selected_model",
            "selected_top_k": "full_top_k",
        }
    )
    restricted_pred = restricted["outer_predictions"].rename(
        columns={
            "y_probability": "restricted_probability",
            "selected_model": "restricted_selected_model",
            "selected_top_k": "restricted_top_k",
        }
    )
    keep_full = ["PTID", "seed", "outer_fold", "y_true", "full_probability", "full_selected_model", "full_top_k"]
    keep_restricted = ["PTID", "seed", "outer_fold", "y_true", "restricted_probability", "restricted_selected_model", "restricted_top_k"]
    paired = full_pred[keep_full].merge(
        restricted_pred[keep_restricted],
        on=["PTID", "seed", "outer_fold", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    rows = []
    for (seed, fold), group in paired.groupby(["seed", "outer_fold"]):
        y_true = group["y_true"].to_numpy()
        full_metrics = compute_metrics(y_true, group["full_probability"].to_numpy())
        restricted_metrics = compute_metrics(y_true, group["restricted_probability"].to_numpy())
        row = {"seed": seed, "outer_fold": fold}
        for metric in ["roc_auc", "pr_auc", "f1", "balanced_accuracy", "accuracy", "sensitivity", "specificity"]:
            row[f"{metric}_full"] = full_metrics[metric]
            row[f"{metric}_restricted"] = restricted_metrics[metric]
            row[f"delta_{metric}"] = restricted_metrics[metric] - full_metrics[metric]
        rows.append(row)
    fold_results = pd.DataFrame(rows)
    summary = fold_results[[c for c in fold_results.columns if c.startswith("delta_")]].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean_delta", "std_delta"]
    return paired, fold_results, summary


def build_common_cohort_labels(full_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = cohort["PTID"].astype(str)
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort["conversion_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype(int)
    cohort["months_to_conversion"] = pd.to_numeric(cohort["months_to_conversion"], errors="coerce")
    cohort["followup_months"] = months_between(cohort["last_followup_date"], cohort["baseline_date"])
    cohort["eligible_60m"] = (
        ((cohort["conversion_label"] == 1) & cohort["months_to_conversion"].notna())
        | ((cohort["conversion_label"] == 0) & (cohort["followup_months"] >= 60))
    )
    common = cohort[cohort["eligible_60m"]].copy()
    for horizon in HORIZONS:
        common[f"label_{horizon}m"] = np.where(
            (common["conversion_label"] == 1) & (common["months_to_conversion"] <= horizon),
            1,
            0,
        )
    members_cols = [
        "PTID",
        "baseline_date",
        "conversion_date",
        "last_followup_date",
        "months_to_conversion",
        "followup_months",
        "eligible_60m",
        "label_24m",
        "label_36m",
        "label_60m",
    ]
    members = common[members_cols].copy()
    dist_rows = []
    for horizon in HORIZONS:
        y = common[f"label_{horizon}m"].astype(int)
        dist_rows.append(
            {
                "horizon_months": horizon,
                "n_total": len(y),
                "positive": int((y == 1).sum()),
                "negative": int((y == 0).sum()),
                "prevalence": float((y == 1).mean()),
            }
        )
    distribution = pd.DataFrame(dist_rows)
    assert distribution["n_total"].nunique() == 1

    dataset = full_df.merge(common[["PTID", "label_24m", "label_36m", "label_60m"]], on="PTID", how="inner")
    assert dataset["PTID"].nunique() == len(dataset)
    assert set(dataset["PTID"]) == set(common["PTID"])
    members.to_csv(COMMON_DIR / "common_cohort_members.csv", index=False)
    distribution.to_csv(COMMON_DIR / "common_cohort_label_distribution.csv", index=False)
    return dataset, distribution


def run_common_horizon_nested(
    full_df: pd.DataFrame,
    max_outer_folds: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    common_dataset, distribution = build_common_cohort_labels(full_df)
    target = target_column(full_df)
    base_features = [
        col
        for col in full_df.columns
        if col not in {"PTID", target, "label_24m", "label_36m", "label_60m"} and not is_leakage_feature(col)
    ]
    all_predictions = []
    all_folds = []
    all_configs = []

    # Use one fixed participant-level outer split source for all horizons.
    y_for_splits = common_dataset["label_60m"].astype(int).to_numpy()
    ptid = common_dataset["PTID"].astype(str)
    x = coerce_numeric_like(common_dataset[base_features].copy())
    create_outer_split_manifest(ptid, y_for_splits, "common_cohort_shared", COMMON_DIR)

    for horizon in HORIZONS:
        y = common_dataset[f"label_{horizon}m"].astype(int).to_numpy()
        log(f"\nCommon cohort nested CV | {horizon} months")
        pred_path = COMMON_DIR / f"common_{horizon}m_predictions.csv"
        fold_path = COMMON_DIR / f"common_{horizon}m_outer_fold_results.csv"
        config_path = COMMON_DIR / f"common_{horizon}m_selected_configs.csv"
        feature_path = COMMON_DIR / f"common_{horizon}m_selected_features.csv"
        inner_path = COMMON_DIR / f"common_{horizon}m_inner_cv_results.csv"
        inner_split_path = COMMON_DIR / f"common_{horizon}m_inner_split_manifest.csv"
        completed = completed_key_set(fold_path, ["horizon_months", "seed", "outer_fold"])
        if completed:
            log(f"  Resume mode: {len(completed)} completed common-cohort folds found for {horizon} months.")
        fold_counter = 0
        for seed in OUTER_SEEDS:
            cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
            for outer_fold, (train_idx, test_idx) in enumerate(cv.split(x, y_for_splits), start=1):
                fold_counter += 1
                if max_outer_folds is not None and fold_counter > max_outer_folds:
                    break
                completed_key = (horizon, seed, outer_fold)
                if completed_key in completed:
                    log(f"  Skip completed common {horizon}m seed {seed}, fold {outer_fold}")
                    continue
                clear_partial_common_fold(
                    [pred_path, fold_path, config_path, feature_path, inner_path, inner_split_path],
                    horizon,
                    seed,
                    outer_fold,
                )
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                ptid_train, ptid_test = ptid.iloc[train_idx], ptid.iloc[test_idx]
                selected, inner_result, inner_splits = inner_cv_select_config(
                    x_train,
                    y_train,
                    ptid_train,
                    seed,
                    outer_fold,
                    semantic_exclusions=(),
                    condition="FULL",
                    analysis_type=f"common_{horizon}m",
                )
                pipe = build_pipeline(selected["selected_model"], selected["selected_top_k"], selected["inner_seed"], ())
                pipe.fit(x_train, y_train)
                y_score = predict_probability(pipe, x_test)
                metrics = compute_metrics(y_test, y_score)
                selected["horizon_months"] = horizon
                inner_result = inner_result.copy()
                inner_result["horizon_months"] = horizon
                inner_splits = inner_splits.copy()
                inner_splits["horizon_months"] = horizon
                append_df(config_path, pd.DataFrame([selected]))
                append_df(inner_path, inner_result)
                append_df(inner_split_path, inner_splits)
                fold_row = {
                    "horizon_months": horizon,
                    "seed": seed,
                    "outer_fold": outer_fold,
                    "selected_model": selected["selected_model"],
                    "selected_top_k": selected["selected_top_k"],
                    **metrics,
                }
                selected_outer_features = pipe.named_steps["feature_selector"].selected_features_
                selected_feature_rows = [
                    {
                        "horizon_months": horizon,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "selected_model": selected["selected_model"],
                        "selected_top_k": selected["selected_top_k"],
                        "rank": rank,
                        "feature": feature,
                    }
                    for rank, feature in enumerate(selected_outer_features, start=1)
                ]
                prediction_rows = [
                    {
                        "PTID": sid,
                        "horizon_months": horizon,
                        "seed": seed,
                        "outer_fold": outer_fold,
                        "y_true": int(yt),
                        "y_probability": float(ys),
                        "selected_model": selected["selected_model"],
                        "selected_top_k": selected["selected_top_k"],
                    }
                    for sid, yt, ys in zip(ptid_test, y_test, y_score)
                ]
                append_df(feature_path, pd.DataFrame(selected_feature_rows))
                append_df(pred_path, pd.DataFrame(prediction_rows))
                # Write fold result last. Its presence is the resume marker for a completed common-cohort fold.
                append_df(fold_path, pd.DataFrame([fold_row]))
                completed.add(completed_key)
                del pipe, y_score, x_train, x_test, y_train, y_test, ptid_train, ptid_test
                del inner_result, inner_splits, selected_feature_rows, prediction_rows
                gc.collect()
            if max_outer_folds is not None and fold_counter >= max_outer_folds:
                break
        pred_df = load_csv_or_empty(pred_path)
        fold_df = load_csv_or_empty(fold_path)
        config_df = load_csv_or_empty(config_path)
        all_predictions.append(pred_df)
        all_folds.append(fold_df)
        all_configs.append(config_df)

    folds_all = pd.concat(all_folds, ignore_index=True)
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    configs_all = pd.concat(all_configs, ignore_index=True)
    folds_all.to_csv(COMMON_DIR / "common_horizon_metrics.csv", index=False)
    predictions_all.to_csv(COMMON_DIR / "common_horizon_predictions.csv", index=False)
    configs_all.to_csv(COMMON_DIR / "common_horizon_selected_configs.csv", index=False)

    summary = summarize_folds(folds_all, ["horizon_months"])
    summary = distribution.merge(summary, on="horizon_months", how="left")
    best_configs = (
        configs_all.groupby(["horizon_months", "selected_model", "selected_top_k"], as_index=False)
        .size()
        .sort_values(["horizon_months", "size"], ascending=[True, False])
    )
    summary.to_csv(COMMON_DIR / "common_horizon_comparison.csv", index=False)
    best_configs.to_csv(COMMON_DIR / "common_horizon_selected_config_frequency.csv", index=False)
    return summary, folds_all, predictions_all


def save_semantic_branch(x: pd.DataFrame, y: np.ndarray, ptid: pd.Series, max_outer_folds: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = build_semantic_exclusion_list(x.columns.tolist())
    semantic_exclusions = tuple(excluded["feature"].tolist())
    full_outputs = run_nested_cv(x, y, ptid, "any_future", "FULL", (), SEMANTIC_DIR, max_outer_folds=max_outer_folds)
    restricted_outputs = run_nested_cv(
        x,
        y,
        ptid,
        "any_future",
        "SEMANTIC_RESTRICTED",
        semantic_exclusions,
        SEMANTIC_DIR,
        max_outer_folds=max_outer_folds,
    )
    save_nested_outputs(full_outputs, SEMANTIC_DIR, "full")
    save_nested_outputs(restricted_outputs, SEMANTIC_DIR, "restricted")
    full_outputs["outer_predictions"].to_csv(SEMANTIC_DIR / "predictions_full.csv", index=False)
    restricted_outputs["outer_predictions"].to_csv(SEMANTIC_DIR / "predictions_restricted.csv", index=False)

    paired, fold_results, sensitivity_summary = semantic_paired_outputs(full_outputs, restricted_outputs)
    paired.to_csv(SEMANTIC_DIR / "semantic_paired_predictions.csv", index=False)
    fold_results.to_csv(SEMANTIC_DIR / "semantic_sensitivity_fold_results.csv", index=False)
    sensitivity_summary.to_csv(SEMANTIC_DIR / "semantic_sensitivity_summary.csv", index=False)

    freq_full = feature_frequency(full_outputs["selected_features"], "FULL")
    freq_restricted = feature_frequency(restricted_outputs["selected_features"], "SEMANTIC_RESTRICTED")
    freq_full.to_csv(SEMANTIC_DIR / "feature_frequency_full.csv", index=False)
    freq_restricted.to_csv(SEMANTIC_DIR / "feature_frequency_restricted.csv", index=False)

    # Also write the FULL any-future nested output under nested_cv as the primary nested validation branch.
    save_nested_outputs(full_outputs, NESTED_DIR)
    selected_config_frequency(full_outputs["selected_configs"]).to_csv(
        NESTED_DIR / "selected_config_frequency.csv",
        index=False,
    )
    return sensitivity_summary, summarize_folds(full_outputs["outer_fold_results"], ["analysis_type", "condition"])


def write_summary_readme(
    nested_summary: pd.DataFrame,
    semantic_summary: pd.DataFrame,
    common_summary: pd.DataFrame,
    x: pd.DataFrame,
    y: np.ndarray,
) -> None:
    lines = [
        "ROBUSTNESS ANALYSIS README",
        "",
        "This folder contains new validation and sensitivity outputs only.",
        "Existing primary-analysis code and outputs were not modified.",
        "",
        "Main analyses added:",
        "1. Nested CV using frozen outer seed/fold logic.",
        "2. Semantic leakage sensitivity excluding CDR, ADAS, and FAQ feature families.",
        "3. Common 60-month ascertainable cohort horizon analysis.",
        "",
        f"Input subjects: {len(y)}",
        f"Converters: {int((y == 1).sum())}",
        f"Stable MCI: {int((y == 0).sum())}",
        f"Candidate features before fold-specific filtering: {x.shape[1]}",
        "",
        "Nested FULL any-future summary:",
        nested_summary.to_string(index=False) if not nested_summary.empty else "Not available.",
        "",
        "Semantic sensitivity delta summary:",
        semantic_summary.to_string(index=False) if not semantic_summary.empty else "Not available.",
        "",
        "Common-cohort horizon summary:",
        common_summary.to_string(index=False) if not common_summary.empty else "Not available.",
        "",
        "Interpretation notes:",
        "- Nested-CV results separate model/configuration selection from outer-fold performance estimation.",
        "- Semantic-restricted results estimate how much performance depends on diagnosis-adjacent CDR/ADAS/FAQ families.",
        "- Common-cohort horizon results test horizon effects while holding subjects constant.",
    ]
    (SUMMARY_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def write_manifest(args: argparse.Namespace, x: pd.DataFrame, y: np.ndarray, target: str) -> None:
    try:
        import sklearn
        import xgboost
        import lightgbm
        import catboost

        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        }
    except Exception as exc:
        versions = {"version_collection_error": str(exc)}

    manifest = {
        "analysis_name": "robustness_analysis_nested_semantic_common_cohort",
        "input_dataset": str(DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH),
        "cohort_file": str(COHORT_PATH),
        "target": target,
        "subjects": int(len(y)),
        "converters": int((y == 1).sum()),
        "stable_mci": int((y == 0).sum()),
        "candidate_features_before_fold_specific_filtering": int(x.shape[1]),
        "outer_split_source": "recreated_from_frozen_StratifedKFold_seeds_42_to_51_no_saved_PTID_manifest_found",
        "outer_seeds": OUTER_SEEDS,
        "outer_folds": OUTER_FOLDS,
        "inner_folds": INNER_FOLDS,
        "inner_seed_rule": "inner_seed = outer_seed * 100 + outer_fold",
        "inner_selection_metric": "ROC-AUC",
        "tie_breaking": [
            "highest mean inner ROC-AUC",
            f"if absolute difference < {TIE_TOLERANCE}, choose smaller Top-k",
            "if still tied, choose fixed model order",
        ],
        "top_k_options": TOP_K_VALUES,
        "models": MODEL_ORDER,
        "hyperparameter_policy": "fixed pre-specified hyperparameters; no Optuna in this robustness script",
        "resume_policy": [
            "Outer-fold outputs are appended immediately after each completed fold.",
            "A fold is considered complete only when its outer_fold_results row exists.",
            "Reruns skip completed analysis_type/condition/seed/outer_fold combinations.",
            "If a crash occurs mid-fold, partial rows for that unfinished fold are cleared before rerun.",
        ],
        "feature_conditions": ["FULL", "SEMANTIC_RESTRICTED"],
        "semantic_restricted_families": SEMANTIC_FAMILIES,
        "missing_threshold_percent": MISSING_THRESHOLD,
        "variance_threshold": VARIANCE_THRESHOLD,
        "correlation_threshold": CORRELATION_THRESHOLD,
        "common_horizons_months": HORIZONS,
        "max_outer_folds_debug": args.max_outer_folds,
        "outputs": str(OUT_DIR),
        "package_versions": versions,
    }
    (SUMMARY_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")




def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nested-CV, semantic-leakage, and common-cohort robustness analyses."
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Skip nested semantic FULL vs SEMANTIC_RESTRICTED analysis.",
    )
    parser.add_argument(
        "--skip-common",
        action="store_true",
        help="Skip common-cohort horizon nested analysis.",
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=None,
        help="Debug option. Limit number of outer folds per branch; omit for full analysis.",
    )
    args = parser.parse_args()

    ensure_dirs()
    x, y, ptid, target = load_modeling_data()
    full_df = pd.read_csv(DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH, low_memory=False)
    full_df["PTID"] = full_df["PTID"].astype(str)

    log("=" * 72)
    log("ROBUSTNESS ANALYSIS: NESTED CV + SEMANTIC LEAKAGE + COMMON COHORT")
    log("=" * 72)
    log(f"Subjects: {len(y)}")
    log(f"Converters: {int((y == 1).sum())}")
    log(f"Stable MCI: {int((y == 0).sum())}")
    log(f"Candidate features: {x.shape[1]}")
    log(f"Output directory: {OUT_DIR}")

    semantic_summary = pd.DataFrame()
    nested_summary = pd.DataFrame()
    if not args.skip_semantic:
        semantic_summary, nested_summary = save_semantic_branch(
            x,
            y,
            ptid,
            max_outer_folds=args.max_outer_folds,
        )

    common_summary = pd.DataFrame()
    if not args.skip_common:
        common_summary, _, _ = run_common_horizon_nested(full_df, max_outer_folds=args.max_outer_folds)

    write_manifest(args, x, y, target)
    write_summary_readme(nested_summary, semantic_summary, common_summary, x, y)

    log("\nFinished robustness analysis.")
    log(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
