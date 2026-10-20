"""Evaluate minimum-follow-up labels and fold-specific Cox feature ranking."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

import argparse
import json
import platform
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from validate_nested_sensitivity import CORRELATION_THRESHOLD, FALLBACK_DATA_PATH, MISSING_THRESHOLD, OUTER_FOLDS, OUTER_SEEDS, PROCESSED_DIR, TOP_K_VALUES, VARIANCE_THRESHOLD, bio_score, coerce_numeric_like, is_leakage_feature, months_between, one_hot_encoder, target_column
from model_survival import (
    TIME_HORIZONS,
    XGB_SURVIVAL_PARAMS,
    breslow_baseline_cumulative_hazard,
    harrell_c_index,
    integrated_brier_score_from_grid,
    ipcw_brier_score,
    survival_probability_from_risk,
    time_dependent_auc,
)


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
    category=UserWarning,
)


DATA_PATH = (
    PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
    if (PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv").exists()
    else FALLBACK_DATA_PATH
)
COHORT_PATH = (
    PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"
    if (PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv").exists()
    else PROCESSED_DIR / "2-strict_mci_to_ad_publication_cohort.csv"
)

OUT_DIR = PROCESSED_DIR / "robustness_min_followup_survival_coxrank"
ANYFUTURE_DIR = OUT_DIR / "anyfuture_min_followup"
SURVIVAL_DIR = OUT_DIR / "survival_aware_feature_selection"
SURVIVAL_RANKING_DIR = SURVIVAL_DIR / "survival_rankings"
SUMMARY_DIR = OUT_DIR / "summary"

RANDOM_STATE = 42
MIN_FOLLOWUP_MONTHS = 24
INNER_NOT_USED = "No nested CV; this script is a controlled sensitivity analysis."

CLASSIFICATION_MODELS = {
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


def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    for path in [ANYFUTURE_DIR, SURVIVAL_DIR, SURVIVAL_RANKING_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


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


class FoldSpecificMISelector(BaseEstimator, TransformerMixin):
    """Classification feature filtering and mutual-information ranking inside each training fold."""

    def __init__(self, top_k: int, random_state: int = RANDOM_STATE) -> None:
        self.top_k = top_k
        self.random_state = random_state

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> "FoldSpecificMISelector":
        x = fold_filter_features(pd.DataFrame(x).copy())
        self.filtered_features_ = x.columns.tolist()
        self.selected_features_ = rank_mi_features(x, y, self.top_k, self.random_state)
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        for col in self.selected_features_:
            if col not in x.columns:
                x[col] = np.nan
        return x[self.selected_features_].copy()


class DynamicPreprocessor(BaseEstimator, TransformerMixin):
    """Fit imputation, encoding, and optional scaling after feature selection."""

    def __init__(self, scale_numeric: bool = False) -> None:
        self.scale_numeric = scale_numeric

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> "DynamicPreprocessor":
        x = coerce_numeric_like(pd.DataFrame(x).copy())
        self.preprocessor_ = make_preprocessor(x, self.scale_numeric)
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


def fold_filter_features(x: pd.DataFrame) -> pd.DataFrame:
    """Apply unsupervised leakage, missingness, variance, and correlation filters."""
    x = coerce_numeric_like(x.copy())
    keep = [col for col in x.columns if not is_leakage_feature(col)]
    x = x[keep].copy()

    missing = x.isna().mean() * 100
    x = x[missing[missing <= MISSING_THRESHOLD].index.tolist()].copy()

    keep = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            vals = pd.to_numeric(x[col], errors="coerce")
            if vals.dropna().nunique() > 1 and vals.var(skipna=True) >= VARIANCE_THRESHOLD:
                keep.append(col)
        elif x[col].dropna().nunique() > 1:
            keep.append(col)
    x = x[keep].copy()

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
                if b in removed or corr.loc[a, b] <= CORRELATION_THRESHOLD:
                    continue
                score_a = (100 - missing_numeric[a], bio_score(a), -missing_numeric[a], a)
                score_b = (100 - missing_numeric[b], bio_score(b), -missing_numeric[b], b)
                drop = b if score_a >= score_b else a
                removed.add(drop)
                if drop == a:
                    break
        x = x.drop(columns=list(removed), errors="ignore")
    return x


def rank_mi_features(x: pd.DataFrame, y: np.ndarray, top_k: int, random_state: int) -> list[str]:
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
    scores = mutual_info_classif(xt, y, discrete_features=discrete, random_state=random_state)

    original_names = []
    for name in names:
        matched = name if name in x.columns else None
        if matched is None:
            for col in categorical:
                if name.startswith(f"{col}_"):
                    matched = col
                    break
        original_names.append(matched or name)
    ranking = pd.DataFrame({"feature": original_names, "mi_score": scores})
    ranking = ranking.groupby("feature", as_index=False)["mi_score"].sum()
    ranking["bio_score"] = ranking["feature"].map(bio_score)
    ranking = ranking.sort_values(["mi_score", "bio_score", "feature"], ascending=[False, False, True])
    return ranking.head(min(top_k, len(ranking)))["feature"].tolist()


def compute_classification_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else np.nan,
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


def classification_pipeline(model_name: str, top_k: int, seed: int) -> Pipeline:
    estimator, scale = CLASSIFICATION_MODELS[model_name]
    estimator = clone(estimator)
    if "random_state" in estimator.get_params():
        estimator.set_params(random_state=seed)
    return Pipeline(
        [
            ("feature_selector", FoldSpecificMISelector(top_k=top_k, random_state=seed)),
            ("preprocess", DynamicPreprocessor(scale_numeric=scale)),
            ("model", estimator),
        ]
    )


def load_feature_dataset() -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(DATA_PATH, low_memory=False)
    if "PTID" not in df.columns:
        raise ValueError("Feature dataset must contain PTID.")
    df["PTID"] = df["PTID"].astype(str)
    target = target_column(df)
    return df, target


def build_anyfuture_min_followup_cohort(feature_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = cohort["PTID"].astype(str)
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort["conversion_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype(int)
    cohort["min_followup_date"] = cohort["baseline_date"].apply(
        lambda x: x + pd.DateOffset(months=MIN_FOLLOWUP_MONTHS) if pd.notna(x) else pd.NaT
    )
    cohort["followup_duration_months"] = months_between(cohort["last_followup_date"], cohort["baseline_date"])
    cohort["is_converter"] = (
        (cohort["conversion_label"] == 1)
        & cohort["conversion_date"].notna()
        & (cohort["conversion_date"] > cohort["baseline_date"])
    )
    cohort["eligible_nonconverter"] = (
        (cohort["conversion_label"] == 0)
        & cohort["conversion_date"].isna()
        & cohort["last_followup_date"].notna()
        & (cohort["last_followup_date"] >= cohort["min_followup_date"])
    )
    cohort["include_anyfuture_24m"] = cohort["is_converter"] | cohort["eligible_nonconverter"]

    excluded = cohort.loc[~cohort["include_anyfuture_24m"]].copy()
    excluded["reason_for_exclusion"] = np.select(
        [
            (excluded["conversion_label"] == 1) & excluded["conversion_date"].isna(),
            excluded["conversion_label"] == 0,
        ],
        [
            "Observed converter but missing conversion date in date-audited sensitivity cohort",
            "No AD conversion and <24 months diagnostic follow-up",
        ],
        default="Unclear date or label information",
    )

    anyfuture = cohort.loc[cohort["include_anyfuture_24m"]].copy()
    anyfuture["target"] = anyfuture["is_converter"].astype(int)
    assert anyfuture["PTID"].is_unique
    assert not (anyfuture["conversion_date"] <= anyfuture["baseline_date"]).fillna(False).any()
    negative = anyfuture[anyfuture["target"] == 0]
    positive = anyfuture[anyfuture["target"] == 1]
    assert (negative["last_followup_date"] >= negative["min_followup_date"]).all()
    assert positive["conversion_date"].notna().all()

    original_summary = {
        "cohort": "Original Any-Future cohort",
        "minimum_followup_months_for_nonconverters": 0,
        "n_subjects": int(len(cohort)),
        "n_converters": int((cohort["conversion_label"] == 1).sum()),
        "n_nonconverters": int((cohort["conversion_label"] == 0).sum()),
        "n_excluded": 0,
    }
    corrected_summary = {
        "cohort": "Corrected Any-Future cohort with 24-month nonconverter follow-up",
        "minimum_followup_months_for_nonconverters": MIN_FOLLOWUP_MONTHS,
        "n_subjects": int(len(anyfuture)),
        "n_converters": int((anyfuture["target"] == 1).sum()),
        "n_nonconverters": int((anyfuture["target"] == 0).sum()),
        "n_excluded": int(len(excluded)),
    }
    summary = pd.DataFrame([original_summary, corrected_summary])

    audit_cols = [
        "PTID",
        "baseline_date",
        "conversion_date",
        "last_followup_date",
        "min_followup_date",
        "followup_duration_months",
        "conversion_label",
        "is_converter",
        "eligible_nonconverter",
        "include_anyfuture_24m",
    ]
    anyfuture[audit_cols + ["target"]].to_csv(ANYFUTURE_DIR / "anyfuture_min24_cohort.csv", index=False)
    excluded_cols = [
        "PTID",
        "baseline_date",
        "last_followup_date",
        "followup_duration_months",
        "conversion_label",
        "reason_for_exclusion",
    ]
    excluded[excluded_cols].to_csv(ANYFUTURE_DIR / "anyfuture_excluded_short_followup.csv", index=False)
    summary.to_csv(ANYFUTURE_DIR / "anyfuture_cohort_summary.csv", index=False)

    modeling = feature_df.drop(columns=[target], errors="ignore").merge(
        anyfuture[["PTID", "target"]], on="PTID", how="inner", validate="one_to_one"
    )
    modeling.to_csv(ANYFUTURE_DIR / "anyfuture_min24_modeling_dataset.csv", index=False)
    return modeling, excluded, summary


def run_anyfuture_models(modeling: pd.DataFrame, models: list[str] | None, max_outer_folds: int | None) -> pd.DataFrame:
    if models is None:
        models = list(CLASSIFICATION_MODELS)
    y = modeling["target"].astype(int).to_numpy()
    ptid = modeling["PTID"].astype(str)
    x = modeling[[col for col in modeling.columns if col not in {"PTID", "target"} and not is_leakage_feature(col)]].copy()
    x = coerce_numeric_like(x)
    fold_rows = []
    pred_rows = []
    selected_rows = []
    split_rows = []
    global_fold = 0
    for top_k in TOP_K_VALUES:
        for model_name in models:
            log(f"Any-Future min24 | {model_name} | Top-{top_k}")
            for seed in OUTER_SEEDS:
                cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
                for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                    global_fold += 1
                    if max_outer_folds is not None and global_fold > max_outer_folds:
                        break
                    assert set(ptid.iloc[train_idx]).isdisjoint(set(ptid.iloc[test_idx]))
                    fit_seed = seed * 100 + fold
                    pipe = classification_pipeline(model_name, top_k, fit_seed)
                    pipe.fit(x.iloc[train_idx], y[train_idx])
                    y_score = pipe.predict_proba(x.iloc[test_idx])[:, 1]
                    metrics = compute_classification_metrics(y[test_idx], y_score)
                    selected = pipe.named_steps["feature_selector"].selected_features_
                    base = {
                        "analysis": "anyfuture_min24",
                        "seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "top_k": top_k,
                        "n_selected_features": len(selected),
                    }
                    fold_rows.append({**base, **metrics})
                    for sid in sorted(ptid.iloc[train_idx]):
                        split_rows.append({**base, "PTID": sid, "split": "train"})
                    for sid, yt, ys in zip(ptid.iloc[test_idx], y[test_idx], y_score):
                        split_rows.append({**base, "PTID": sid, "split": "test"})
                        pred_rows.append({**base, "PTID": sid, "y_true": int(yt), "y_probability": float(ys)})
                    for rank, feature in enumerate(selected, start=1):
                        selected_rows.append({**base, "rank": rank, "feature": feature})
                if max_outer_folds is not None and global_fold >= max_outer_folds:
                    break
            global_fold = 0

    folds = pd.DataFrame(fold_rows)
    predictions = pd.DataFrame(pred_rows)
    selected_features = pd.DataFrame(selected_rows)
    splits = pd.DataFrame(split_rows)
    folds.to_csv(ANYFUTURE_DIR / "anyfuture_min24_fold_results.csv", index=False)
    predictions.to_csv(ANYFUTURE_DIR / "anyfuture_min24_predictions.csv", index=False)
    selected_features.to_csv(ANYFUTURE_DIR / "anyfuture_min24_selected_features.csv", index=False)
    splits.to_csv(ANYFUTURE_DIR / "anyfuture_min24_split_manifest.csv", index=False)
    metric_cols = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "recall", "f1"]
    summary = folds.groupby(["model", "top_k", "n_selected_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False)
    summary.to_csv(ANYFUTURE_DIR / "anyfuture_min24_model_summary.csv", index=False)
    return summary


def build_survival_dataset_no_write(feature_df: pd.DataFrame, target: str) -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = cohort["PTID"].astype(str)
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort.get("conversion_date"), errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype("Int64")
    if "exclude_from_time_to_conversion_analysis" in cohort.columns:
        exclude = cohort["exclude_from_time_to_conversion_analysis"].fillna(False).astype(bool)
    else:
        exclude = pd.Series(False, index=cohort.index)
    cohort["event"] = cohort["conversion_label"].astype(float)
    cohort.loc[exclude, "event"] = np.nan
    cohort["time_to_event_months"] = np.nan
    converted = (cohort["event"] == 1) & cohort["conversion_date"].notna()
    censored = cohort["event"] == 0
    cohort.loc[converted, "time_to_event_months"] = months_between(
        cohort.loc[converted, "conversion_date"], cohort.loc[converted, "baseline_date"]
    )
    cohort.loc[censored, "time_to_event_months"] = months_between(
        cohort.loc[censored, "last_followup_date"], cohort.loc[censored, "baseline_date"]
    )
    valid = cohort["event"].notna() & cohort["time_to_event_months"].notna()
    valid &= cohort["time_to_event_months"] > 0
    survival_core = cohort.loc[valid, ["PTID", "time_to_event_months", "event"]].copy()
    survival_core["event"] = survival_core["event"].astype(int)
    feature_cols = [
        col
        for col in feature_df.columns
        if col not in {"PTID", target, "label", "conversion_label", "converter_label", "converter"}
        and not is_leakage_feature(col)
    ]
    merged = survival_core.merge(feature_df[["PTID"] + feature_cols], on="PTID", how="left", validate="one_to_one")
    merged.to_csv(SURVIVAL_DIR / "adni_survival_dataset_coxrank_input.csv", index=False)
    return merged


def preprocess_survival_fold(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filtered_train = fold_filter_features(x_train)
    selected_cols = filtered_train.columns.tolist()
    train = x_train[selected_cols].copy()
    test = x_test[selected_cols].copy()
    preprocessor = make_preprocessor(train, scale_numeric=True)
    train_mat = preprocessor.fit_transform(train)
    test_mat = preprocessor.transform(test)
    names = preprocessor.get_feature_names_out().tolist()
    train_df = pd.DataFrame(train_mat, columns=names, index=x_train.index)
    test_df = pd.DataFrame(test_mat, columns=names, index=x_test.index)
    return train_df, test_df


def univariate_cox_score(x: np.ndarray, time: np.ndarray, event: np.ndarray) -> tuple[float, float, str]:
    x = np.asarray(x, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    if np.nanstd(x) < 1e-12 or len(np.unique(x[~np.isnan(x)])) <= 1:
        return np.nan, -np.inf, "constant"
    valid = np.isfinite(x) & np.isfinite(time) & np.isfinite(event)
    x = x[valid]
    time = time[valid]
    event = event[valid]
    if event.sum() < 2:
        return np.nan, -np.inf, "too_few_events"

    order = np.argsort(-time)
    x_ord = x[order]
    event_ord = event[order].astype(bool)

    def nll_grad_hess(beta_arr: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        beta = float(beta_arr[0])
        eta = np.clip(beta * x_ord, -30, 30)
        exp_eta = np.exp(eta)
        risk_sum = np.cumsum(exp_eta)
        x_risk_sum = np.cumsum(exp_eta * x_ord)
        x2_risk_sum = np.cumsum(exp_eta * x_ord * x_ord)
        mean_x = x_risk_sum[event_ord] / risk_sum[event_ord]
        mean_x2 = x2_risk_sum[event_ord] / risk_sum[event_ord]
        loglik = (eta[event_ord] - np.log(risk_sum[event_ord])).sum()
        grad = (x_ord[event_ord] - mean_x).sum()
        hess = -(mean_x2 - mean_x * mean_x).sum()
        return -float(loglik), np.array([-grad]), np.array([[-hess]])

    try:
        result = minimize(
            lambda b: nll_grad_hess(b)[0],
            x0=np.array([0.0]),
            jac=lambda b: nll_grad_hess(b)[1],
            method="BFGS",
            options={"maxiter": 100, "gtol": 1e-5},
        )
        beta = float(result.x[0])
        _, _, hess = nll_grad_hess(np.array([beta]))
        info = float(hess[0, 0])
        if not np.isfinite(info) or info <= 0:
            return beta, -np.inf, "failed_information"
        se = float(np.sqrt(1.0 / info))
        z = beta / se
        if not np.isfinite(z):
            return beta, -np.inf, "failed_z"
        return float(z), float(abs(z)), "ok"
    except Exception as exc:
        return np.nan, -np.inf, f"failed_{type(exc).__name__}"


def rank_features_univariate_cox(x_train: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> pd.DataFrame:
    rows = []
    for feature in x_train.columns:
        z, abs_z, status = univariate_cox_score(x_train[feature].to_numpy(), duration, event)
        rows.append({"feature": feature, "cox_z": z, "abs_cox_z": abs_z, "status": status})
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["abs_cox_z", "feature"], ascending=[False, True]).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def cox_neg_loglik_and_grad(beta: np.ndarray, x: np.ndarray, time: np.ndarray, event: np.ndarray, alpha: float) -> tuple[float, np.ndarray]:
    order = np.argsort(-time)
    x_ord = x[order]
    event_ord = event[order].astype(bool)
    eta = np.clip(x_ord @ beta, -30, 30)
    exp_eta = np.exp(eta)
    risk_sum = np.cumsum(exp_eta)
    weighted_x = np.cumsum(exp_eta[:, None] * x_ord, axis=0)
    event_xb = eta[event_ord].sum()
    event_log_risk = np.log(risk_sum[event_ord]).sum()
    loglik = event_xb - event_log_risk - 0.5 * alpha * np.sum(beta**2)
    means = weighted_x[event_ord] / risk_sum[event_ord, None]
    grad = x_ord[event_ord].sum(axis=0) - means.sum(axis=0) - alpha * beta
    return -float(loglik), -grad


def fit_cox_processed(x_train: pd.DataFrame, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    x_mat = x_train.to_numpy(float)
    beta0 = np.zeros(x_mat.shape[1])
    result = minimize(
        lambda b: cox_neg_loglik_and_grad(b, x_mat, time, event, alpha=0.1),
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300},
    )
    return result.x


def fit_xgb_survival_processed(x_train: pd.DataFrame, time: np.ndarray, event: np.ndarray, seed: int) -> XGBRegressor:
    params = dict(XGB_SURVIVAL_PARAMS)
    params["random_state"] = seed
    model = XGBRegressor(**params)
    y_xgb = np.where(event == 1, time, -time)
    model.fit(x_train, y_xgb)
    return model


def evaluate_survival_fold(
    model_name: str,
    risk: np.ndarray,
    train_risk: np.ndarray,
    t_train: np.ndarray,
    e_train: np.ndarray,
    t_test: np.ndarray,
    e_test: np.ndarray,
) -> dict[str, float]:
    td_aucs = {f"auc_{h}m": time_dependent_auc(t_test, e_test, risk, h) for h in TIME_HORIZONS}
    baseline_hazards = breslow_baseline_cumulative_hazard(t_train, e_train, train_risk, TIME_HORIZONS)
    ipcw_briers = {}
    for h in TIME_HORIZONS:
        survival_prob = survival_probability_from_risk(risk, baseline_hazards[h])
        ipcw_briers[f"ipcw_brier_{h}m"] = ipcw_brier_score(t_train, e_train, t_test, e_test, survival_prob, h)
    ibs = integrated_brier_score_from_grid({h: ipcw_briers[f"ipcw_brier_{h}m"] for h in TIME_HORIZONS})
    return {
        "model": model_name,
        "c_index": harrell_c_index(t_test, e_test, risk),
        "mean_time_dependent_auc": float(np.nanmean(list(td_aucs.values()))),
        "integrated_brier_score_ipcw": ibs,
        **td_aucs,
        **ipcw_briers,
    }


def run_survival_coxrank(survival_df: pd.DataFrame, max_outer_folds: int | None) -> pd.DataFrame:
    log("\nSurvival-aware feature selection using univariate Cox ranking")
    y_event = survival_df["event"].astype(int).to_numpy()
    duration = survival_df["time_to_event_months"].astype(float).to_numpy()
    ptid = survival_df["PTID"].astype(str)
    x = survival_df[
        [col for col in survival_df.columns if col not in {"PTID", "time_to_event_months", "event"} and not is_leakage_feature(col)]
    ].copy()
    x = coerce_numeric_like(x)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    prediction_rows = []
    selected_rows = []
    split_rows = []
    ranking_summary_rows = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y_event), start=1):
        if max_outer_folds is not None and fold > max_outer_folds:
            break
        assert set(ptid.iloc[train_idx]).isdisjoint(set(ptid.iloc[test_idx]))
        x_train_raw, x_test_raw = x.iloc[train_idx], x.iloc[test_idx]
        x_train, x_test = preprocess_survival_fold(x_train_raw, x_test_raw)
        t_train, t_test = duration[train_idx], duration[test_idx]
        e_train, e_test = y_event[train_idx], y_event[test_idx]
        ranking = rank_features_univariate_cox(x_train, t_train, e_train)
        ranking.to_csv(SURVIVAL_RANKING_DIR / f"fold_{fold:03d}_cox_ranking.csv", index=False)
        n_valid = int((ranking["status"] == "ok").sum())
        if n_valid < max(TOP_K_VALUES):
            raise RuntimeError(f"Fold {fold} has only {n_valid} valid Cox-ranked features.")

        for top_k in TOP_K_VALUES:
            selected = ranking.loc[ranking["status"] == "ok"].head(top_k)["feature"].tolist()
            if top_k == 50:
                assert set(ranking.loc[ranking["status"] == "ok"].head(20)["feature"]).issubset(set(selected))
            if top_k == 100:
                assert set(ranking.loc[ranking["status"] == "ok"].head(50)["feature"]).issubset(set(selected))
            xtr = x_train[selected]
            xte = x_test[selected]

            beta = fit_cox_processed(xtr, t_train, e_train)
            cox_train_risk = xtr.to_numpy(float) @ beta
            cox_risk = xte.to_numpy(float) @ beta
            cox_metrics = evaluate_survival_fold("CoxPH", cox_risk, cox_train_risk, t_train, e_train, t_test, e_test)

            xgb = fit_xgb_survival_processed(xtr, t_train, e_train, seed=RANDOM_STATE + fold)
            xgb_train_risk = xgb.predict(xtr)
            xgb_risk = xgb.predict(xte)
            xgb_metrics = evaluate_survival_fold(
                "XGBoostSurvival", xgb_risk, xgb_train_risk, t_train, e_train, t_test, e_test
            )

            for metrics, risk in [(cox_metrics, cox_risk), (xgb_metrics, xgb_risk)]:
                base = {
                    "fold": fold,
                    "top_k": top_k,
                    "n_valid_ranked_features": n_valid,
                    "n_selected_features": len(selected),
                    **metrics,
                }
                rows.append(base)
                for sid, tt, ee, rr in zip(ptid.iloc[test_idx], t_test, e_test, risk):
                    prediction_rows.append(
                        {
                            "fold": fold,
                            "top_k": top_k,
                            "model": metrics["model"],
                            "PTID": sid,
                            "time_to_event_months": float(tt),
                            "event": int(ee),
                            "risk_score": float(rr),
                        }
                    )
            for rank, feature in enumerate(selected, start=1):
                selected_rows.append({"fold": fold, "top_k": top_k, "rank": rank, "feature": feature})
        for sid in sorted(ptid.iloc[train_idx]):
            split_rows.append({"fold": fold, "PTID": sid, "split": "train"})
        for sid in sorted(ptid.iloc[test_idx]):
            split_rows.append({"fold": fold, "PTID": sid, "split": "test"})
        ranking_summary_rows.append(
            {
                "fold": fold,
                "n_processed_features": x_train.shape[1],
                "n_valid_cox_ranked_features": n_valid,
                "n_failed_or_constant_features": int(len(ranking) - n_valid),
            }
        )

    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(prediction_rows)
    selected = pd.DataFrame(selected_rows)
    splits = pd.DataFrame(split_rows)
    ranking_summary = pd.DataFrame(ranking_summary_rows)
    results.to_csv(SURVIVAL_DIR / "survival_coxrank_fold_results.csv", index=False)
    predictions.to_csv(SURVIVAL_DIR / "survival_coxrank_predictions.csv", index=False)
    selected.to_csv(SURVIVAL_DIR / "survival_coxrank_selected_features.csv", index=False)
    splits.to_csv(SURVIVAL_DIR / "survival_coxrank_split_manifest.csv", index=False)
    ranking_summary.to_csv(SURVIVAL_DIR / "survival_coxrank_ranking_audit.csv", index=False)

    metric_cols = [
        "c_index",
        "integrated_brier_score_ipcw",
        "mean_time_dependent_auc",
        *[f"auc_{h}m" for h in TIME_HORIZONS],
        *[f"ipcw_brier_{h}m" for h in TIME_HORIZONS],
    ]
    summary = results.groupby(["model", "top_k", "n_selected_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["c_index_mean", "mean_time_dependent_auc_mean"], ascending=False)
    summary.to_csv(SURVIVAL_DIR / "survival_coxrank_model_summary.csv", index=False)

    stability = []
    total_folds = selected["fold"].nunique()
    for feature, group in selected.groupby("feature"):
        row: dict[str, Any] = {"feature": feature, "mean_cox_rank": float(group["rank"].mean())}
        for top_k in TOP_K_VALUES:
            row[f"top{top_k}_frequency"] = group.loc[group["top_k"] == top_k, "fold"].nunique() / total_folds
        stability.append(row)
    stability_df = pd.DataFrame(stability)
    frequency_cols = [f"top{top_k}_frequency" for top_k in TOP_K_VALUES if f"top{top_k}_frequency" in stability_df.columns]
    sort_cols = frequency_cols + ["mean_cox_rank"]
    ascending = [False] * len(frequency_cols) + [True]
    stability_df.sort_values(sort_cols, ascending=ascending).to_csv(
        SURVIVAL_DIR / "survival_feature_stability.csv",
        index=False,
    )
    compare_old_new_survival(summary)
    return summary


def compare_old_new_survival(new_summary: pd.DataFrame) -> None:
    old_path = PROCESSED_DIR / "survival_analysis" / "results" / "survival_model_comparison.csv"
    if not old_path.exists():
        return
    old = pd.read_csv(old_path)
    old = old.rename(columns={"feature_set": "top_k_label"})
    old["top_k"] = old["top_k_label"].str.extract(r"(\d+)").astype(int)
    old["model"] = old["model"].replace({"XGBoostSurvival": "XGBoostSurvival", "CoxPH": "CoxPH"})
    rows = []
    for _, new_row in new_summary.iterrows():
        model = new_row["model"]
        top_k = int(new_row["top_k"])
        match = old[(old["model"] == model) & (old["top_k"] == top_k)]
        if match.empty:
            continue
        old_row = match.iloc[0]
        rows.append(
            {
                "model": model,
                "top_k": top_k,
                "old_c_index": old_row.get("c_index_mean", np.nan),
                "new_c_index": new_row.get("c_index_mean", np.nan),
                "delta_c_index": new_row.get("c_index_mean", np.nan) - old_row.get("c_index_mean", np.nan),
                "old_mean_time_dependent_auc": old_row.get("mean_time_dependent_auc_mean", np.nan),
                "new_mean_time_dependent_auc": new_row.get("mean_time_dependent_auc_mean", np.nan),
                "delta_mean_time_dependent_auc": new_row.get("mean_time_dependent_auc_mean", np.nan)
                - old_row.get("mean_time_dependent_auc_mean", np.nan),
                "old_ipcw_ibs": old_row.get("integrated_brier_score_ipcw_mean", np.nan),
                "new_ipcw_ibs": new_row.get("integrated_brier_score_ipcw_mean", np.nan),
                "delta_ipcw_ibs": new_row.get("integrated_brier_score_ipcw_mean", np.nan)
                - old_row.get("integrated_brier_score_ipcw_mean", np.nan),
            }
        )
    pd.DataFrame(rows).to_csv(SURVIVAL_DIR / "survival_feature_selection_comparison.csv", index=False)


def write_manifest(args: argparse.Namespace, any_summary: pd.DataFrame, survival_summary: pd.DataFrame) -> None:
    try:
        import sklearn
        import xgboost

        versions = {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        }
    except Exception as exc:
        versions = {"version_collection_error": str(exc)}
    manifest = {
        "analysis_name": "anyfuture_min24_and_survival_coxrank_robustness",
        "purpose": "Partial robustness additions without modifying prior code or outputs.",
        "input_dataset": str(DATA_PATH),
        "cohort_file": str(COHORT_PATH),
        "minimum_followup_months_for_anyfuture_nonconverters": MIN_FOLLOWUP_MONTHS,
        "anyfuture_models": args.models,
        "classification_top_k_values": TOP_K_VALUES,
        "survival_feature_selection": "fold-specific univariate Cox ranking by absolute Wald z statistic",
        "survival_models": ["CoxPH", "XGBoostSurvival"],
        "survival_top_k_values": TOP_K_VALUES,
        "survival_time_horizons": TIME_HORIZONS,
        "old_outputs_modified": False,
        "hyperparameter_policy": "fixed pre-specified parameters; no Optuna",
        "note": INNER_NOT_USED,
        "max_outer_folds_debug": args.max_outer_folds,
        "outputs": str(OUT_DIR),
        "package_versions": versions,
    }
    (SUMMARY_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lines = [
        "ROBUSTNESS: ANY-FUTURE MINIMUM FOLLOW-UP AND SURVIVAL COX-RANKING",
        "",
        "Previous analyses were not modified.",
        f"Outputs: {OUT_DIR}",
        "",
        "Any-Future corrected cohort summary:",
        any_summary.to_string(index=False) if not any_summary.empty else "Skipped.",
        "",
        "Survival-aware Cox-ranking model summary:",
        survival_summary.head(12).to_string(index=False) if not survival_summary.empty else "Skipped.",
    ]
    (SUMMARY_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial robustness pipeline: any-future min follow-up and survival Cox ranking.")
    parser.add_argument("--skip-anyfuture", action="store_true")
    parser.add_argument("--skip-survival", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CLASSIFICATION_MODELS),
        choices=list(CLASSIFICATION_MODELS),
        help="Classification models for corrected Any-Future sensitivity.",
    )
    parser.add_argument("--max-outer-folds", type=int, default=None, help="Debug only. Omit for the full run.")
    args = parser.parse_args()
    ensure_dirs()
    feature_df, target = load_feature_dataset()

    log("=" * 72)
    log("ROBUSTNESS: ANY-FUTURE MIN24 + SURVIVAL COX-RANK FEATURE SELECTION")
    log("=" * 72)
    log(f"Input dataset: {DATA_PATH}")
    log(f"Output directory: {OUT_DIR}")
    log("Previous code and output folders are not modified.")

    any_cohort_summary = pd.DataFrame()
    any_model_summary = pd.DataFrame()
    if not args.skip_anyfuture:
        anyfuture_modeling, _, any_cohort_summary = build_anyfuture_min_followup_cohort(feature_df, target)
        log("\nAny-Future cohort summary:")
        log(any_cohort_summary.to_string(index=False))
        any_model_summary = run_anyfuture_models(anyfuture_modeling, args.models, args.max_outer_folds)
        log("\nAny-Future min24 model summary:")
        log(any_model_summary.head(10).to_string(index=False))

    survival_summary = pd.DataFrame()
    if not args.skip_survival:
        survival_df = build_survival_dataset_no_write(feature_df, target)
        log(
            f"\nSurvival cohort: N={len(survival_df)}, events={int(survival_df['event'].sum())}, "
            f"censored={int((survival_df['event'] == 0).sum())}"
        )
        survival_summary = run_survival_coxrank(survival_df, args.max_outer_folds)
        log("\nSurvival Cox-rank summary:")
        log(survival_summary.head(10).to_string(index=False))

    write_manifest(args, any_cohort_summary, survival_summary)
    if not any_model_summary.empty:
        any_model_summary.to_csv(ANYFUTURE_DIR / "anyfuture_min24_model_summary_copy.csv", index=False)
    log(f"\nDone. New outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
