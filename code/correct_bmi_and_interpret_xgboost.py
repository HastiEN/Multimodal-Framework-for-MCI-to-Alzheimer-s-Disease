"""Repair legacy BMI units and evaluate the corresponding interpretation."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json

import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
RAW_DIR = PROJECT_DIR / "data" / "raw" / "ALL"

FINAL_DATASET_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed.csv"
CORRECTED_DATASET_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed_bmi_corrected.csv"
LABEL_PATH = PROCESSED_DIR / "mci_to_ad_labels_from_all_csv_scan.csv"
VITALS_PATH = RAW_DIR / "All_Subjects_VITALS_08Jun2026.csv"
OPTUNA_REPORT_JSON = PROCESSED_DIR / "optuna_xgb_report.json"
OLD_SHAP_PATH = PROCESSED_DIR / "shap_interpretation" / "top20_shap_importance.csv"

BMI_AUDIT_PATH = PROCESSED_DIR / "bmi_audit_report.csv"
BMI_TRACE_PATH = PROCESSED_DIR / "bmi_correction_subject_trace.csv"
BMI_MODEL_RESULTS_PATH = PROCESSED_DIR / "bmi_corrected_xgboost_results.csv"
BMI_SHAP_RANKING_PATH = PROCESSED_DIR / "bmi_corrected_shap_ranking.csv"

RANDOM_STATE = 42
MAX_POST_BASELINE_DAYS = 180
MAX_PRE_BASELINE_DAYS = 365


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_visit(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def audit_bmi(series: pd.Series, label: str) -> dict:
    return {
        "stage": label,
        "min": series.min(skipna=True),
        "max": series.max(skipna=True),
        "median": series.median(skipna=True),
        "mean": series.mean(skipna=True),
        "n_bmi_lt_10": int((series < 10).sum()),
        "n_bmi_gt_70": int((series > 70).sum()),
        "n_missing_bmi": int(series.isna().sum()),
    }


def prepare_vitals() -> pd.DataFrame:
    vitals = pd.read_csv(VITALS_PATH, low_memory=False)
    vitals["_PTID_KEY"] = normalize_id(vitals["PTID"])
    vitals["_source_date"] = pd.to_datetime(vitals["VISDATE"], errors="coerce")
    vitals["_viscode_norm"] = vitals["VISCODE"].apply(normalize_visit) if "VISCODE" in vitals.columns else ""
    vitals["_viscode2_norm"] = vitals["VISCODE2"].apply(normalize_visit) if "VISCODE2" in vitals.columns else ""
    return vitals


def recalculate_bmi(row: pd.Series) -> float:
    weight = pd.to_numeric(row.get("VSWEIGHT"), errors="coerce")
    height = pd.to_numeric(row.get("VSHEIGHT"), errors="coerce")
    wt_unit = pd.to_numeric(row.get("VSWTUNIT"), errors="coerce")
    ht_unit = pd.to_numeric(row.get("VSHTUNIT"), errors="coerce")

    if pd.isna(weight) or pd.isna(height) or pd.isna(wt_unit) or pd.isna(ht_unit):
        return np.nan
    if wt_unit == 1:
        weight_kg = weight * 0.453592
    elif wt_unit == 2:
        weight_kg = weight
    else:
        return np.nan

    if ht_unit == 1:
        height_m = height * 0.0254
    elif ht_unit == 2:
        height_m = height / 100.0
    else:
        return np.nan

    if weight_kg <= 0 or height_m <= 0:
        return np.nan
    bmi = weight_kg / (height_m**2)
    if bmi < 10 or bmi > 70:
        return np.nan
    return float(bmi)


def select_safe_vitals_row(subject: pd.Series, vitals: pd.DataFrame) -> dict:
    ptid_key = str(subject["PTID"]).strip()
    baseline_date = subject["baseline_date"]
    baseline_visit = normalize_visit(subject["baseline_visit"])
    candidates = vitals[vitals["_PTID_KEY"] == ptid_key].copy()
    if candidates.empty:
        return {"PTID": subject["PTID"], "corrected_bmi": np.nan, "status": "NO_VITALS_ROW"}

    candidates["_days_from_baseline"] = (candidates["_source_date"] - baseline_date).dt.days
    candidates["_abs_days_from_baseline"] = candidates["_days_from_baseline"].abs()

    # Never use records after baseline + 180 days.
    candidates = candidates[
        candidates["_source_date"].notna()
        & (candidates["_days_from_baseline"] <= MAX_POST_BASELINE_DAYS)
    ].copy()
    if candidates.empty:
        return {"PTID": subject["PTID"], "corrected_bmi": np.nan, "status": "NO_TEMPORALLY_SAFE_ROW"}

    exact_baseline = candidates["_viscode2_norm"].eq(baseline_visit) | candidates["_viscode_norm"].eq(baseline_visit)
    bl_visit = candidates["_viscode2_norm"].eq("bl") | candidates["_viscode_norm"].eq("bl")
    near_180 = candidates["_days_from_baseline"].between(-MAX_POST_BASELINE_DAYS, MAX_POST_BASELINE_DAYS, inclusive="both")
    pre_365 = candidates["_days_from_baseline"].between(-MAX_PRE_BASELINE_DAYS, 0, inclusive="both")

    candidates["_priority"] = np.select(
        [exact_baseline, bl_visit, near_180, pre_365],
        [1, 2, 3, 4],
        default=99,
    )
    candidates = candidates[candidates["_priority"] < 99].copy()
    if candidates.empty:
        return {"PTID": subject["PTID"], "corrected_bmi": np.nan, "status": "NO_BASELINE_COMPATIBLE_ROW"}

    candidates["corrected_bmi"] = candidates.apply(recalculate_bmi, axis=1)
    candidates = candidates.sort_values(["_priority", "_abs_days_from_baseline", "_days_from_baseline"])
    selected = candidates.iloc[0]
    return {
        "PTID": subject["PTID"],
        "corrected_bmi": selected["corrected_bmi"],
        "source_date": selected["_source_date"],
        "baseline_date": baseline_date,
        "days_from_baseline": selected["_days_from_baseline"],
        "VISCODE": selected.get("VISCODE", pd.NA),
        "VISCODE2": selected.get("VISCODE2", pd.NA),
        "VSWEIGHT": selected.get("VSWEIGHT", np.nan),
        "VSWTUNIT": selected.get("VSWTUNIT", np.nan),
        "VSHEIGHT": selected.get("VSHEIGHT", np.nan),
        "VSHTUNIT": selected.get("VSHTUNIT", np.nan),
        "status": "SELECTED",
    }


def load_best_params() -> dict:
    report = json.loads(OPTUNA_REPORT_JSON.read_text(encoding="utf-8"))
    params = dict(report["best_params"])
    params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "n_jobs": -1})
    return params


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(x: pd.DataFrame, params: dict) -> Pipeline:
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
    return Pipeline([("preprocess", preprocessor), ("model", XGBClassifier(**params))])


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_pred = (y_score >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def train_evaluate_and_shap(dataset: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    y = dataset["conversion_label"].astype(int)
    x = dataset.drop(columns=["conversion_label"])
    params = load_best_params()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipe = make_pipeline(x_train, params)
    pipe.fit(x_train, y_train)
    y_score = pipe.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_score)

    final_pipe = make_pipeline(x, params)
    final_pipe.fit(x, y)
    x_transformed = final_pipe.named_steps["preprocess"].transform(x)
    names = final_pipe.named_steps["preprocess"].get_feature_names_out().tolist()
    x_transformed_df = pd.DataFrame(x_transformed, columns=names, index=x.index)
    explainer = shap.TreeExplainer(final_pipe.named_steps["model"])
    shap_values = explainer.shap_values(x_transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_ranking = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_ranking["rank"] = np.arange(1, len(shap_ranking) + 1)
    return metrics, shap_ranking[["rank", "feature", "mean_abs_shap"]]


def main() -> None:
    dataset = pd.read_csv(FINAL_DATASET_PATH)
    labels = pd.read_csv(LABEL_PATH)
    labels["PTID"] = normalize_id(labels["PTID"])
    labels["baseline_date"] = pd.to_datetime(labels["baseline_date"], errors="coerce")
    vitals = prepare_vitals()

    old_audit = audit_bmi(dataset["VITALS_BMI"], "before_correction")
    log("====================================================")
    log("STEP 1 - CURRENT VITALS_BMI AUDIT")
    log("====================================================")
    for key, value in old_audit.items():
        if key != "stage":
            log(f"{key}: {value}")

    trace = pd.DataFrame([select_safe_vitals_row(row, vitals) for _, row in labels.iterrows()])
    corrected = dataset.copy()
    corrected["VITALS_BMI"] = trace["corrected_bmi"].values
    corrected.to_csv(CORRECTED_DATASET_PATH, index=False)
    trace.to_csv(BMI_TRACE_PATH, index=False)

    new_audit = audit_bmi(corrected["VITALS_BMI"], "after_correction")
    pd.DataFrame([old_audit, new_audit]).to_csv(BMI_AUDIT_PATH, index=False)

    log("\n====================================================")
    log("STEP 5 - CORRECTED VITALS_BMI AUDIT")
    log("====================================================")
    for key, value in new_audit.items():
        if key != "stage":
            log(f"{key}: {value}")

    log("\n====================================================")
    log("STEP 7 - RETRAIN OPTIMIZED XGBOOST")
    log("====================================================")
    metrics, shap_ranking = train_evaluate_and_shap(corrected)
    pd.DataFrame([metrics]).to_csv(BMI_MODEL_RESULTS_PATH, index=False)
    shap_ranking.to_csv(BMI_SHAP_RANKING_PATH, index=False)
    log(json.dumps(metrics, indent=2))

    old_rank = "not available"
    if OLD_SHAP_PATH.exists():
        old_shap = pd.read_csv(OLD_SHAP_PATH)
        if "VITALS_BMI" in set(old_shap["feature"]):
            old_rank = int(old_shap.loc[old_shap["feature"] == "VITALS_BMI", "rank"].iloc[0])
    new_rank = int(shap_ranking.loc[shap_ranking["feature"] == "VITALS_BMI", "rank"].iloc[0])

    log("\n====================================================")
    log("STEP 8 - BMI SHAP RANK CHANGE")
    log("====================================================")
    log(f"Old BMI rank: {old_rank}")
    log(f"New BMI rank: {new_rank}")
    log("\nNew Top 20 SHAP features:")
    log(shap_ranking.head(20).to_string(index=False))

    log(f"\nSaved corrected dataset: {CORRECTED_DATASET_PATH}")
    log(f"Saved BMI audit report: {BMI_AUDIT_PATH}")
    log(f"Saved BMI trace: {BMI_TRACE_PATH}")
    log(f"Saved BMI-corrected model results: {BMI_MODEL_RESULTS_PATH}")
    log(f"Saved BMI-corrected SHAP ranking: {BMI_SHAP_RANKING_PATH}")


if __name__ == "__main__":
    main()
