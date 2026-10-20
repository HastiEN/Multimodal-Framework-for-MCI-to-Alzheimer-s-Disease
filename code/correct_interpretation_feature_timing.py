"""Rebuild the historical Top-20 inputs using the documented timing repair."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import re

import numpy as np
import pandas as pd
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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_DIR = _paths.PACKAGE_ROOT
RAW_DIR = PROJECT_DIR / "data" / "raw" / "ALL"
PROCESSED_DIR = _paths.processed_dir()

LABEL_PATH = PROCESSED_DIR / "mci_to_ad_labels_from_all_csv_scan.csv"
CURRENT_DATASET_PATH = PROCESSED_DIR / "baseline_mci_to_ad_selected_features_dataset.csv"
LEAKAGE_DETAILS_PATH = PROCESSED_DIR / "top20_leakage_audit_subject_details.csv"

FIXED_DATASET_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed.csv"
FIXED_TRACE_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed_trace.csv"
LEAKED_PAIRS_PATH = PROCESSED_DIR / "top20_leaked_subject_feature_pairs.csv"
AFTER_AUDIT_PATH = PROCESSED_DIR / "top20_leakage_audit_after_fix.csv"
PERFORMANCE_PATH = PROCESSED_DIR / "top20_leakage_fixed_xgboost_results.csv"
CV_FOLDS_PATH = PROCESSED_DIR / "top20_leakage_fixed_repeated_cv_folds.csv"
IMPACT_REPORT_PATH = PROCESSED_DIR / "leakage_fix_impact_report.md"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MAX_POST_BASELINE_DAYS = 180
MAX_PRE_BASELINE_DAYS = 365

TOP20_FEATURES = [
    "ADAS_TOTAL13",
    "AMY_SUMMARY_SUVR",
    "VITALS_BMI",
    "FAQ_FAQTOTAL",
    "CDR_CDRSB",
    "AMY_CENTILOIDS",
    "CDR_CDJUDGE",
    "NEUROBAT_BNTPHON",
    "MOCA_DELW3",
    "NEUROBAT_AVTOT5",
    "NEUROBAT_LDELTOTAL",
    "PTDEMOG_PTEDUCAT",
    "NEUROBAT_BNTSPONT",
    "NEUROBAT_TRAASCOR",
    "NEUROBAT_ANARTERR",
    "VITALS_VSWEIGHT",
    "FAQ_FAQFINAN",
    "VITALS_VSBPSYS",
    "NEUROBAT_TRABSCOR",
    "VITALS_VSBPDIA",
]

SELECTED_FILES = {
    "CDR": "All_Subjects_CDR_08Jun2026.csv",
    "ADAS": "All_Subjects_ADAS_08Jun2026.csv",
    "NEUROBAT": "All_Subjects_NEUROBAT_08Jun2026.csv",
    "FAQ": "All_Subjects_FAQ_08Jun2026.csv",
    "MOCA": "All_Subjects_MOCA_08Jun2026.csv",
    "PTDEMOG": "All_Subjects_PTDEMOG_08Jun2026.csv",
    "AMY": "All_Subjects_UCBERKELEY_AMY_6MM_08Jun2026.csv",
    "VITALS": "All_Subjects_VITALS_08Jun2026.csv",
}

DATE_COLUMNS = ["EXAMDATE", "VISDATE", "SCANDATE", "APTESTDT"]
DIAGNOSTIC_NAME_PATTERNS = [r"DIAG", r"\bDX\b", r"CONVERT", r"CHANGE", r"FUTURE", r"LABEL", r"\bAD\b", r"\bMCI\b"]


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_visit(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def source_module(feature: str) -> str:
    module = feature.split("_", 1)[0]
    if module not in SELECTED_FILES:
        raise ValueError(f"Unknown Top-20 feature module: {feature}")
    return module


def raw_column(feature: str) -> str:
    module = source_module(feature)
    return "BMI" if feature == "VITALS_BMI" else feature[len(module) + 1 :]


def select_date_column(df: pd.DataFrame) -> str | None:
    for col in DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def has_diagnostic_name_leakage(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in DIAGNOSTIC_NAME_PATTERNS)


def is_future_visit_code(value: object) -> bool:
    text = normalize_visit(value)
    return bool(re.fullmatch(r"(m|v)\d+", text)) and text not in {"m00", "v00"}


def add_engineered_columns(raw: pd.DataFrame, module: str) -> pd.DataFrame:
    df = raw.copy()
    if module == "VITALS" and {"VSWEIGHT", "VSHEIGHT"}.issubset(df.columns):
        weight = pd.to_numeric(df["VSWEIGHT"], errors="coerce")
        height = pd.to_numeric(df["VSHEIGHT"], errors="coerce")
        height_m = height / 100.0
        if "VSHTUNIT" in df.columns:
            unit = df["VSHTUNIT"].astype(str).str.lower()
            height_m = np.where(unit.str.contains("in"), height * 0.0254, height_m)
        weight_kg = weight
        if "VSWTUNIT" in df.columns:
            unit = df["VSWTUNIT"].astype(str).str.lower()
            weight_kg = np.where(unit.str.contains("lb"), weight * 0.453592, weight_kg)
        df["BMI"] = weight_kg / (pd.Series(height_m, index=df.index) ** 2)
    return df


def prepare_raw_module(module: str) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / SELECTED_FILES[module], low_memory=False)
    df = add_engineered_columns(df, module)
    if "PTID" not in df.columns:
        raise ValueError(f"{SELECTED_FILES[module]} has no PTID column.")
    df["_PTID_KEY"] = normalize_id(df["PTID"])
    date_col = select_date_column(df)
    df["_source_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    for col in ["VISCODE", "VISCODE2"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["_viscode_norm"] = df["VISCODE"].apply(normalize_visit)
    df["_viscode2_norm"] = df["VISCODE2"].apply(normalize_visit)
    return df


def select_safe_feature_value(subject: pd.Series, raw_module: pd.DataFrame, feature: str) -> dict:
    col = raw_column(feature)
    ptid_key = str(subject["PTID"]).strip()
    baseline_date = subject["baseline_date"]
    baseline_visit = normalize_visit(subject["baseline_visit"])
    empty = {
        "PTID": subject["PTID"],
        "feature_name": feature,
        "source_file": SELECTED_FILES[source_module(feature)],
        "raw_column": col,
        "value": np.nan,
        "source_date": pd.NaT,
        "VISCODE": pd.NA,
        "VISCODE2": pd.NA,
        "days_from_baseline": np.nan,
        "selection_priority": pd.NA,
        "status": "NO_VALID_BASELINE_SOURCE",
    }
    if col not in raw_module.columns:
        empty["status"] = "RAW_COLUMN_MISSING"
        return empty

    candidates = raw_module[raw_module["_PTID_KEY"] == ptid_key].copy()
    candidates = candidates[candidates[col].notna()].copy()
    if candidates.empty:
        empty["status"] = "NO_SOURCE_VALUE"
        return empty

    candidates["_days_from_baseline"] = (candidates["_source_date"] - baseline_date).dt.days
    candidates["_abs_days_from_baseline"] = candidates["_days_from_baseline"].abs()

    exact_baseline_visit = candidates["_viscode2_norm"].eq(baseline_visit) | candidates["_viscode_norm"].eq(baseline_visit)
    bl_visit = candidates["_viscode2_norm"].eq("bl") | candidates["_viscode_norm"].eq("bl")
    near_window = candidates["_days_from_baseline"].between(
        -MAX_POST_BASELINE_DAYS,
        MAX_POST_BASELINE_DAYS,
        inclusive="both",
    )
    pre_window = candidates["_days_from_baseline"].between(-MAX_PRE_BASELINE_DAYS, 0, inclusive="both")

    # Strict temporal safety gate: no selected source row may occur after baseline + 180 days.
    temporally_safe = candidates["_source_date"].notna() & (candidates["_days_from_baseline"] <= MAX_POST_BASELINE_DAYS)
    candidates = candidates[temporally_safe].copy()
    if candidates.empty:
        return empty

    candidates["_priority"] = np.select(
        [
            exact_baseline_visit.loc[candidates.index],
            bl_visit.loc[candidates.index],
            near_window.loc[candidates.index],
            pre_window.loc[candidates.index],
        ],
        [1, 2, 3, 4],
        default=99,
    )
    candidates = candidates[candidates["_priority"] < 99].copy()
    if candidates.empty:
        return empty

    candidates = candidates.sort_values(["_priority", "_abs_days_from_baseline", "_days_from_baseline"])
    selected = candidates.iloc[0]
    return {
        "PTID": subject["PTID"],
        "feature_name": feature,
        "source_file": SELECTED_FILES[source_module(feature)],
        "raw_column": col,
        "value": selected[col],
        "source_date": selected["_source_date"],
        "VISCODE": selected["VISCODE"],
        "VISCODE2": selected["VISCODE2"],
        "days_from_baseline": selected["_days_from_baseline"],
        "selection_priority": int(selected["_priority"]),
        "status": "SELECTED_SAFE_SOURCE",
    }


def build_fixed_dataset(labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    modules = sorted({source_module(feature) for feature in TOP20_FEATURES})
    raw_cache = {module: prepare_raw_module(module) for module in modules}
    trace_rows = []
    final = pd.DataFrame({"conversion_label": labels["conversion_label"].astype(int)})

    for feature in TOP20_FEATURES:
        module = source_module(feature)
        log(f"Re-extracting safe feature: {feature}")
        selected_rows = [
            select_safe_feature_value(subject, raw_cache[module], feature)
            for _, subject in labels.iterrows()
        ]
        feature_trace = pd.DataFrame(selected_rows)
        trace_rows.append(feature_trace)
        final[feature] = feature_trace["value"].values

    trace = pd.concat(trace_rows, ignore_index=True)
    trace = trace.merge(labels[["PTID", "baseline_date", "baseline_visit"]], on="PTID", how="left")
    return final, trace


def make_xgb() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_pipeline(x: pd.DataFrame) -> Pipeline:
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
    return Pipeline([("preprocess", preprocessor), ("model", make_xgb())])


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


def train_holdout(dataset: pd.DataFrame) -> dict:
    x = dataset[TOP20_FEATURES].copy()
    y = dataset["conversion_label"].astype(int).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipe = make_pipeline(x_train)
    pipe.fit(x_train, y_train)
    y_score = pipe.predict_proba(x_test)[:, 1]
    return evaluate(y_test, y_score)


def repeated_cv(dataset: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    x = dataset[TOP20_FEATURES].copy()
    y = dataset["conversion_label"].astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        pipe = make_pipeline(x_train)
        pipe.fit(x_train, y_train)
        y_score = pipe.predict_proba(x_test)[:, 1]
        metrics = evaluate(y_test, y_score)
        metrics["fold"] = fold
        rows.append(metrics)
    folds = pd.DataFrame(rows)
    summary = {
        "cv_roc_auc_mean": folds["roc_auc"].mean(),
        "cv_roc_auc_std": folds["roc_auc"].std(ddof=1),
        "cv_pr_auc_mean": folds["pr_auc"].mean(),
        "cv_f1_mean": folds["f1"].mean(),
        "cv_recall_mean": folds["recall"].mean(),
        "cv_specificity_mean": folds["specificity"].mean(),
        "cv_balanced_accuracy_mean": folds["balanced_accuracy"].mean(),
    }
    return summary, folds


def build_after_fix_audit(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feature, group in trace.groupby("feature_name", sort=False):
        available = group[group["value"].notna()].copy()
        after_180 = available["days_from_baseline"] > MAX_POST_BASELINE_DAYS
        future_code = available["VISCODE2"].apply(is_future_visit_code) | available["VISCODE"].apply(is_future_visit_code)
        future_outside = future_code & ~available["days_from_baseline"].between(
            -MAX_POST_BASELINE_DAYS,
            MAX_POST_BASELINE_DAYS,
            inclusive="both",
        )
        name_leak = has_diagnostic_name_leakage(feature)
        if name_leak or int(after_180.sum()) or int(future_outside.sum()):
            risk = "HIGH-RISK"
        elif int(available["source_date"].notna().sum()) < len(available):
            risk = "WARNING"
        else:
            risk = "SAFE"
        reason = "No temporal or diagnostic-name leakage detected."
        if risk != "SAFE":
            reason = (
                f"name_leak={name_leak}; after_180={int(after_180.sum())}; "
                f"future_visit_outside_window={int(future_outside.sum())}"
            )
        rows.append(
            {
                "feature_name": feature,
                "source_file": group["source_file"].iloc[0],
                "n_subjects_available": int(len(available)),
                "n_subjects_with_source_date": int(available["source_date"].notna().sum()),
                "n_after_baseline_180d": int(after_180.sum()),
                "n_future_visit_code": int(future_outside.sum()),
                "max_days_from_baseline": available["days_from_baseline"].max(),
                "median_days_from_baseline": available["days_from_baseline"].median(),
                "leakage_risk_level": risk,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def extract_leaked_pairs() -> pd.DataFrame:
    details = pd.read_csv(LEAKAGE_DETAILS_PATH)
    details["source_date"] = pd.to_datetime(details["source_date"], errors="coerce")
    details["baseline_date"] = pd.to_datetime(details["baseline_date"], errors="coerce")
    leaked = details[
        (details["feature_available_in_source"] == True)
        & (details["source_date"] > details["baseline_date"] + pd.Timedelta(days=MAX_POST_BASELINE_DAYS))
    ].copy()
    cols = ["PTID", "feature_name", "source_file", "source_date", "baseline_date", "days_from_baseline"]
    leaked[cols].to_csv(LEAKED_PAIRS_PATH, index=False)
    return leaked[cols]


def write_impact_report(
    leaked_pairs: pd.DataFrame,
    original: pd.DataFrame,
    fixed: pd.DataFrame,
    trace: pd.DataFrame,
    before_metrics: dict,
    after_metrics: dict,
    cv_summary: dict,
    audit: pd.DataFrame,
) -> None:
    changed_to_nan = 0
    changed_values = 0
    for feature in TOP20_FEATURES:
        before = original[feature]
        after = fixed[feature]
        changed_to_nan += int(before.notna().sum() - after.notna().sum())
        changed_values += int((before.astype(str) != after.astype(str)).sum())

    safe = bool(
        (audit["n_after_baseline_180d"] == 0).all()
        and (audit["n_future_visit_code"] == 0).all()
        and (audit["leakage_risk_level"] != "HIGH-RISK").all()
    )
    verdict = "SAFE TO PUBLISH" if safe else "MANUAL REVIEW REQUIRED"
    lines = [
        "# Top-20 Temporal Leakage Fix Impact Report",
        "",
        "## Correction Summary",
        f"- Leaked subject-feature pairs identified: {len(leaked_pairs)}",
        f"- Values changed after strict re-extraction: {changed_values}",
        f"- Values converted to NaN or unavailable after strict re-extraction: {changed_to_nan}",
        f"- Selected safe source rows: {int((trace['status'] == 'SELECTED_SAFE_SOURCE').sum())}",
        f"- Missing/no valid safe source rows: {int((trace['status'] != 'SELECTED_SAFE_SOURCE').sum())}",
        "",
        "## Performance Before Fix",
        f"- ROC-AUC: {before_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {before_metrics['pr_auc']:.4f}",
        f"- Accuracy: {before_metrics['accuracy']:.4f}",
        f"- Precision: {before_metrics['precision']:.4f}",
        f"- Recall: {before_metrics['recall']:.4f}",
        f"- Specificity: {before_metrics['specificity']:.4f}",
        f"- F1: {before_metrics['f1']:.4f}",
        f"- Balanced Accuracy: {before_metrics['balanced_accuracy']:.4f}",
        "",
        "## Performance After Fix",
        f"- ROC-AUC: {after_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {after_metrics['pr_auc']:.4f}",
        f"- Accuracy: {after_metrics['accuracy']:.4f}",
        f"- Precision: {after_metrics['precision']:.4f}",
        f"- Recall: {after_metrics['recall']:.4f}",
        f"- Specificity: {after_metrics['specificity']:.4f}",
        f"- F1: {after_metrics['f1']:.4f}",
        f"- Balanced Accuracy: {after_metrics['balanced_accuracy']:.4f}",
        "",
        "## Performance Delta",
        f"- Delta ROC-AUC: {after_metrics['roc_auc'] - before_metrics['roc_auc']:.4f}",
        f"- Delta PR-AUC: {after_metrics['pr_auc'] - before_metrics['pr_auc']:.4f}",
        "",
        "## Repeated CV After Fix",
        f"- ROC-AUC: {cv_summary['cv_roc_auc_mean']:.4f} +/- {cv_summary['cv_roc_auc_std']:.4f}",
        f"- PR-AUC: {cv_summary['cv_pr_auc_mean']:.4f}",
        f"- F1: {cv_summary['cv_f1_mean']:.4f}",
        f"- Recall: {cv_summary['cv_recall_mean']:.4f}",
        f"- Specificity: {cv_summary['cv_specificity_mean']:.4f}",
        "",
        "## Final Recommendation",
        f"- Verdict: {verdict}",
        "- Use the leakage-fixed Top-20 dataset for publication modeling. Values without a valid baseline-compatible source are left as missing and should be imputed inside cross-validation only.",
    ]
    IMPACT_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    labels = pd.read_csv(LABEL_PATH)
    labels["PTID"] = normalize_id(labels["PTID"])
    labels["baseline_date"] = pd.to_datetime(labels["baseline_date"], errors="coerce")
    labels["conversion_label"] = pd.to_numeric(labels["conversion_label"], errors="coerce").astype(int)
    labels = labels.sort_values("PTID").reset_index(drop=True)

    original = pd.read_csv(CURRENT_DATASET_PATH)
    original["PTID"] = normalize_id(original["PTID"])
    original = labels[["PTID"]].merge(original, on="PTID", how="left")

    log("====================================================")
    log("STEP 1 - IDENTIFY LEAKED SUBJECT-FEATURE PAIRS")
    log("====================================================")
    leaked_pairs = extract_leaked_pairs()
    log(f"Leaked subject-feature pairs: {len(leaked_pairs)}")
    log(f"Unique leaked subjects: {leaked_pairs['PTID'].nunique()}")

    log("\n====================================================")
    log("STEP 2 - RE-EXTRACT SAFE BASELINE FEATURES")
    log("====================================================")
    fixed, trace = build_fixed_dataset(labels)
    fixed.to_csv(FIXED_DATASET_PATH, index=False)
    trace.to_csv(FIXED_TRACE_PATH, index=False)

    log("\n====================================================")
    log("STEP 3 - VERIFY FIX")
    log("====================================================")
    audit = build_after_fix_audit(trace)
    audit.to_csv(AFTER_AUDIT_PATH, index=False)
    log(audit.to_string(index=False))

    log("\n====================================================")
    log("STEP 5 - RETRAIN XGBOOST")
    log("====================================================")
    before_dataset = pd.DataFrame({"conversion_label": labels["conversion_label"]})
    for feature in TOP20_FEATURES:
        before_dataset[feature] = original[feature].values
    before_metrics = train_holdout(before_dataset)
    after_metrics = train_holdout(fixed)
    pd.DataFrame(
        [
            {"dataset": "before_fix", **before_metrics},
            {"dataset": "after_fix", **after_metrics},
        ]
    ).to_csv(PERFORMANCE_PATH, index=False)
    log("Before fix:")
    log(json.dumps(before_metrics, indent=2))
    log("After fix:")
    log(json.dumps(after_metrics, indent=2))

    log("\n====================================================")
    log("STEP 6 - REPEATED CROSS VALIDATION")
    log("====================================================")
    cv_summary, cv_folds = repeated_cv(fixed)
    cv_folds.to_csv(CV_FOLDS_PATH, index=False)
    log(json.dumps(cv_summary, indent=2))

    write_impact_report(
        leaked_pairs=leaked_pairs,
        original=before_dataset,
        fixed=fixed,
        trace=trace,
        before_metrics=before_metrics,
        after_metrics=after_metrics,
        cv_summary=cv_summary,
        audit=audit,
    )

    safe_to_publish = bool(
        (audit["n_after_baseline_180d"] == 0).all()
        and (audit["n_future_visit_code"] == 0).all()
        and (audit["leakage_risk_level"] != "HIGH-RISK").all()
    )
    log("\n====================================================")
    log("STEP 8 - FINAL VALIDATION")
    log("====================================================")
    if safe_to_publish:
        log("SAFE TO PUBLISH")
    else:
        log("MANUAL REVIEW REQUIRED")
        log(audit[audit["leakage_risk_level"] != "SAFE"].to_string(index=False))

    log(f"\nSaved fixed dataset: {FIXED_DATASET_PATH}")
    log(f"Saved after-fix audit: {AFTER_AUDIT_PATH}")
    log(f"Saved impact report: {IMPACT_REPORT_PATH}")


if __name__ == "__main__":
    main()
