"""Audit acquisition timing in the legacy Top-20 interpretation inputs."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import re

import numpy as np
import pandas as pd


PROJECT_DIR = _paths.PACKAGE_ROOT
RAW_DIR = PROJECT_DIR / "data" / "raw" / "ALL"
PROCESSED_DIR = _paths.processed_dir()

DATASET_PATH = PROCESSED_DIR / "baseline_mci_to_ad_selected_features_dataset.csv"
LABEL_PATH = PROCESSED_DIR / "mci_to_ad_labels_from_all_csv_scan.csv"
REPORT_PATH = PROCESSED_DIR / "top20_leakage_audit_report.csv"
DETAIL_PATH = PROCESSED_DIR / "top20_leakage_audit_subject_details.csv"

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
    "MMSE": "All_Subjects_MMSE_08Jun2026.csv",
    "CDR": "All_Subjects_CDR_08Jun2026.csv",
    "ADAS": "All_Subjects_ADAS_08Jun2026.csv",
    "NEUROBAT": "All_Subjects_NEUROBAT_08Jun2026.csv",
    "FAQ": "All_Subjects_FAQ_08Jun2026.csv",
    "MOCA": "All_Subjects_MOCA_08Jun2026.csv",
    "PTDEMOG": "All_Subjects_PTDEMOG_08Jun2026.csv",
    "GENETIC": "All_Subjects_GENETIC_08Jun2026.csv",
    "APOERES": "All_Subjects_APOERES_08Jun2026.csv",
    "AMY": "All_Subjects_UCBERKELEY_AMY_6MM_08Jun2026.csv",
    "VITALS": "All_Subjects_VITALS_08Jun2026.csv",
}

DATE_COLUMNS = ["EXAMDATE", "VISDATE", "SCANDATE", "APTESTDT"]
LEAKAGE_NAME_PATTERNS = [
    r"DIAG",
    r"\bDX\b",
    r"CONVERT",
    r"CHANGE",
    r"FUTURE",
    r"LABEL",
    r"DEMENTIA",
    r"FOLLOWUP",
    r"MCI",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_visit(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def select_date_column(df: pd.DataFrame) -> str | None:
    for col in DATE_COLUMNS:
        if col in df.columns:
            return col
    return None


def module_for_feature(feature: str) -> str:
    prefix = feature.split("_", 1)[0]
    if prefix not in SELECTED_FILES:
        raise ValueError(f"Cannot map feature to source module: {feature}")
    return prefix


def raw_column_for_feature(feature: str) -> str:
    module = module_for_feature(feature)
    raw_col = feature[len(module) + 1 :]
    if feature == "VITALS_BMI":
        return "BMI"
    return raw_col


def has_leakage_name(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_NAME_PATTERNS)


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


def select_source_rows(labels: pd.DataFrame, raw: pd.DataFrame, module: str) -> pd.DataFrame:
    df = add_engineered_columns(raw, module)
    if "PTID" not in df.columns:
        raise ValueError(f"{module} source file has no PTID column.")

    df["_PTID_KEY"] = normalize_id(df["PTID"])
    date_col = select_date_column(df)
    if date_col:
        df["_source_date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        df["_source_date"] = pd.NaT

    for col in ["VISCODE", "VISCODE2"]:
        if col not in df.columns:
            df[col] = pd.NA

    base = labels[["PTID", "baseline_date", "baseline_visit"]].copy()
    base["_PTID_KEY"] = normalize_id(base["PTID"])
    base["baseline_date"] = pd.to_datetime(base["baseline_date"], errors="coerce")
    base["_baseline_visit_norm"] = base["baseline_visit"].apply(normalize_visit)

    merged = base.merge(df, on="_PTID_KEY", how="left", suffixes=("_label", ""))
    merged["_viscode_norm"] = merged["VISCODE"].apply(normalize_visit)
    merged["_viscode2_norm"] = merged["VISCODE2"].apply(normalize_visit)
    merged["_days_from_baseline"] = (merged["_source_date"] - merged["baseline_date"]).dt.days
    merged["_abs_days_from_baseline"] = merged["_days_from_baseline"].abs()

    exact_viscode2 = merged["_viscode2_norm"].eq(merged["_baseline_visit_norm"])
    exact_viscode = merged["_viscode_norm"].eq(merged["_baseline_visit_norm"])
    bl_viscode2 = merged["_viscode2_norm"].eq("bl")
    bl_viscode = merged["_viscode_norm"].eq("bl")
    near_180 = merged["_days_from_baseline"].between(-180, 180, inclusive="both")
    pre_365 = merged["_days_from_baseline"].between(-365, 0, inclusive="both")

    merged["_selection_priority"] = np.select(
        [exact_viscode2, exact_viscode, bl_viscode2, bl_viscode, near_180, pre_365],
        [1, 2, 3, 4, 5, 6],
        default=99,
    )
    usable = merged[merged["_selection_priority"] < 99].copy()
    if usable.empty:
        return base[["PTID", "baseline_date", "baseline_visit"]].copy()

    usable = usable.sort_values(
        ["_PTID_KEY", "_selection_priority", "_abs_days_from_baseline", "_days_from_baseline"]
    )
    selected = usable.groupby("_PTID_KEY", as_index=False).first()
    selected["PTID"] = selected["PTID_label"]
    return selected


def audit_feature(
    feature: str,
    source_rows: pd.DataFrame,
    source_file: str,
    modeling_dataset: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    raw_col = raw_column_for_feature(feature)
    if raw_col not in source_rows.columns:
        available_mask = pd.Series(False, index=source_rows.index)
    else:
        available_mask = source_rows[raw_col].notna()

    feature_model_available = modeling_dataset.set_index("PTID")[feature].notna()
    detail = source_rows[["PTID", "baseline_date", "baseline_visit", "VISCODE", "VISCODE2", "_source_date"]].copy()
    detail["feature_name"] = feature
    detail["source_file"] = source_file
    detail["raw_column"] = raw_col
    detail["feature_available_in_source"] = available_mask.values
    detail["feature_available_in_model_dataset"] = detail["PTID"].map(feature_model_available).fillna(False)
    detail["days_from_baseline"] = (detail["_source_date"] - detail["baseline_date"]).dt.days
    detail["after_baseline_180d"] = detail["days_from_baseline"] > 180
    detail["future_visit_code"] = detail["VISCODE2"].apply(is_future_visit_code) | detail["VISCODE"].apply(is_future_visit_code)
    detail["future_visit_outside_window"] = detail["future_visit_code"] & ~detail["days_from_baseline"].between(
        -180,
        180,
        inclusive="both",
    )
    detail["feature_name_leakage_flag"] = has_leakage_name(feature)

    available_detail = detail[detail["feature_available_in_source"]].copy()
    n_subjects_available = int(len(available_detail))
    n_subjects_with_source_date = int(available_detail["_source_date"].notna().sum())
    n_after_baseline_180d = int(available_detail["after_baseline_180d"].sum())
    n_future_visit_code = int(available_detail["future_visit_outside_window"].sum())
    max_days = available_detail["days_from_baseline"].max()
    median_days = available_detail["days_from_baseline"].median()

    reasons = []
    if has_leakage_name(feature):
        reasons.append("Feature name contains diagnostic/leakage term.")
    if n_after_baseline_180d:
        reasons.append(f"{n_after_baseline_180d} available subjects use source date > baseline + 180 days.")
    if n_future_visit_code:
        reasons.append(f"{n_future_visit_code} available subjects use future visit code outside baseline window.")
    if n_subjects_available == 0:
        reasons.append("No available source values found.")
    if not reasons:
        reasons.append("No temporal or name-based leakage flags detected.")

    if has_leakage_name(feature) or n_after_baseline_180d > 0 or n_future_visit_code > 0:
        risk = "HIGH-RISK"
    elif n_subjects_with_source_date < n_subjects_available:
        risk = "WARNING"
        reasons.append("Some available values have missing source dates; visit-code selection was used.")
    else:
        risk = "SAFE"

    summary = {
        "feature_name": feature,
        "source_file": source_file,
        "n_subjects_available": n_subjects_available,
        "n_subjects_with_source_date": n_subjects_with_source_date,
        "n_after_baseline_180d": n_after_baseline_180d,
        "n_future_visit_code": n_future_visit_code,
        "max_days_from_baseline": max_days,
        "median_days_from_baseline": median_days,
        "leakage_risk_level": risk,
        "reason": " ".join(reasons),
    }
    detail = detail.rename(columns={"_source_date": "source_date"})
    return summary, detail


def main() -> None:
    labels = pd.read_csv(LABEL_PATH)
    labels["PTID"] = normalize_id(labels["PTID"])
    labels["baseline_date"] = pd.to_datetime(labels["baseline_date"], errors="coerce")
    modeling_dataset = pd.read_csv(DATASET_PATH)
    modeling_dataset["PTID"] = normalize_id(modeling_dataset["PTID"])

    log("====================================================")
    log("TOP-20 SHAP FEATURE LEAKAGE AUDIT")
    log("====================================================")
    log(f"Subjects in label file: {labels['PTID'].nunique()}")
    log(f"Subjects in modeling dataset: {modeling_dataset['PTID'].nunique()}")

    missing_top20 = [feature for feature in TOP20_FEATURES if feature not in modeling_dataset.columns]
    if missing_top20:
        raise ValueError(f"Top-20 features missing from modeling dataset: {missing_top20}")

    source_cache = {}
    summary_rows = []
    detail_rows = []

    for feature in TOP20_FEATURES:
        module = module_for_feature(feature)
        source_file = SELECTED_FILES[module]
        if module not in source_cache:
            log(f"Loading and selecting source rows for {module}: {source_file}")
            raw = pd.read_csv(RAW_DIR / source_file, low_memory=False)
            source_cache[module] = select_source_rows(labels, raw, module)
        summary, detail = audit_feature(feature, source_cache[module], source_file, modeling_dataset)
        summary_rows.append(summary)
        detail_rows.append(detail)

    report = pd.DataFrame(summary_rows)
    details = pd.concat(detail_rows, ignore_index=True)
    report.to_csv(REPORT_PATH, index=False)
    details.to_csv(DETAIL_PATH, index=False)

    log("\n====================================================")
    log("AUDIT SUMMARY")
    log("====================================================")
    for level in ["SAFE", "WARNING", "HIGH-RISK"]:
        subset = report[report["leakage_risk_level"] == level]
        log(f"\n{level} features ({len(subset)}):")
        if subset.empty:
            log("  None")
        else:
            for feature in subset["feature_name"]:
                log(f"  {feature}")

    log("\nFull report:")
    log(report.to_string(index=False))
    log(f"\nSaved leakage audit report: {REPORT_PATH}")
    log(f"Saved subject-level audit details: {DETAIL_PATH}")


if __name__ == "__main__":
    main()
