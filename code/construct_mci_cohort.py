"""Apply the documented quality-control exclusions to the MCI cohort."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths


import pandas as pd


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
RAW_DXSUM_PATH = PROJECT_DIR / "data" / "raw" / "ALL" / "All_Subjects_DXSUM_08Jun2026.csv"
LABEL_PATH = PROCESSED_DIR / "mci_to_ad_labels_from_all_csv_scan.csv"

STRICT_COHORT_PATH = PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"
STRICT_AUDIT_PATH = PROCESSED_DIR / "strict_mci_to_ad_publication_audit.csv"
SUMMARY_PATH = PROCESSED_DIR / "strict_mci_to_ad_publication_summary.txt"

NO_TRUE_FOLLOWUP_EXCLUSIONS = {"033_S_10738", "073_S_7052", "137_S_6883", "941_S_7041"}
TIMELINE_INCONSISTENCY_EXCLUSIONS = {"018_S_2180"}
MISSING_CONVERSION_DATE_FLAG = "941_S_4420"


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def normalize_diagnosis(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        code = int(float(value))
    except Exception:
        return None
    return {1: "CN", 2: "MCI", 3: "AD"}.get(code)


def load_dxsum() -> pd.DataFrame:
    dx = pd.read_csv(RAW_DXSUM_PATH, low_memory=False)
    keep = ["PTID", "RID", "VISCODE", "VISCODE2", "EXAMDATE", "DIAGNOSIS"]
    dx = dx[[col for col in keep if col in dx.columns]].copy()
    dx["PTID"] = normalize_id(dx["PTID"])
    dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
    dx["diagnosis_norm"] = dx["DIAGNOSIS"].apply(normalize_diagnosis)
    dx = dx.dropna(subset=["PTID", "RID", "diagnosis_norm"])
    return dx


def future_dx_summary(row: pd.Series, dx: pd.DataFrame) -> tuple[int, bool]:
    subject_dx = dx[(dx["PTID"] == row["PTID"]) & (dx["RID"] == row["RID"])].copy()
    baseline_date = row["baseline_date"]
    future = subject_dx[subject_dx["EXAMDATE"] > baseline_date].copy()
    has_future_ad = bool((future["diagnosis_norm"] == "AD").any())
    return int(len(future)), has_future_ad


def exclusion_reason(ptid: str) -> str:
    if ptid in NO_TRUE_FOLLOWUP_EXCLUSIONS:
        return "No true longitudinal follow-up after baseline EXAMDATE"
    if ptid in TIMELINE_INCONSISTENCY_EXCLUSIONS:
        return "Visit/date inconsistency: valid MCI m12 visit occurs before selected baseline"
    return ""


def main() -> None:
    labels = pd.read_csv(LABEL_PATH)
    labels["PTID"] = normalize_id(labels["PTID"])
    labels["baseline_date"] = pd.to_datetime(labels["baseline_date"], errors="coerce")
    labels["conversion_date"] = pd.to_datetime(labels["conversion_date"], errors="coerce")
    labels["last_followup_date"] = pd.to_datetime(labels["last_followup_date"], errors="coerce")
    labels["conversion_label"] = pd.to_numeric(labels["conversion_label"], errors="coerce").astype(int)

    dx = load_dxsum()
    future_counts = labels.apply(lambda row: future_dx_summary(row, dx), axis=1)
    labels["true_followup_count_after_baseline_date"] = [x[0] for x in future_counts]
    labels["has_future_ad_after_baseline_date"] = [x[1] for x in future_counts]
    labels["missing_conversion_date"] = False
    labels.loc[labels["PTID"] == MISSING_CONVERSION_DATE_FLAG, "missing_conversion_date"] = True
    labels["exclude_from_time_to_conversion_analysis"] = labels["missing_conversion_date"]

    excluded = NO_TRUE_FOLLOWUP_EXCLUSIONS | TIMELINE_INCONSISTENCY_EXCLUSIONS
    strict = labels[~labels["PTID"].isin(excluded)].copy()
    strict.to_csv(STRICT_COHORT_PATH, index=False)

    audit_rows = []
    for _, row in labels.iterrows():
        reason = exclusion_reason(row["PTID"])
        included = "No" if reason else "Yes"
        notes = []
        if row["PTID"] == MISSING_CONVERSION_DATE_FLAG:
            notes.append("missing_conversion_date; retain for binary classification; exclude from time-to-conversion analyses")
        if included == "Yes" and row["conversion_label"] == 0 and row["has_future_ad_after_baseline_date"]:
            notes.append("WARNING: stable subject has future AD after baseline date")
        if included == "Yes" and row["true_followup_count_after_baseline_date"] < 1:
            notes.append("WARNING: included subject lacks true follow-up after baseline date")

        audit_rows.append(
            {
                "PTID": row["PTID"],
                "Included": included,
                "Exclusion Reason": reason,
                "Baseline Date": row["baseline_date"],
                "Follow-up Count": row["true_followup_count_after_baseline_date"],
                "Converter Label": row["conversion_label"],
                "Notes": "; ".join(notes),
            }
        )
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(STRICT_AUDIT_PATH, index=False)

    n_total = len(strict)
    n_converters = int((strict["conversion_label"] == 1).sum())
    n_stable = int((strict["conversion_label"] == 0).sum())
    stable_future_ad = int(((strict["conversion_label"] == 0) & strict["has_future_ad_after_baseline_date"]).sum())
    no_true_followup = int((strict["true_followup_count_after_baseline_date"] < 1).sum())

    summary = (
        "After quality-control review, subjects without true longitudinal follow-up and subjects with "
        "unresolved visit-date inconsistencies were excluded. Conversion labels were derived exclusively "
        "from longitudinal diagnosis records and were not used as predictive features. One subject with "
        "a missing conversion date was retained for classification analyses but excluded from "
        "time-to-conversion analyses."
    )
    SUMMARY_PATH.write_text(
        "\n".join(
            [
                summary,
                "",
                f"Total subjects: {n_total}",
                f"Converters: {n_converters}",
                f"Stable MCI: {n_stable}",
                f"Stable subjects with future AD diagnosis: {stable_future_ad}",
                f"Included subjects without true follow-up after baseline date: {no_true_followup}",
                f"Missing conversion date flagged subject: {MISSING_CONVERSION_DATE_FLAG}",
            ]
        ),
        encoding="utf-8",
    )

    log("====================================================")
    log("STRICT PUBLICATION COHORT")
    log("====================================================")
    log(f"Total subjects: {n_total}")
    log(f"Converters: {n_converters}")
    log(f"Stable MCI: {n_stable}")
    log("")
    log("Verification:")
    log(f"No stable subject has future AD diagnosis: {'PASS' if stable_future_ad == 0 else 'FAIL'}")
    log(f"Every included subject has true follow-up after baseline date: {'PASS' if no_true_followup == 0 else 'FAIL'}")
    log("")
    log(summary)
    log("")
    log(f"Saved strict cohort: {STRICT_COHORT_PATH}")
    log(f"Saved audit file: {STRICT_AUDIT_PATH}")
    log(f"Saved summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
