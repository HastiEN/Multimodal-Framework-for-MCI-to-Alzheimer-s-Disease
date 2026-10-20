"""Reproduce baseline-compatible extraction and its feature catalog."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = _paths.PACKAGE_ROOT
RAW_DIR = PROJECT_DIR / "data" / "raw" / "ALL"
PROCESSED_DIR = _paths.processed_dir()
COHORT_PATH = PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"

DATA_INVENTORY_PATH = PROCESSED_DIR / "data_inventory_report.csv"
CANDIDATE_CATALOG_PATH = PROCESSED_DIR / "candidate_feature_catalog.csv"
FEATURE_COVERAGE_PATH = PROCESSED_DIR / "feature_coverage_report.csv"
SUBJECT_SYNC_PATH = PROCESSED_DIR / "subject_sync_audit.csv"
MISSINGNESS_PATH = PROCESSED_DIR / "feature_missingness_report.csv"
LEAKAGE_AUDIT_PATH = PROCESSED_DIR / "feature_leakage_audit.csv"
RECOMMENDED_FEATURES_PATH = PROCESSED_DIR / "recommended_features.csv"
FINAL_DATASET_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
FINAL_REPORT_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset_report.txt"

SUBJECT_COLUMNS = ["PTID", "RID", "subject_id", "SUBJECT_ID", "participant_id", "ParticipantID", "LONIUID"]
VISIT_COLUMNS = ["VISCODE", "VISCODE2", "VISIT", "VISITCODE"]
DATE_COLUMNS = ["EXAMDATE", "VISDATE", "SCANDATE", "APTESTDT", "EXAM_DATE", "Study Date", "USERDATE"]

ADMIN_PATTERNS = [
    r"^ID$",
    r"^RID$",
    r"^PTID$",
    r"^SITEID$",
    r"^LONIUID$",
    r"USERDATE",
    r"UPDATE_STAMP",
    r"PHASE",
    r"VISCODE",
    r"VISIT",
    r"EXAMDATE",
    r"VISDATE",
    r"SCANDATE",
    r"RUNDATE",
    r"COMMENT",
    r"SOURCE",
    r"VERSION",
    r"STATUS",
    r"RECNO",
    r"IMAGEUID",
    r"PROCESS",
    r"QC",
    r"DD_CRF",
    r"LANGUAGE_CODE",
]

LEAKAGE_PATTERNS = [
    r"DIAGNOSIS",
    r"DXCHANGE",
    r"DXCURREN",
    r"DXNORM",
    r"DXMCI",
    r"DXAD",
    r"DXCONFID",
    r"\bDX\b",
    r"CONVERT",
    r"CONVERSION",
    r"FOLLOWUP",
    r"FUTURE",
    r"LABEL",
    r"DEMENTIA",
    r"FINAL_DIAGNOSIS",
    r"BASELINE_DIAGNOSIS",
]

CATEGORY_KEYWORDS = {
    "Cognitive": ["MMSE", "MMSCORE", "ADAS", "TOTSCORE", "TOTAL13", "CDR", "CDRSB", "FAQ", "MOCA", "NEUROBAT", "ECOG", "RAVLT", "AVTOT", "LDEL", "BNT"],
    "Demographic": ["PTDEMOG", "AGE", "GENDER", "SEX", "PTEDUCAT", "EDUC", "RACE", "ETH", "MARRY", "HAND"],
    "Genetic": ["GENETIC", "APOE", "GENOTYPE", "APOERES"],
    "MedicalHistory": ["MEDHIST", "RECMHIST", "FAMHX", "FHQ", "BACKMEDS", "HYPERT", "DIABET", "CARDIO", "DEPRESS", "SMOK", "ALCOHOL"],
    "Laboratory": ["LAB", "CSF", "BIOMARK", "PLASMA", "TAU", "ABETA", "PTAU", "ELECSYS", "QUANTERIX", "FUJIREBIO", "ROCHE"],
    "MRI": ["MRI", "UCSFFSX", "HIPPOCAMPUS", "ENTORHINAL", "VENTRICLE", "CORTEX", "THICKNESS", "VOLUME", "FSX"],
    "PET": ["PET", "AMY", "TAU", "SUVR", "CENTILOID", "FDG", "AV45", "TRACER"],
    "DTI": ["DTI", "DTIROI", "FA_", "MD_", "RD_", "AD_", "AXIAL", "RADIAL", "DIFFUSION"],
    "Vitals": ["VITALS", "VSWEIGHT", "VSHEIGHT", "VSBP", "VSPULSE", "VSRESP", "VSTEMP", "BMI"],
}

LITERATURE_FEATURE_HINTS = [
    "MMSCORE",
    "TOTSCORE",
    "TOTAL13",
    "CDRSB",
    "FAQTOTAL",
    "MOCA",
    "APOE",
    "GENOTYPE",
    "AGE",
    "PTEDUCAT",
    "HIPPOCAMPUS",
    "ENTORHINAL",
    "VENTRICLE",
    "SUVR",
    "CENTILOID",
    "ABETA",
    "PTAU",
    "TAU",
    "BMI",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def clean_list(values: list[str], limit: int | None = None) -> str:
    values = [str(v) for v in values if str(v)]
    if limit is not None and len(values) > limit:
        return ";".join(values[:limit]) + f";...(+{len(values) - limit})"
    return ";".join(values)


def sanitize_name(text: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", text).strip("_")
    return re.sub(r"_+", "_", cleaned)


def source_prefix(file_name: str) -> str:
    name = file_name.replace("All_Subjects_", "").replace("_08Jun2026.csv", "")
    return sanitize_name(name).upper()[:45]


def first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def columns_matching(columns: list[str], patterns: list[str]) -> list[str]:
    out = []
    for col in columns:
        upper = col.upper()
        if any(re.search(pattern, upper) for pattern in patterns):
            out.append(col)
    return out


def infer_category(file_name: str, column: str) -> str:
    haystack = f"{file_name} {column}".upper()
    hits = []
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword.upper() in haystack for keyword in keywords):
            hits.append(category)
    if "PET" in hits:
        return "PET"
    if "MRI" in hits and "PET" not in hits:
        return "MRI"
    return hits[0] if hits else "Other"


def infer_data_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    numeric = pd.to_numeric(series, errors="coerce")
    non_missing = int(series.notna().sum())
    if non_missing and numeric.notna().sum() / non_missing >= 0.8:
        return "numeric_like"
    unique = series.dropna().astype(str).nunique()
    if unique <= 30:
        return "categorical"
    return "text_or_high_cardinality"


def is_admin_column(col: str) -> bool:
    upper = col.upper()
    return any(re.search(pattern, upper) for pattern in ADMIN_PATTERNS)


def leakage_status(file_name: str, col: str) -> tuple[str, str]:
    upper = f"{file_name} {col}".upper()
    if any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS):
        return "LEAKAGE", "Diagnosis, conversion, or follow-up related column/file name."
    if any(token in upper for token in ["BLCHANGE", "DXSUM", "MY_TABLE"]):
        return "POTENTIAL_LEAKAGE", "Diagnosis/change-summary file; avoid as predictor unless manually adjudicated."
    return "SAFE", "No name-based leakage signal; temporal extraction still required."


def is_candidate_feature(file_name: str, col: str, series: pd.Series) -> bool:
    if is_admin_column(col):
        return False
    status, _ = leakage_status(file_name, col)
    if status == "LEAKAGE":
        return False
    dtype = infer_data_type(series)
    if dtype == "text_or_high_cardinality":
        return False
    return series.notna().sum() > 0


def feature_description(file_name: str, col: str, category: str) -> str:
    return f"{category} candidate predictor from {file_name}: {col}"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    if suffix == ".txt":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def choose_subject_col(df: pd.DataFrame) -> str | None:
    return first_existing_column(df.columns.tolist(), SUBJECT_COLUMNS)


def choose_date_col(df: pd.DataFrame) -> str | None:
    return first_existing_column(df.columns.tolist(), DATE_COLUMNS)


def visit_order(value: object) -> int:
    if pd.isna(value):
        return 10_000_000
    text = str(value).strip().lower()
    if text in {"sc", "screen", "screening"}:
        return -2
    if text in {"bl", "baseline"}:
        return 0
    match = re.search(r"m(\d+)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"v(\d+)", text)
    if match:
        return int(match.group(1)) * 10
    return 10_000_000


def baseline_extract_file(cohort: pd.DataFrame, raw: pd.DataFrame, subject_col: str) -> tuple[pd.DataFrame, pd.Series]:
    df = raw.copy()
    df["_subject_key"] = normalize_id(df[subject_col])
    date_col = choose_date_col(df)
    df["_source_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
    for col in ["VISCODE", "VISCODE2", "VISIT", "VISITCODE"]:
        if col not in df.columns:
            df[col] = pd.NA
    df["_visit_primary"] = df["VISCODE2"].fillna(df["VISCODE"]).fillna(df["VISIT"]).fillna(df["VISITCODE"])
    df["_visit_norm"] = df["_visit_primary"].astype(str).str.strip().str.lower()
    df["_visit_order"] = df["_visit_primary"].apply(visit_order)

    if subject_col == "RID":
        base = cohort[["PTID", "RID", "baseline_date", "baseline_visit", "conversion_date"]].copy()
        base["_subject_key"] = normalize_id(base["RID"])
    else:
        base = cohort[["PTID", "RID", "baseline_date", "baseline_visit", "conversion_date"]].copy()
        base["_subject_key"] = normalize_id(base["PTID"])

    base["baseline_date"] = pd.to_datetime(base["baseline_date"], errors="coerce")
    base["conversion_date"] = pd.to_datetime(base["conversion_date"], errors="coerce")
    base["_baseline_visit_norm"] = base["baseline_visit"].astype(str).str.strip().str.lower()
    merged = base.merge(df, on="_subject_key", how="left", suffixes=("_cohort", ""))
    merged["_days_from_baseline"] = (merged["_source_date"] - merged["baseline_date"]).dt.days
    merged["_abs_days_from_baseline"] = merged["_days_from_baseline"].abs()

    exact = merged["_visit_norm"].eq(merged["_baseline_visit_norm"])
    pre_baseline = merged["_source_date"].notna() & (merged["_source_date"] <= merged["baseline_date"])
    near_baseline = merged["_days_from_baseline"].between(-180, 180, inclusive="both")
    not_after_conversion = merged["conversion_date"].isna() | merged["_source_date"].isna() | (merged["_source_date"] < merged["conversion_date"])

    # Strictly prefer exact baseline and pre-baseline. Use +180 near-baseline only if no better source exists
    # and never after conversion.
    merged["_priority"] = np.select(
        [
            exact & not_after_conversion,
            pre_baseline & not_after_conversion,
            near_baseline & not_after_conversion,
        ],
        [1, 2, 3],
        default=99,
    )
    usable = merged[merged["_priority"] < 99].copy()
    if usable.empty:
        selected = base[["PTID", "RID", "baseline_date", "baseline_visit"]].copy()
        selected["_extraction_method"] = "missing"
        selected["_source_date"] = pd.NaT
        selected["_days_from_baseline"] = np.nan
        return selected, pd.Series(False, index=cohort.index)

    ptid_key_col = "PTID_cohort" if "PTID_cohort" in usable.columns else "PTID"
    rid_key_col = "RID_cohort" if "RID_cohort" in usable.columns else "RID"
    usable = usable.sort_values(
        [ptid_key_col, "_priority", "_abs_days_from_baseline", "_days_from_baseline", "_visit_order"]
    )
    selected = usable.groupby(ptid_key_col, as_index=False).first()
    drop_raw_ids = []
    if ptid_key_col != "PTID" and "PTID" in selected.columns:
        drop_raw_ids.append("PTID")
    if rid_key_col != "RID" and "RID" in selected.columns:
        drop_raw_ids.append("RID")
    selected = selected.drop(columns=drop_raw_ids, errors="ignore")
    selected = selected.rename(columns={ptid_key_col: "PTID", rid_key_col: "RID"})
    method_map = {1: "exact_baseline_visit", 2: "nearest_pre_baseline", 3: "nearest_within_180d"}
    selected["_extraction_method"] = selected["_priority"].map(method_map).fillna("missing")
    selected = cohort[["PTID", "RID", "baseline_date", "baseline_visit"]].merge(
        selected,
        on=["PTID", "RID"],
        how="left",
        suffixes=("", "_selected"),
    )
    selected["_extraction_method"] = selected["_extraction_method"].fillna("missing")
    selected["_source_date"] = selected["_source_date"]
    selected["_days_from_baseline"] = selected["_days_from_baseline"]
    return selected, selected["_extraction_method"].ne("missing")


def recommendation_score(category: str, coverage: float, missing: float, leak_status: str, feature: str) -> float:
    if leak_status == "LEAKAGE":
        return 0.0
    score = coverage * 0.55 + max(0, 100 - missing) * 0.25
    if leak_status == "POTENTIAL_LEAKAGE":
        score -= 30
    if category in {"Cognitive", "Demographic", "Genetic", "MRI", "PET", "Vitals", "Laboratory"}:
        score += 10
    upper = feature.upper()
    if any(hint in upper for hint in LITERATURE_FEATURE_HINTS):
        score += 12
    return round(max(score, 0), 3)


def missing_class(missing_percent: float) -> str:
    if missing_percent < 20:
        return "LOW_MISSING"
    if missing_percent <= 50:
        return "MODERATE_MISSING"
    if missing_percent <= 80:
        return "HIGH_MISSING"
    return "EXTREME_MISSING"


def main() -> None:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = normalize_id(cohort["PTID"])
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort_n = cohort["PTID"].nunique()
    cohort_ptids = set(cohort["PTID"])

    files = sorted([p for p in RAW_DIR.iterdir() if p.suffix.lower() in {".csv", ".txt", ".xls", ".xlsx"}])
    log(f"Strict cohort subjects: {cohort_n}")
    log(f"Scanning {len(files)} raw files")

    inventory_rows = []
    catalog_rows = []
    coverage_rows = []
    sync_rows = []
    leakage_rows = []
    feature_blocks = [cohort[["PTID", "conversion_label"]].copy()]

    for idx, path in enumerate(files, start=1):
        log(f"[{idx}/{len(files)}] {path.name}")
        try:
            raw = read_table(path)
        except Exception as exc:
            inventory_rows.append({"filename": path.name, "error": str(exc)})
            continue

        columns = raw.columns.tolist()
        subject_col = choose_subject_col(raw)
        date_cols = [col for col in DATE_COLUMNS if col in columns]
        visit_cols = [col for col in VISIT_COLUMNS if col in columns]
        diagnosis_cols = columns_matching(columns, LEAKAGE_PATTERNS)
        category_cols = {cat: [] for cat in CATEGORY_KEYWORDS}
        for col in columns:
            cat = infer_category(path.name, col)
            if cat in category_cols:
                category_cols[cat].append(col)

        unique_subjects = 0
        overlap = 0
        if subject_col:
            raw_subjects = set(normalize_id(raw[subject_col]).dropna())
            unique_subjects = len(raw_subjects)
            compare_subjects = set(normalize_id(cohort["RID"]).dropna()) if subject_col == "RID" else cohort_ptids
            overlap = len(raw_subjects.intersection(compare_subjects))
        else:
            raw_subjects = set()

        sync_rows.append(
            {
                "filename": path.name,
                "subject_identifier": subject_col or "",
                "total_subjects_in_file": unique_subjects,
                "overlap_count": overlap,
                "overlap_percent": 100 * overlap / cohort_n if cohort_n else 0,
                "subjects_missing_from_cohort": max(unique_subjects - overlap, 0),
                "cohort_subjects_missing_from_file": cohort_n - overlap,
            }
        )

        candidate_cols = []
        for col in columns:
            if is_candidate_feature(path.name, col, raw[col]):
                candidate_cols.append(col)

        inventory_rows.append(
            {
                "filename": path.name,
                "n_rows": len(raw),
                "n_columns": len(columns),
                "n_unique_subjects": unique_subjects,
                "subject_identifier_columns": clean_list([col for col in SUBJECT_COLUMNS if col in columns]),
                "date_columns": clean_list(date_cols),
                "visit_columns": clean_list(visit_cols),
                "candidate_feature_columns": clean_list(candidate_cols, limit=250),
                "diagnosis_related_columns": clean_list(diagnosis_cols),
                "imaging_related_columns": clean_list(sorted(set(category_cols["MRI"] + category_cols["PET"] + category_cols["DTI"])), limit=250),
                "cognitive_related_columns": clean_list(category_cols["Cognitive"], limit=250),
                "demographic_related_columns": clean_list(category_cols["Demographic"], limit=250),
                "medical_history_columns": clean_list(category_cols["MedicalHistory"], limit=250),
                "laboratory_columns": clean_list(category_cols["Laboratory"], limit=250),
                "genetic_columns": clean_list(category_cols["Genetic"], limit=250),
                "mri_pet_dti_related_columns": clean_list(sorted(set(category_cols["MRI"] + category_cols["PET"] + category_cols["DTI"])), limit=250),
            }
        )

        if not subject_col or overlap == 0 or not candidate_cols:
            for col in candidate_cols:
                status, reason = leakage_status(path.name, col)
                leakage_rows.append({"feature_name": col, "source_file": path.name, "leakage_status": status, "reason": reason})
            continue

        selected, _ = baseline_extract_file(cohort, raw, subject_col)
        block = cohort[["PTID"]].copy()
        prefix = source_prefix(path.name)
        for col in candidate_cols:
            if col not in selected.columns:
                continue
            out_col = f"{prefix}_{sanitize_name(col)}"
            values = selected[col]
            dtype = infer_data_type(values)
            if dtype == "numeric_like":
                values = pd.to_numeric(values, errors="coerce")
                dtype = "numeric"
            if dtype == "text_or_high_cardinality":
                continue

            available = int(values.notna().sum())
            missing = cohort_n - available
            coverage = 100 * available / cohort_n if cohort_n else 0
            missing_percent = 100 * missing / cohort_n if cohort_n else 100
            category = infer_category(path.name, col)
            status, leak_reason = leakage_status(path.name, col)
            score = recommendation_score(category, coverage, missing_percent, status, out_col)

            catalog_rows.append(
                {
                    "feature_name": out_col,
                    "raw_feature_name": col,
                    "source_file": path.name,
                    "category": category,
                    "description": feature_description(path.name, col, category),
                    "data_type": dtype,
                }
            )
            coverage_rows.append(
                {
                    "feature_name": out_col,
                    "source_file": path.name,
                    "category": category,
                    "available_subjects": available,
                    "missing_subjects": missing,
                    "coverage_percent": coverage,
                    "missing_count": missing,
                    "missing_percent": missing_percent,
                    "missingness_class": missing_class(missing_percent),
                    "extraction_method_counts": selected["_extraction_method"].value_counts(dropna=False).to_dict(),
                }
            )
            leakage_rows.append(
                {
                    "feature_name": out_col,
                    "source_file": path.name,
                    "leakage_status": status,
                    "reason": leak_reason,
                }
            )
            block[out_col] = values.values

        if block.shape[1] > 1:
            feature_blocks.append(block)

    inventory = pd.DataFrame(inventory_rows)
    catalog = pd.DataFrame(catalog_rows)
    coverage = pd.DataFrame(coverage_rows)
    sync = pd.DataFrame(sync_rows)
    leakage = pd.DataFrame(leakage_rows)

    inventory.to_csv(DATA_INVENTORY_PATH, index=False)
    catalog.to_csv(CANDIDATE_CATALOG_PATH, index=False)
    coverage.to_csv(FEATURE_COVERAGE_PATH, index=False)
    sync.to_csv(SUBJECT_SYNC_PATH, index=False)
    leakage.to_csv(LEAKAGE_AUDIT_PATH, index=False)
    coverage.to_csv(MISSINGNESS_PATH, index=False)

    if not coverage.empty:
        recommended = coverage.merge(leakage[["feature_name", "leakage_status"]], on="feature_name", how="left")
        recommended["recommendation_score"] = recommended.apply(
            lambda r: recommendation_score(
                r["category"],
                r["coverage_percent"],
                r["missing_percent"],
                r["leakage_status"],
                r["feature_name"],
            ),
            axis=1,
        )
        recommended = recommended[
            (recommended["leakage_status"] == "SAFE")
            & (recommended["coverage_percent"] >= 20)
            & (recommended["recommendation_score"] >= 55)
        ].copy()
        recommended = recommended.sort_values(["recommendation_score", "coverage_percent"], ascending=False)
    else:
        recommended = pd.DataFrame()
    recommended.to_csv(RECOMMENDED_FEATURES_PATH, index=False)

    final = feature_blocks[0].copy()
    for block in feature_blocks[1:]:
        final = final.merge(block, on="PTID", how="left")

    selected_features = recommended["feature_name"].tolist() if not recommended.empty else []
    selected_features = [col for col in selected_features if col in final.columns]
    final_dataset = final[["PTID", "conversion_label"] + selected_features].copy()
    final_dataset.to_csv(FINAL_DATASET_PATH, index=False)

    overall_missing = final_dataset[selected_features].isna().mean().mean() * 100 if selected_features else 0
    lines = [
        "Final strict MCI-to-AD publication feature discovery report",
        "",
        f"Final subjects: {len(final_dataset)}",
        f"Converters: {int((final_dataset['conversion_label'] == 1).sum())}",
        f"Stable MCI: {int((final_dataset['conversion_label'] == 0).sum())}",
        f"Final selected features: {len(selected_features)}",
        f"Overall selected-feature missingness: {overall_missing:.2f}%",
        "",
        "Best features by category:",
    ]
    for category in ["Cognitive", "MRI", "PET", "DTI", "Demographic", "Genetic", "MedicalHistory", "Laboratory", "Vitals"]:
        top = recommended[recommended["category"] == category].head(15) if not recommended.empty else pd.DataFrame()
        lines.append(f"\n{category}:")
        if top.empty:
            lines.append("  None selected by current safety/coverage criteria.")
        else:
            for row in top.itertuples(index=False):
                lines.append(f"  {row.feature_name} | coverage={row.coverage_percent:.1f}% | score={row.recommendation_score:.1f}")
    lines.extend(
        [
            "",
            "Features likely to improve AUC beyond the current model include high-coverage cognitive/functional measures, APOE/genetic variables if available, amyloid PET SUVR/Centiloids, MRI regional volumes/thickness, and validated plasma/CSF biomarkers. All require fold-internal imputation and external validation.",
        ]
    )
    FINAL_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    log("\n====================================================")
    log("FINAL DATASET BUILD COMPLETE")
    log("====================================================")
    log(f"Files audited: {len(files)}")
    log(f"Candidate features cataloged: {len(catalog)}")
    log(f"Recommended features: {len(selected_features)}")
    log(f"Final dataset subjects: {len(final_dataset)}")
    log(f"Final dataset path: {FINAL_DATASET_PATH}")
    log(f"Final report path: {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
