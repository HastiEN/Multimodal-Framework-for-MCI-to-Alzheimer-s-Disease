"""Scan diagnosis tables and construct the original conversion labels."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import re
from pathlib import Path

import pandas as pd


RAW_ALL_DIR = _paths.project_path('data/raw/ALL')
PROCESSED_DIR = _paths.project_path('data/processed')
REPORT_PATH = PROCESSED_DIR / "csv_label_candidate_report.csv"
LABEL_PATH = PROCESSED_DIR / "mci_to_ad_labels_from_all_csv_scan.csv"
BASELINE_COMPATIBLE_VISITS = {"bl", "sc"}

SUBJECT_COLUMNS = ["PTID", "RID", "subject_id", "Subject ID", "SUBJECT", "LONIUID"]
VISIT_COLUMNS = ["VISCODE", "VISCODE2", "EXAMDATE", "VISDATE", "SCANDATE", "USERDATE", "Study Date", "EXAM_DATE"]
DATE_COLUMNS = ["EXAMDATE", "VISDATE", "SCANDATE", "Study Date", "USERDATE", "USERDATE2", "update_stamp"]
DIAGNOSIS_COLUMNS = [
    "DIAGNOSIS",
    "DXCHANGE",
    "DXCURREN",
    "DXNORM",
    "DXMCI",
    "DXAD",
    "DXNODEP",
    "DXCONFID",
    "DX_bl",
    "DX",
    "Research Group",
]
BEST_DIAGNOSIS_PRIORITY = ["DIAGNOSIS", "DXCHANGE", "DXCURREN", "DX", "Research Group", "DX_bl"]


def log(message: str) -> None:
    print(message, flush=True)


def read_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0, low_memory=False).columns.tolist()


def read_sample(path: Path, nrows: int = 8) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, low_memory=False)


def get_shape(path: Path) -> tuple[int, int]:
    # Reads only the first column to count rows while preserving a true row count.
    one_col = pd.read_csv(path, usecols=[0], low_memory=False)
    n_cols = len(read_header(path))
    return int(len(one_col)), int(n_cols)


def present(columns: list[str], candidates: list[str]) -> list[str]:
    lower_lookup = {column.lower(): column for column in columns}
    found = []
    for candidate in candidates:
        if candidate.lower() in lower_lookup:
            found.append(lower_lookup[candidate.lower()])
    return found


def diagnosis_like_columns(columns: list[str]) -> list[str]:
    direct = present(columns, DIAGNOSIS_COLUMNS)
    pattern_cols = [
        col
        for col in columns
        if re.search(r"(diag|diagnosis|dx|dementia|mci|ad\b|research group)", col, flags=re.IGNORECASE)
    ]
    return sorted(set(direct + pattern_cols), key=lambda x: columns.index(x))


def sample_values(df: pd.DataFrame, columns: list[str], max_values: int = 8) -> dict[str, str]:
    out = {}
    for col in columns:
        if col not in df.columns:
            continue
        values = df[col].dropna().astype(str).unique().tolist()[:max_values]
        out[col] = " | ".join(values)
    return out


def score_candidate(file_name: str, columns: list[str], sample: pd.DataFrame, n_rows: int) -> tuple[int, bool, str]:
    subject_cols = present(columns, SUBJECT_COLUMNS)
    visit_cols = present(columns, VISIT_COLUMNS)
    date_cols = present(columns, DATE_COLUMNS)
    dx_cols = diagnosis_like_columns(columns)
    score = 0
    reasons = []

    if "PTID" in subject_cols:
        score += 3
    if "RID" in subject_cols:
        score += 3
    if subject_cols:
        score += 2
    if date_cols:
        score += 3
    if any(col in visit_cols for col in ["VISCODE", "VISCODE2"]):
        score += 2
    if "DIAGNOSIS" in dx_cols:
        score += 8
    if "DXCHANGE" in dx_cols:
        score += 6
    if "DXCURREN" in dx_cols:
        score += 5
    if dx_cols:
        score += 4
    if n_rows > 100:
        score += 1

    name_upper = file_name.upper()
    if "DXSUM" in name_upper:
        score += 10
        reasons.append("File name indicates diagnosis summary.")
    if "MY_TABLE" in name_upper:
        score += 5
        reasons.append("File name may contain longitudinal diagnosis export.")
    if any(token in name_upper for token in ["ADAS", "MMSE", "CDR", "PET", "MRI", "BIOMARK", "LAB", "VITAL", "APOE", "GENETIC"]):
        score -= 4
        reasons.append("File name suggests feature/biomarker table rather than diagnosis-label table.")

    good = bool(subject_cols and dx_cols and (date_cols or visit_cols))
    if "DIAGNOSIS" in dx_cols and "PTID" in subject_cols and "RID" in subject_cols:
        reasons.append("Contains PTID/RID plus explicit DIAGNOSIS column.")
    elif dx_cols:
        reasons.append("Contains diagnosis-like columns but may need coding interpretation.")
    else:
        reasons.append("No diagnosis-like columns found.")

    if not subject_cols:
        reasons.append("No clear subject identifier.")
    if not (date_cols or visit_cols):
        reasons.append("No clear visit/date column.")
    if not good:
        reasons.append("Not suitable for label construction.")

    return score, good, " ".join(reasons)


def scan_all_csvs() -> pd.DataFrame:
    rows = []
    csv_paths = sorted(RAW_ALL_DIR.glob("*.csv"))
    log(f"Scanning {len(csv_paths)} CSV files in {RAW_ALL_DIR}")

    for idx, path in enumerate(csv_paths, start=1):
        log(f"[{idx}/{len(csv_paths)}] Inspecting {path.name}")
        try:
            columns = read_header(path)
            sample = read_sample(path)
            n_rows, n_cols = get_shape(path)
            subject_cols = present(columns, SUBJECT_COLUMNS)
            visit_cols = present(columns, VISIT_COLUMNS)
            dx_cols = diagnosis_like_columns(columns)
            dx_values = sample_values(sample, dx_cols)
            score, good, reason = score_candidate(path.name, columns, sample, n_rows)
            rows.append(
                {
                    "file_name": path.name,
                    "n_rows": n_rows,
                    "n_columns": n_cols,
                    "shape": f"({n_rows}, {n_cols})",
                    "column_names": json.dumps(columns, ensure_ascii=False),
                    "subject_id_columns": ", ".join(subject_cols),
                    "visit_date_columns": ", ".join(visit_cols),
                    "diagnosis_related_columns": ", ".join(dx_cols),
                    "diagnosis_sample_values": json.dumps(dx_values, ensure_ascii=False),
                    "is_good_label_candidate": good,
                    "candidate_score": score,
                    "reason": reason,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "file_name": path.name,
                    "n_rows": None,
                    "n_columns": None,
                    "shape": "ERROR",
                    "column_names": "[]",
                    "subject_id_columns": "",
                    "visit_date_columns": "",
                    "diagnosis_related_columns": "",
                    "diagnosis_sample_values": "{}",
                    "is_good_label_candidate": False,
                    "candidate_score": -999,
                    "reason": f"Failed to inspect file: {exc}",
                }
            )

    report = pd.DataFrame(rows).sort_values(["candidate_score", "file_name"], ascending=[False, True])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(REPORT_PATH, index=False)
    return report


def choose_best_label_file(report: pd.DataFrame) -> Path:
    good = report[report["is_good_label_candidate"] == True].copy()
    if good.empty:
        raise RuntimeError("No good label candidate files were found.")
    best_name = good.iloc[0]["file_name"]
    return RAW_ALL_DIR / best_name


def visit_order_value(value: object) -> int:
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


def normalized_visit_name(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_baseline_compatible_visit(value: object) -> bool:
    return normalized_visit_name(value) in BASELINE_COMPATIBLE_VISITS


def normalize_diagnosis(value: object, source_col: str) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]

    if source_col.upper() == "DIAGNOSIS":
        if pd.notna(numeric):
            code = int(numeric)
            if code == 1:
                return "CN"
            if code == 2:
                return "MCI"
            if code == 3:
                return "AD"
        if "MCI" in text:
            return "MCI"
        if text in {"AD", "DEMENTIA"} or "ALZ" in text:
            return "AD"
        if text in {"CN", "NL", "NORMAL"}:
            return "CN"

    if source_col.upper() == "DXCHANGE":
        if pd.notna(numeric):
            code = int(numeric)
            if code in {1, 7, 9}:
                return "CN"
            if code in {2, 4, 8}:
                return "MCI"
            if code in {3, 5, 6}:
                return "AD"

    if source_col.upper() in {"DXCURREN", "DX", "DX_BL", "RESEARCH GROUP"}:
        if pd.notna(numeric):
            code = int(numeric)
            if code == 1:
                return "CN"
            if code == 2:
                return "MCI"
            if code == 3:
                return "AD"
        if "MCI" in text or "EMCI" in text or "LMCI" in text:
            return "MCI"
        if text in {"AD", "DEMENTIA"} or "ALZ" in text:
            return "AD"
        if text in {"CN", "NL", "NORMAL", "SMC"}:
            return "CN"

    return None


def choose_diagnosis_column(columns: list[str]) -> str:
    available = diagnosis_like_columns(columns)
    for col in BEST_DIAGNOSIS_PRIORITY:
        for actual in available:
            if actual.lower() == col.lower():
                return actual
    raise ValueError(f"No supported diagnosis column found. Diagnosis-like columns: {available}")


def build_labels_from_file(path: Path) -> pd.DataFrame:
    log(f"Building labels from selected file: {path.name}")
    df = pd.read_csv(path, low_memory=False)
    columns = df.columns.tolist()
    require = ["PTID", "RID"]
    missing = [col for col in require if col not in df.columns]
    if missing:
        raise ValueError(f"Selected label file must contain PTID and RID. Missing: {missing}")

    dx_col = choose_diagnosis_column(columns)
    log(f"Using diagnosis column: {dx_col}")

    date_col = None
    for candidate in ["EXAMDATE", "VISDATE", "USERDATE", "Study Date"]:
        if candidate in df.columns:
            date_col = candidate
            break
    visit_col = "VISCODE2" if "VISCODE2" in df.columns else ("VISCODE" if "VISCODE" in df.columns else None)

    label_df = df.copy()
    label_df["diagnosis_norm"] = label_df[dx_col].apply(lambda x: normalize_diagnosis(x, dx_col))
    label_df = label_df.dropna(subset=["PTID", "RID", "diagnosis_norm"])

    if date_col:
        label_df["visit_date"] = pd.to_datetime(label_df[date_col], errors="coerce")
    else:
        label_df["visit_date"] = pd.NaT

    if visit_col:
        label_df["visit_order"] = label_df[visit_col].apply(visit_order_value)
        label_df["visit_name"] = label_df[visit_col]
    else:
        label_df["visit_order"] = range(len(label_df))
        label_df["visit_name"] = pd.NA

    label_df["sort_date_missing"] = label_df["visit_date"].isna().astype(int)
    label_df = label_df.sort_values(["PTID", "RID", "sort_date_missing", "visit_date", "visit_order"])

    rows = []
    baseline_mci_count = 0
    no_followup_count = 0
    no_baseline_compatible_visit_count = 0
    for (ptid, rid), group in label_df.groupby(["PTID", "RID"], sort=False):
        group = group.sort_values(["sort_date_missing", "visit_date", "visit_order"]).reset_index(drop=True)
        baseline_candidates = group[group["visit_name"].apply(is_baseline_compatible_visit)].copy()
        if baseline_candidates.empty:
            no_baseline_compatible_visit_count += 1
            continue

        baseline = baseline_candidates.sort_values(
            ["sort_date_missing", "visit_date", "visit_order"]
        ).iloc[0]
        if baseline["diagnosis_norm"] != "MCI":
            continue
        baseline_mci_count += 1

        future = group.drop(index=baseline.name).copy()
        if pd.notna(baseline["visit_date"]):
            future = future[
                (future["visit_date"].isna())
                | (future["visit_date"] > baseline["visit_date"])
                | (future["visit_order"] > baseline["visit_order"])
            ].copy()
        else:
            future = future[future["visit_order"] > baseline["visit_order"]].copy()
        future = future.sort_values(["sort_date_missing", "visit_date", "visit_order"])
        if future.empty:
            no_followup_count += 1
            continue

        ad_future = future[future["diagnosis_norm"] == "AD"]
        conversion_label = int(not ad_future.empty)
        conversion_date = pd.NaT
        months_to_conversion = pd.NA
        if conversion_label:
            first_ad = ad_future.iloc[0]
            conversion_date = first_ad["visit_date"]
            if pd.notna(conversion_date) and pd.notna(baseline["visit_date"]):
                months_to_conversion = (conversion_date - baseline["visit_date"]).days / 30.4375

        last = group.iloc[-1]
        rows.append(
            {
                "PTID": ptid,
                "RID": rid,
                "baseline_date": baseline["visit_date"],
                "baseline_visit": baseline["visit_name"],
                "baseline_diagnosis": baseline["diagnosis_norm"],
                "final_diagnosis": last["diagnosis_norm"],
                "conversion_label": conversion_label,
                "conversion_date": conversion_date,
                "months_to_conversion": months_to_conversion,
                "n_followup_visits": int(len(future)),
                "last_followup_date": last["visit_date"],
            }
        )

    labels = pd.DataFrame(rows)
    if not labels.empty:
        labels = labels.sort_values(["PTID", "RID"]).reset_index(drop=True)
    labels.to_csv(LABEL_PATH, index=False)

    log(f"Excluded subjects without bl/sc diagnosis visit: {no_baseline_compatible_visit_count}")
    log(f"Baseline MCI subjects before follow-up exclusion: {baseline_mci_count}")
    log(f"Excluded no-follow-up baseline MCI subjects: {no_followup_count}")
    log(f"Final labeled subjects: {len(labels)}")
    if not labels.empty:
        log(f"Converters: {int(labels['conversion_label'].sum())}")
        log(f"Stable/non-AD follow-up: {int((labels['conversion_label'] == 0).sum())}")
    return labels


def print_ranked_candidates(report: pd.DataFrame, top_n: int = 15) -> None:
    cols = [
        "file_name",
        "shape",
        "subject_id_columns",
        "visit_date_columns",
        "diagnosis_related_columns",
        "is_good_label_candidate",
        "candidate_score",
        "reason",
    ]
    log("\nBest candidate files for label construction:")
    print(report[cols].head(top_n).to_string(index=False), flush=True)


def main() -> None:
    if not RAW_ALL_DIR.exists():
        raise FileNotFoundError(RAW_ALL_DIR)
    report = scan_all_csvs()
    log(f"\nSaved scan report to: {REPORT_PATH}")
    print_ranked_candidates(report)

    best_path = choose_best_label_file(report)
    log(f"\nSelected best label file: {best_path.name}")
    labels = build_labels_from_file(best_path)
    log(f"Saved labels to: {LABEL_PATH}")
    log("\nDone.")


if __name__ == "__main__":
    main()
