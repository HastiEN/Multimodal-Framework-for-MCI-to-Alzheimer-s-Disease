"""Reproduce the full-cohort consensus ranking and fixed Top-k lists."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import re

import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
REQUESTED_DATA_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
FALLBACK_DATA_PATH = PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
DATA_PATH = REQUESTED_DATA_PATH if REQUESTED_DATA_PATH.exists() else FALLBACK_DATA_PATH

OUT_DIR = PROCESSED_DIR / "publication_feature_ranking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REMOVED_METADATA_PATH = OUT_DIR / "removed_metadata_features.csv"
REMOVED_MISSING_PATH = OUT_DIR / "removed_high_missingness.csv"
REMOVED_VARIANCE_PATH = OUT_DIR / "removed_low_variance.csv"
REMOVED_CORRELATION_PATH = OUT_DIR / "removed_high_correlation.csv"
AUDIT_PATH = OUT_DIR / "feature_audit_summary.csv"
UNIVARIATE_PATH = OUT_DIR / "univariate_feature_ranking.csv"
MI_PATH = OUT_DIR / "mutual_information_ranking.csv"
XGB_PATH = OUT_DIR / "xgboost_feature_importance.csv"
SHAP_PATH = OUT_DIR / "shap_feature_importance.csv"
CONSENSUS_PATH = OUT_DIR / "consensus_feature_ranking.csv"
TOP20_PATH = OUT_DIR / "top20_features.csv"
TOP50_PATH = OUT_DIR / "top50_features.csv"
TOP100_PATH = OUT_DIR / "top100_features.csv"
REPORT_PATH = OUT_DIR / "feature_ranking_report.txt"

RANDOM_STATE = 42
MISSING_THRESHOLD = 40.0
VARIANCE_THRESHOLD = 0.001
CORRELATION_THRESHOLD = 0.95

METADATA_PATTERNS = [
    r"^PTID$",
    r"^RID$",
    r"IMAGEUID",
    r"IMAGE_ID",
    r"LONIUID",
    r"EXAMDATE",
    r"SCANDATE",
    r"VISCODE",
    r"VISCODE2",
    r"VISIT",
    r"FIELD_STRENGTH",
    r"FSVER",
    r"IMAGE_RESOLUTION",
    r"MRFIELD",
    r"MRSEQUENCE",
    r"SCANNER",
    r"ACQUISITION",
    r"PROCESS",
    r"VERSION",
    r"SOFTWARE",
    r"FILENAME",
    r"FILE",
    r"SITEID",
    r"RECNO",
    r"USERDATE",
    r"UPDATE_STAMP",
    r"PHASE",
    r"LANGUAGE_CODE",
    r"QC",
    r"UNIT$",
    r"_UNIT",
    r"VSTMPUNT",
    r"VSWTUNIT",
    r"VSHTUNIT",
    r"MMDATE",
    r"MMYEAR",
    r"MMMONTH",
    r"MMDAY",
]

LEAKAGE_PATTERNS = [
    r"DIAG",
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


def target_column(df: pd.DataFrame) -> str:
    for col in ["label", "conversion_label", "converter_label", "converter"]:
        if col in df.columns:
            return col
    raise ValueError("No target column found. Expected label or conversion_label.")


def is_metadata(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in METADATA_PATTERNS)


def is_leakage_name(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def category(feature: str) -> str:
    upper = feature.upper()
    if any(k in upper for k in ["MMSE", "MMSCORE", "ADAS", "TOTSCORE", "TOTAL13", "CDR", "CDRSB", "FAQ", "MOCA", "NEUROBAT", "ECOG"]):
        return "Cognitive/Clinical"
    if any(k in upper for k in ["UCSFFSX", "MRI", "HIPPOCAMPUS", "ENTORHINAL", "VENTRICLE", "CORTEX", "THICK", "VOLUME"]):
        return "MRI"
    if any(k in upper for k in ["AMY", "PET", "SUVR", "CENTILOID", "TAU_6MM", "FDG"]):
        return "PET"
    if any(k in upper for k in ["DTI", "FA", "MD", "RD", "DIFFUSION"]):
        return "DTI"
    if any(k in upper for k in ["PTDEMOG", "PTEDUCAT", "PTGENDER", "SEX", "AGE", "RACE", "ETH", "MARRY"]):
        return "Demographic"
    if any(k in upper for k in ["APOE", "GENOTYPE", "GENETIC"]):
        return "Genetic"
    if any(k in upper for k in ["LAB", "PLASMA", "CSF", "PTAU", "TAU", "ABETA", "BIOMARK", "ELECSYS", "QUANTERIX"]):
        return "Biomarker/Lab"
    if any(k in upper for k in ["VITALS", "VSWEIGHT", "VSBP", "BMI", "PULSE", "TEMP"]):
        return "Vitals"
    return "Other"


def feature_types(x: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    binary = []
    for col in x.columns:
        vals = set(x[col].dropna().unique())
        if 1 <= len(vals) <= 2:
            binary.append(col)
    return numeric, categorical, binary


def coerce_numeric_like(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = pd.to_numeric(out[col], errors="coerce")
            non_missing = out[col].notna().sum()
            if non_missing and numeric.notna().sum() / non_missing >= 0.8:
                out[col] = numeric
    return out


def remove_metadata_and_leakage(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keep = []
    for col in x.columns:
        if is_metadata(col):
            rows.append({"feature": col, "reason": "identifier/technical/admin metadata"})
        elif is_leakage_name(col):
            rows.append({"feature": col, "reason": "diagnosis/conversion/follow-up leakage name"})
        else:
            keep.append(col)
    pd.DataFrame(rows).to_csv(REMOVED_METADATA_PATH, index=False)
    return x[keep].copy(), pd.DataFrame(rows)


def missingness_filter(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    miss = x.isna().mean() * 100
    remove = miss[miss > MISSING_THRESHOLD].sort_values(ascending=False)
    pd.DataFrame({"feature": remove.index, "missing_percent": remove.values}).to_csv(REMOVED_MISSING_PATH, index=False)
    return x.drop(columns=remove.index), pd.DataFrame({"feature": remove.index, "missing_percent": remove.values})


def variance_filter(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    keep = []
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            vals = pd.to_numeric(x[col], errors="coerce")
            var = vals.var(skipna=True)
            unique = vals.dropna().nunique()
            if unique <= 1 or (pd.notna(var) and var < VARIANCE_THRESHOLD):
                rows.append({"feature": col, "variance": var, "unique_values": unique, "reason": "constant_or_low_variance"})
            else:
                keep.append(col)
        else:
            unique = x[col].dropna().nunique()
            if unique <= 1:
                rows.append({"feature": col, "variance": np.nan, "unique_values": unique, "reason": "constant_categorical"})
            else:
                keep.append(col)
    pd.DataFrame(rows).to_csv(REMOVED_VARIANCE_PATH, index=False)
    return x[keep].copy(), pd.DataFrame(rows)


def bio_score(feature: str) -> int:
    upper = feature.upper()
    return sum(1 for k in BIO_PRIORITY if k in upper)


def correlation_filter(x: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric) < 2:
        pd.DataFrame().to_csv(REMOVED_CORRELATION_PATH, index=False)
        return x, pd.DataFrame()
    imputed = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x[numeric]), columns=numeric)
    corr = imputed.corr().abs()
    missing = x[numeric].isna().mean() * 100
    coverage = 100 - missing
    removed = set()
    rows = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        if a in removed:
            continue
        for b in cols[i + 1 :]:
            if b in removed:
                continue
            if corr.loc[a, b] > CORRELATION_THRESHOLD:
                score_a = (coverage[a], -missing[a], bio_score(a))
                score_b = (coverage[b], -missing[b], bio_score(b))
                keep, drop = (a, b) if score_a >= score_b else (b, a)
                removed.add(drop)
                rows.append({"removed_feature": drop, "kept_feature": keep, "abs_correlation": corr.loc[a, b]})
    pd.DataFrame(rows).to_csv(REMOVED_CORRELATION_PATH, index=False)
    return x.drop(columns=list(removed)), pd.DataFrame(rows)


def univariate_ranking(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    rows = []
    y0 = y == 0
    y1 = y == 1
    for col in x.columns:
        if pd.api.types.is_numeric_dtype(x[col]):
            a = pd.to_numeric(x.loc[y0, col], errors="coerce").dropna()
            b = pd.to_numeric(x.loc[y1, col], errors="coerce").dropna()
            if len(a) < 3 or len(b) < 3:
                p = np.nan
                effect = np.nan
            else:
                _, p_t = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                _, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
                pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
                effect = (b.mean() - a.mean()) / pooled if pooled and not np.isnan(pooled) else np.nan
                p = np.nanmin([p_t, p_u])
            rows.append({"feature": col, "p_value": p, "effect_size": effect, "test": "t_test_mannwhitney"})
        else:
            tab = pd.crosstab(x[col].fillna("MISSING"), y)
            try:
                if tab.shape == (2, 2):
                    _, p = stats.fisher_exact(tab)
                else:
                    _, p, _, _ = stats.chi2_contingency(tab)
            except Exception:
                p = np.nan
            effect = np.nan
            rows.append({"feature": col, "p_value": p, "effect_size": effect, "test": "fisher_or_chi_square"})
    out = pd.DataFrame(rows).sort_values(["p_value", "feature"], na_position="last").reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.to_csv(UNIVARIATE_PATH, index=False)
    return out


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [c for c in x.columns if c not in numeric]
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transformed_to_original(names: list[str], original: list[str], categorical: list[str]) -> dict[str, str]:
    mapping = {}
    for name in names:
        if name in original:
            mapping[name] = name
        else:
            match = next((c for c in sorted(categorical, key=len, reverse=True) if name.startswith(f"{c}_")), name)
            mapping[name] = match
    return mapping


def transformed_matrix(x: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    pre = make_preprocessor(x)
    arr = pre.fit_transform(x)
    names = pre.get_feature_names_out().tolist()
    categorical = [c for c in x.columns if c not in x.select_dtypes(include=[np.number]).columns]
    return pd.DataFrame(arr, columns=names), names, categorical


def mutual_info_ranking(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    xt, names, categorical = transformed_matrix(x)
    discrete = np.array([name not in x.columns for name in names], dtype=bool)
    scores = mutual_info_classif(xt, y, discrete_features=discrete, random_state=RANDOM_STATE)
    mapping = transformed_to_original(names, x.columns.tolist(), categorical)
    raw = pd.DataFrame({"transformed_feature": names, "feature": [mapping[n] for n in names], "mi_score": scores})
    out = raw.groupby("feature", as_index=False)["mi_score"].sum().sort_values("mi_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.to_csv(MI_PATH, index=False)
    return out


def xgb_cv_importance(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)
    importance_accumulator: dict[str, list[float]] = {}
    aucs = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        pre = make_preprocessor(x_train)
        xt_train = pre.fit_transform(x_train)
        xt_test = pre.transform(x_test)
        names = pre.get_feature_names_out().tolist()
        categorical = [c for c in x_train.columns if c not in x_train.select_dtypes(include=[np.number]).columns]
        mapping = transformed_to_original(names, x_train.columns.tolist(), categorical)
        model = XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1,
        )
        model.fit(xt_train, y_train)
        aucs.append(roc_auc_score(y_test, model.predict_proba(xt_test)[:, 1]))
        gain = model.get_booster().get_score(importance_type="gain")
        gain_by_original: dict[str, float] = {}
        for transformed, value in gain.items():
            original = mapping.get(transformed, transformed)
            gain_by_original[original] = gain_by_original.get(original, 0.0) + float(value)
        for feature in x.columns:
            importance_accumulator.setdefault(feature, []).append(gain_by_original.get(feature, 0.0))
        if fold % 5 == 0:
            log(f"  XGBoost importance fold {fold}/25, mean AUC so far={np.mean(aucs):.3f}")
    out = pd.DataFrame(
        {
            "feature": list(importance_accumulator),
            "mean_gain_importance": [np.mean(v) for v in importance_accumulator.values()],
            "std_gain_importance": [np.std(v, ddof=1) for v in importance_accumulator.values()],
        }
    ).sort_values("mean_gain_importance", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.to_csv(XGB_PATH, index=False)
    return out


def shap_ranking(x: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    pre = make_preprocessor(x)
    xt = pre.fit_transform(x)
    names = pre.get_feature_names_out().tolist()
    categorical = [c for c in x.columns if c not in x.select_dtypes(include=[np.number]).columns]
    mapping = transformed_to_original(names, x.columns.tolist(), categorical)
    model = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        n_jobs=-1,
    )
    model.fit(xt, y)
    xt_df = pd.DataFrame(xt, columns=names)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(xt_df)
    if isinstance(values, list):
        values = values[1]
    raw = pd.DataFrame({"transformed_feature": names, "feature": [mapping[n] for n in names], "mean_abs_shap": np.abs(values).mean(axis=0)})
    out = raw.groupby("feature", as_index=False)["mean_abs_shap"].sum().sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    out.to_csv(SHAP_PATH, index=False)
    return out


def consensus(univ: pd.DataFrame, mi: pd.DataFrame, xgb: pd.DataFrame, sh: pd.DataFrame) -> pd.DataFrame:
    features = sorted(set(univ.feature) | set(mi.feature) | set(xgb.feature) | set(sh.feature))
    out = pd.DataFrame({"feature": features})
    out = out.merge(univ[["feature", "rank"]].rename(columns={"rank": "univariate_rank"}), on="feature", how="left")
    out = out.merge(mi[["feature", "rank"]].rename(columns={"rank": "mi_rank"}), on="feature", how="left")
    out = out.merge(xgb[["feature", "rank"]].rename(columns={"rank": "xgb_rank"}), on="feature", how="left")
    out = out.merge(sh[["feature", "rank"]].rename(columns={"rank": "shap_rank"}), on="feature", how="left")
    max_rank = len(features) + 1
    rank_cols = ["univariate_rank", "mi_rank", "xgb_rank", "shap_rank"]
    out[rank_cols] = out[rank_cols].fillna(max_rank)
    out["average_rank"] = out[rank_cols].mean(axis=1)
    out = out.sort_values("average_rank").reset_index(drop=True)
    out["consensus_rank"] = np.arange(1, len(out) + 1)
    out = out[["feature", "univariate_rank", "mi_rank", "xgb_rank", "shap_rank", "average_rank", "consensus_rank"]]
    out.to_csv(CONSENSUS_PATH, index=False)
    out.head(20).to_csv(TOP20_PATH, index=False)
    out.head(50).to_csv(TOP50_PATH, index=False)
    out.head(100).to_csv(TOP100_PATH, index=False)
    return out


def write_report(df: pd.DataFrame, x_filtered: pd.DataFrame, con: pd.DataFrame) -> None:
    lines = [
        "Publication Feature Ranking Report",
        "",
        f"Dataset: {DATA_PATH}",
        f"Subjects: {len(df)}",
        f"Final ranked features: {x_filtered.shape[1]}",
        "",
    ]
    for cat in ["Cognitive/Clinical", "MRI", "PET", "DTI", "Biomarker/Lab", "Demographic", "Genetic", "Vitals", "Other"]:
        top = con[con["feature"].apply(category).eq(cat)].head(15)
        lines.append(f"Top {cat} Features:")
        if top.empty:
            lines.append("  None")
        else:
            for row in top.itertuples(index=False):
                lines.append(f"  {int(row.consensus_rank)}. {row.feature} (avg rank={row.average_rank:.1f})")
        lines.append("")
    highlights = {
        "CDRSB": "CDRSB",
        "ADAS13": "TOTAL13",
        "MMSE": "MMSCORE",
        "FAQ": "FAQ",
        "Hippocampus Volume": "HIPPOCAMPUS",
        "Entorhinal Volume": "ENTORHINAL",
        "Amyloid Centiloids": "CENTILOIDS",
        "Amyloid Summary SUVR": "SUMMARY_SUVR",
        "pTau": "PTAU",
        "Tau": "TAU",
        "APOE4": "APOE",
    }
    lines.append("Requested Feature Highlight Check:")
    for label, token in highlights.items():
        matches = con[con["feature"].str.upper().str.contains(token)]
        if matches.empty:
            lines.append(f"  {label}: not found after filters")
        else:
            best = matches.iloc[0]
            lines.append(f"  {label}: {best['feature']} ranked {int(best['consensus_rank'])}")
    lines.extend(
        [
            "",
            "Interpretation:",
            "The smallest biologically meaningful subset should prioritize consensus-ranked cognitive/clinical variables, amyloid PET measures, MRI neurodegeneration markers, validated lab biomarkers, demographics such as education/sex, and APOE/genetic markers where available. Avoid administrative, acquisition, processing, and diagnosis/change fields.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    target = target_column(df)
    y = df[target].astype(int)
    x = df.drop(columns=[target])
    x = coerce_numeric_like(x)

    numeric, categorical, binary = feature_types(x)
    audit = pd.DataFrame(
        [
            {"metric": "subjects", "value": len(df)},
            {"metric": "initial_features", "value": x.shape[1]},
            {"metric": "stable_mci", "value": int((y == 0).sum())},
            {"metric": "converter", "value": int((y == 1).sum())},
            {"metric": "numeric_features", "value": len(numeric)},
            {"metric": "categorical_features", "value": len(categorical)},
            {"metric": "binary_features", "value": len(binary)},
        ]
    )
    audit.to_csv(AUDIT_PATH, index=False)

    log("====================================================")
    log("FEATURE AUDIT")
    log("====================================================")
    log(audit.to_string(index=False))

    x, removed_meta = remove_metadata_and_leakage(x)
    x, removed_missing = missingness_filter(x)
    x, removed_var = variance_filter(x)
    x, removed_corr = correlation_filter(x)

    log(f"After metadata/leakage removal: {df.drop(columns=[target]).shape[1] - len(removed_meta)} candidates")
    log(f"After missingness filter: {x.shape[1] + len(removed_corr)} before correlation filter")
    log(f"After all filters: {x.shape[1]} features")

    log("Running univariate ranking...")
    univ = univariate_ranking(x, y)
    log("Running mutual information ranking...")
    mi = mutual_info_ranking(x, y)
    log("Running XGBoost repeated-CV gain importance...")
    xgb = xgb_cv_importance(x, y)
    log("Running SHAP feature importance...")
    sh = shap_ranking(x, y)
    con = consensus(univ, mi, xgb, sh)
    write_report(df, x, con)

    log("\nTop 30 consensus features:")
    log(con.head(30).to_string(index=False))
    log(f"\nSaved reports to: {OUT_DIR}")


if __name__ == "__main__":
    main()
