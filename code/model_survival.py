"""Reproduce historical survival results; metric limitations are documented."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
COHORT_PATH = (
    PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv"
    if (PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv").exists()
    else PROCESSED_DIR / "2-strict_mci_to_ad_publication_cohort.csv"
)
FEATURE_DATA_PATH = (
    PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
    if (PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv").exists()
    else PROCESSED_DIR / "3-final_mci_to_ad_publication_dataset.csv"
)
RANKING_DIR = PROCESSED_DIR / "publication_feature_ranking"
OUT_DIR = PROCESSED_DIR / "survival_analysis"
REPORTS_DIR = OUT_DIR / "reports"
PLOTS_DIR = OUT_DIR / "plots"
RESULTS_DIR = OUT_DIR / "results"
MODELS_DIR = OUT_DIR / "models"
for folder in [REPORTS_DIR, PLOTS_DIR, RESULTS_DIR, MODELS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

SURVIVAL_DATA_PATH = RESULTS_DIR / "adni_survival_dataset.csv"
AUDIT_PATH = REPORTS_DIR / "survival_dataset_audit.txt"
KM_PLOT_PATH = PLOTS_DIR / "km_survival_curves.png"
COX_IMPORTANCE_PATH = RESULTS_DIR / "cox_feature_importance.csv"
CV_RESULTS_PATH = RESULTS_DIR / "survival_cv_results.csv"
TD_AUC_PATH = RESULTS_DIR / "time_dependent_auc_table.csv"
BRIER_PATH = RESULTS_DIR / "ipcw_brier_score_table.csv"
CALIBRATION_PATH = RESULTS_DIR / "survival_calibration_table.csv"
CALIBRATION_PLOT_PATH = PLOTS_DIR / "survival_calibration_curves.png"
BIOMARKER_REPORT_PATH = REPORTS_DIR / "survival_biomarker_report.txt"
FINAL_REPORT_PATH = REPORTS_DIR / "survival_experiment_report.txt"
JSON_REPORT_PATH = REPORTS_DIR / "survival_experiment_report.json"

RANDOM_STATE = 42
TIME_HORIZONS = [24, 36, 60, 84]
TOP_FILES = {
    "Top20": RANKING_DIR / "top20_features.csv",
    "Top50": RANKING_DIR / "top50_features.csv",
    "Top100": RANKING_DIR / "top100_features.csv",
}
XGB_SURVIVAL_PARAMS = {
    "objective": "survival:cox",
    "n_estimators": 500,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}
LEAKAGE_COLUMNS = {
    "PTID",
    "RID",
    "label",
    "conversion_label",
    "baseline_date",
    "baseline_visit",
    "conversion_date",
    "months_to_conversion",
    "last_followup_date",
    "n_followup_visits",
    "time_to_event_months",
    "event",
}
LEAKAGE_PATTERNS = [
    r"DIAG",
    r"\bDX\b",
    r"CONVERT",
    r"CONVERSION",
    r"FOLLOWUP",
    r"FUTURE",
    r"LABEL",
    r"BASELINE_DATE",
    r"LAST_FOLLOWUP",
    r"MONTHS_TO_CONVERSION",
    r"VISCODE",
    r"VISIT",
    r"EXAMDATE",
    r"SCANDATE",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def months_between(later: pd.Series, earlier: pd.Series) -> pd.Series:
    return (later - earlier).dt.days / 30.4375


def is_leakage_feature(feature: str) -> bool:
    upper = feature.upper()
    if upper in LEAKAGE_COLUMNS:
        return True
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def load_feature_set(path: Path) -> list[str]:
    df = pd.read_csv(path)
    col = "feature" if "feature" in df.columns else df.columns[0]
    return [str(x) for x in df[col].dropna().tolist()]


def make_preprocessor(x: pd.DataFrame, scale: bool = True) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
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


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return fallback


def build_survival_dataset() -> pd.DataFrame:
    cohort = pd.read_csv(COHORT_PATH)
    features = pd.read_csv(FEATURE_DATA_PATH)
    required = ["PTID", "baseline_date", "last_followup_date", "conversion_label"]
    missing = [col for col in required if col not in cohort.columns]
    if missing:
        raise ValueError(f"Cohort is missing required survival columns: {missing}")

    cohort = cohort.copy()
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort.get("conversion_date"), errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype("Int64")
    cohort["exclude_from_time_to_conversion_analysis"] = (
        cohort.get("exclude_from_time_to_conversion_analysis", False).fillna(False).astype(bool)
        if "exclude_from_time_to_conversion_analysis" in cohort.columns
        else False
    )

    cohort["event"] = cohort["conversion_label"].astype(int)
    cohort.loc[cohort["exclude_from_time_to_conversion_analysis"], "event"] = np.nan
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
        for col in features.columns
        if col not in LEAKAGE_COLUMNS and not is_leakage_feature(col)
    ]
    merged = survival_core.merge(features[["PTID"] + feature_cols], on="PTID", how="left", validate="one_to_one")
    merged.to_csv(SURVIVAL_DATA_PATH, index=False)
    return merged


def write_survival_audit(df: pd.DataFrame) -> None:
    t = df["time_to_event_months"]
    event = df["event"].astype(int)
    converters = int(event.sum())
    censored = int((event == 0).sum())
    converted_times = df.loc[event == 1, "time_to_event_months"]
    lines = [
        "ADNI MCI-to-AD Survival Dataset Audit",
        "",
        f"Source cohort: {COHORT_PATH}",
        f"Source baseline feature table: {FEATURE_DATA_PATH}",
        f"Total subjects: {len(df)}",
        f"Converters/events: {converters}",
        f"Censored: {censored}",
        f"Event fraction: {converters / len(df):.3f}",
        f"Median follow-up/event time: {t.median():.2f} months",
        f"Minimum follow-up/event time: {t.min():.2f} months",
        f"Maximum follow-up/event time: {t.max():.2f} months",
        f"Median time-to-conversion among converters: {converted_times.median():.2f} months",
        f"Minimum time-to-conversion among converters: {converted_times.min():.2f} months",
        f"Maximum time-to-conversion among converters: {converted_times.max():.2f} months",
        "",
        "Leakage statement:",
        "Survival time and censoring were derived from longitudinal diagnosis/follow-up records only.",
        "Predictive features were taken from the baseline publication feature dataset and leakage-pattern columns were excluded.",
    ]
    AUDIT_PATH.write_text("\n".join(lines), encoding="utf-8")
    log("\n".join(lines))


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    order = np.argsort(time)
    time = time[order]
    event = event[order]
    unique_event_times = np.unique(time[event == 1])
    survival = 1.0
    xs = [0.0]
    ys = [1.0]
    median_survival = np.nan
    for t in unique_event_times:
        at_risk = np.sum(time >= t)
        observed = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            survival *= 1.0 - observed / at_risk
        xs.extend([float(t), float(t)])
        ys.extend([ys[-1], survival])
        if np.isnan(median_survival) and survival <= 0.5:
            median_survival = float(t)
    return np.asarray(xs), np.asarray(ys), median_survival


def logrank_test(time: np.ndarray, event: np.ndarray, groups: np.ndarray) -> tuple[float, float]:
    unique_groups = pd.Series(groups).dropna().unique().tolist()
    if len(unique_groups) < 2:
        return np.nan, np.nan
    event_times = np.unique(time[event == 1])
    observed_minus_expected = np.zeros(len(unique_groups))
    variance = np.zeros((len(unique_groups), len(unique_groups)))
    for t in event_times:
        at_risk = np.array([np.sum((groups == g) & (time >= t)) for g in unique_groups], dtype=float)
        observed = np.array([np.sum((groups == g) & (time == t) & (event == 1)) for g in unique_groups], dtype=float)
        total_risk = at_risk.sum()
        total_events = observed.sum()
        if total_risk <= 1 or total_events <= 0:
            continue
        expected = total_events * at_risk / total_risk
        observed_minus_expected += observed - expected
        factor = total_events * (total_risk - total_events) / (total_risk - 1)
        for i in range(len(unique_groups)):
            for j in range(len(unique_groups)):
                if i == j:
                    variance[i, j] += factor * at_risk[i] / total_risk * (1 - at_risk[i] / total_risk)
                else:
                    variance[i, j] -= factor * at_risk[i] * at_risk[j] / (total_risk**2)
    reduced_v = variance[:-1, :-1]
    reduced_oe = observed_minus_expected[:-1]
    try:
        stat = float(reduced_oe.T @ np.linalg.pinv(reduced_v) @ reduced_oe)
        p_value = float(1 - chi2.cdf(stat, df=len(unique_groups) - 1))
    except Exception:
        stat, p_value = np.nan, np.nan
    return stat, p_value


def harrell_c_index(time: np.ndarray, event: np.ndarray, risk: np.ndarray) -> float:
    permissible = 0
    concordant = 0.0
    n = len(time)
    for i in range(n):
        if event[i] != 1:
            continue
        comparable = time[i] < time
        comparable[i] = False
        n_comp = int(comparable.sum())
        if n_comp == 0:
            continue
        permissible += n_comp
        concordant += np.sum(risk[i] > risk[comparable])
        concordant += 0.5 * np.sum(risk[i] == risk[comparable])
    return concordant / permissible if permissible else np.nan


def time_dependent_auc(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: int) -> float:
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    valid = cases | controls
    y = cases[valid].astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, risk[valid]))


def horizon_brier_score(time: np.ndarray, event: np.ndarray, risk: np.ndarray, horizon: int) -> float:
    cases = (event == 1) & (time <= horizon)
    controls = time > horizon
    valid = cases | controls
    if valid.sum() == 0:
        return np.nan
    y = cases[valid].astype(float)
    score = 1 / (1 + np.exp(-risk[valid]))
    return float(np.mean((y - score) ** 2))


def km_survival_at(query_times: np.ndarray, time: np.ndarray, event: np.ndarray) -> np.ndarray:
    """Kaplan-Meier survival estimate evaluated at query times."""
    query_times = np.asarray(query_times, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    out = np.ones(len(query_times), dtype=float)
    survival = 1.0
    for t in np.sort(np.unique(time[event == 1])):
        at_risk = np.sum(time >= t)
        observed = np.sum((time == t) & (event == 1))
        if at_risk > 0:
            survival *= 1.0 - observed / at_risk
        out[query_times >= t] = survival
    return np.clip(out, 1e-6, 1.0)


def breslow_baseline_cumulative_hazard(
    train_time: np.ndarray,
    train_event: np.ndarray,
    train_risk: np.ndarray,
    horizons: list[int],
) -> dict[int, float]:
    """Estimate baseline cumulative hazard for a Cox-type risk score."""
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=int)
    train_risk = np.asarray(train_risk, dtype=float)
    exp_risk = np.exp(np.clip(train_risk, -30, 30))
    event_times = np.sort(np.unique(train_time[train_event == 1]))
    cumulative = []
    hazard = 0.0
    for t in event_times:
        d_t = np.sum((train_time == t) & (train_event == 1))
        risk_sum = exp_risk[train_time >= t].sum()
        if risk_sum > 0:
            hazard += d_t / risk_sum
        cumulative.append((t, hazard))

    output = {}
    for horizon in horizons:
        h0 = 0.0
        for t, value in cumulative:
            if t <= horizon:
                h0 = value
            else:
                break
        output[horizon] = float(h0)
    return output


def survival_probability_from_risk(risk: np.ndarray, baseline_hazard: float) -> np.ndarray:
    """Convert Cox-type risk scores to survival probabilities at one horizon."""
    relative_risk = np.exp(np.clip(risk, -30, 30))
    return np.exp(-baseline_hazard * relative_risk)


def ipcw_brier_score(
    train_time: np.ndarray,
    train_event: np.ndarray,
    test_time: np.ndarray,
    test_event: np.ndarray,
    predicted_survival: np.ndarray,
    horizon: int,
) -> float:
    """IPCW Brier score for survival probability predictions at a horizon."""
    train_time = np.asarray(train_time, dtype=float)
    train_event = np.asarray(train_event, dtype=int)
    test_time = np.asarray(test_time, dtype=float)
    test_event = np.asarray(test_event, dtype=int)
    predicted_survival = np.asarray(predicted_survival, dtype=float)

    # Kaplan-Meier model for censoring: censoring is the event of interest.
    censor_event_train = 1 - train_event
    g_horizon = km_survival_at(np.array([horizon]), train_time, censor_event_train)[0]
    g_observed = km_survival_at(test_time, train_time, censor_event_train)

    observed_before_horizon = (test_time <= horizon) & (test_event == 1)
    observed_after_horizon = test_time > horizon
    weights = np.zeros(len(test_time), dtype=float)
    outcomes = np.zeros(len(test_time), dtype=float)

    weights[observed_before_horizon] = 1.0 / g_observed[observed_before_horizon]
    outcomes[observed_before_horizon] = 0.0
    weights[observed_after_horizon] = 1.0 / g_horizon
    outcomes[observed_after_horizon] = 1.0

    valid = weights > 0
    if not valid.any():
        return np.nan
    errors = weights[valid] * (outcomes[valid] - predicted_survival[valid]) ** 2
    return float(np.mean(errors))


def integrated_brier_score_from_grid(brier_scores: dict[int, float]) -> float:
    """Approximate IBS by integrating IPCW Brier scores across the horizon grid."""
    clean = [(float(t), float(v)) for t, v in brier_scores.items() if not np.isnan(v)]
    if len(clean) < 2:
        return clean[0][1] if clean else np.nan
    times = np.array([x[0] for x in clean], dtype=float)
    scores = np.array([x[1] for x in clean], dtype=float)
    return float(np.trapezoid(scores, times) / (times.max() - times.min()))


def fit_xgb_survival(x_train: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray) -> Pipeline:
    # XGBoost survival:cox expects positive times for events and negative times for censored observations.
    y_xgb = np.where(y_event == 1, y_time, -y_time)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(x_train, scale=False)),
            (
                "model",
                XGBRegressor(**XGB_SURVIVAL_PARAMS),
            ),
        ]
    )
    pipe.fit(x_train, y_xgb)
    return pipe


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


def fit_penalized_cox(x_train: pd.DataFrame, y_time: np.ndarray, y_event: np.ndarray) -> tuple[Pipeline, np.ndarray, list[str]]:
    preprocessor = make_preprocessor(x_train, scale=True)
    x_mat = preprocessor.fit_transform(x_train)
    names = get_feature_names(preprocessor, x_train.columns.tolist())
    beta0 = np.zeros(x_mat.shape[1])
    alpha = 0.1
    result = minimize(
        lambda b: cox_neg_loglik_and_grad(b, x_mat, y_time, y_event, alpha),
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300},
    )
    pipe = Pipeline([("preprocess", preprocessor)])
    return pipe, result.x, names


def cox_predict_risk(pipe: Pipeline, beta: np.ndarray, x: pd.DataFrame) -> np.ndarray:
    return pipe.named_steps["preprocess"].transform(x) @ beta


def run_cross_validation(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    td_rows = []
    brier_rows = []
    time = df["time_to_event_months"].to_numpy(float)
    event = df["event"].to_numpy(int)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    stratify = event

    for set_name, features in feature_sets.items():
        x = df[features].copy()
        for model_name in ["CoxPH", "XGBoostSurvival"]:
            log(f"Evaluating survival model | {model_name} | {set_name}")
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, stratify), start=1):
                x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
                t_train, t_test = time[train_idx], time[test_idx]
                e_train, e_test = event[train_idx], event[test_idx]
                if model_name == "CoxPH":
                    pipe, beta, _ = fit_penalized_cox(x_train, t_train, e_train)
                    risk = cox_predict_risk(pipe, beta, x_test)
                    train_risk = cox_predict_risk(pipe, beta, x_train)
                else:
                    pipe = fit_xgb_survival(x_train, t_train, e_train)
                    risk = pipe.predict(x_test)
                    train_risk = pipe.predict(x_train)

                td_aucs = {
                    f"auc_{h}m": time_dependent_auc(t_test, e_test, risk, h)
                    for h in TIME_HORIZONS
                }
                baseline_hazards = breslow_baseline_cumulative_hazard(
                    t_train,
                    e_train,
                    train_risk,
                    TIME_HORIZONS,
                )
                ipcw_briers = {}
                for h in TIME_HORIZONS:
                    survival_prob = survival_probability_from_risk(risk, baseline_hazards[h])
                    ipcw_briers[f"ipcw_brier_{h}m"] = ipcw_brier_score(
                        t_train,
                        e_train,
                        t_test,
                        e_test,
                        survival_prob,
                        h,
                    )
                briers = [
                    horizon_brier_score(t_test, e_test, risk, h)
                    for h in TIME_HORIZONS
                    if not np.isnan(horizon_brier_score(t_test, e_test, risk, h))
                ]
                ibs = integrated_brier_score_from_grid(
                    {h: ipcw_briers[f"ipcw_brier_{h}m"] for h in TIME_HORIZONS}
                )
                row = {
                    "model": model_name,
                    "feature_set": set_name,
                    "fold": fold,
                    "c_index": harrell_c_index(t_test, e_test, risk),
                    "integrated_brier_score_approx": float(np.mean(briers)) if briers else np.nan,
                    "integrated_brier_score_ipcw": ibs,
                    "mean_time_dependent_auc": float(np.nanmean(list(td_aucs.values()))),
                    **td_aucs,
                    **ipcw_briers,
                }
                rows.append(row)
                for h in TIME_HORIZONS:
                    td_rows.append(
                        {
                            "model": model_name,
                            "feature_set": set_name,
                            "fold": fold,
                            "horizon_months": h,
                            "time_dependent_auc": td_aucs[f"auc_{h}m"],
                        }
                    )
                    brier_rows.append(
                        {
                            "model": model_name,
                            "feature_set": set_name,
                            "fold": fold,
                            "horizon_months": h,
                            "ipcw_brier_score": ipcw_briers[f"ipcw_brier_{h}m"],
                        }
                    )

    folds = pd.DataFrame(rows)
    folds.to_csv(CV_RESULTS_PATH, index=False)
    pd.DataFrame(td_rows).to_csv(TD_AUC_PATH, index=False)
    pd.DataFrame(brier_rows).to_csv(BRIER_PATH, index=False)
    metric_cols = [
        "c_index",
        "integrated_brier_score_ipcw",
        "integrated_brier_score_approx",
        "mean_time_dependent_auc",
    ] + [
        f"auc_{h}m" for h in TIME_HORIZONS
    ] + [
        f"ipcw_brier_{h}m" for h in TIME_HORIZONS
    ]
    summary = folds.groupby(["model", "feature_set"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["c_index_mean", "mean_time_dependent_auc_mean"], ascending=False)
    summary.to_csv(RESULTS_DIR / "survival_model_comparison.csv", index=False)
    return folds, summary


def train_full_xgb_risk(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    pipe = fit_xgb_survival(df[features], df["time_to_event_months"].to_numpy(float), df["event"].to_numpy(int))
    return pipe.predict(df[features])


def train_full_cox_risk(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, dict[int, float]]:
    pipe, beta, _ = fit_penalized_cox(
        df[features],
        df["time_to_event_months"].to_numpy(float),
        df["event"].to_numpy(int),
    )
    risk = cox_predict_risk(pipe, beta, df[features])
    baseline_hazards = breslow_baseline_cumulative_hazard(
        df["time_to_event_months"].to_numpy(float),
        df["event"].to_numpy(int),
        risk,
        TIME_HORIZONS,
    )
    return risk, baseline_hazards


def plot_km(df: pd.DataFrame, risk: np.ndarray) -> dict:
    q25, q75 = np.nanquantile(risk, [0.25, 0.75])
    groups = np.where(risk >= q75, "Top quartile risk", np.where(risk <= q25, "Lowest risk", "Middle risk"))
    df = df.copy()
    df["risk_group"] = groups
    plt.figure(figsize=(8, 6))
    median_by_group = {}
    for group, sub in df.groupby("risk_group"):
        xs, ys, median_survival = kaplan_meier(
            sub["time_to_event_months"].to_numpy(float),
            sub["event"].to_numpy(int),
        )
        median_by_group[group] = median_survival
        plt.step(xs, ys, where="post", label=f"{group} (n={len(sub)})")

    xs, ys, median_all = kaplan_meier(df["time_to_event_months"].to_numpy(float), df["event"].to_numpy(int))
    plt.step(xs, ys, where="post", color="black", linestyle="--", label=f"Entire cohort (n={len(df)})")
    stat, p_value = logrank_test(
        df["time_to_event_months"].to_numpy(float),
        df["event"].to_numpy(int),
        df["risk_group"].to_numpy(),
    )
    plt.xlabel("Months from baseline")
    plt.ylabel("AD-free survival probability")
    plt.title(f"Kaplan-Meier Curves by Baseline Risk Group\nLog-rank p={p_value:.3e}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(KM_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    return {"median_entire_cohort": median_all, "median_by_group": median_by_group, "logrank_statistic": stat, "logrank_p": p_value}


def univariate_cox_table(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    time = df["time_to_event_months"].to_numpy(float)
    event = df["event"].to_numpy(int)
    for set_name, features in feature_sets.items():
        for feature in features:
            values = pd.to_numeric(df[feature], errors="coerce")
            if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
                continue
            x = values.fillna(values.median()).to_numpy(float)
            x = (x - np.nanmean(x)) / (np.nanstd(x) or 1.0)

            def objective(beta_arr: np.ndarray) -> float:
                beta = float(beta_arr[0])
                return cox_neg_loglik_and_grad(np.array([beta]), x.reshape(-1, 1), time, event, alpha=0.0)[0]

            result = minimize(objective, np.array([0.0]), method="BFGS")
            beta = float(result.x[0])
            try:
                se = float(np.sqrt(result.hess_inv[0, 0]))
            except Exception:
                se = np.nan
            z = beta / se if se and not np.isnan(se) else np.nan
            p_value = 2 * (1 - norm.cdf(abs(z))) if not np.isnan(z) else np.nan
            rows.append(
                {
                    "feature_set": set_name,
                    "feature": feature,
                    "beta_per_1sd": beta,
                    "hazard_ratio_per_1sd": math.exp(beta),
                    "ci95_lower": math.exp(beta - 1.96 * se) if not np.isnan(se) else np.nan,
                    "ci95_upper": math.exp(beta + 1.96 * se) if not np.isnan(se) else np.nan,
                    "p_value": p_value,
                }
            )
    out = pd.DataFrame(rows).sort_values("p_value", na_position="last")
    out.to_csv(COX_IMPORTANCE_PATH, index=False)
    return out


def plot_time_dependent_roc(df: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    rows = []
    time = df["time_to_event_months"].to_numpy(float)
    event = df["event"].to_numpy(int)
    plt.figure(figsize=(7, 6))
    for horizon in TIME_HORIZONS:
        cases = (event == 1) & (time <= horizon)
        controls = time > horizon
        valid = cases | controls
        y = cases[valid].astype(int)
        if len(np.unique(y)) < 2:
            auc_value = np.nan
            continue
        fpr, tpr, _ = roc_curve(y, risk[valid])
        auc_value = auc(fpr, tpr)
        rows.append({"horizon_months": horizon, "time_dependent_auc": auc_value, "n_evaluable": int(valid.sum())})
        plt.plot(fpr, tpr, label=f"{horizon}m AUC={auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Time-dependent ROC Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "time_dependent_roc_curves.png", dpi=300, bbox_inches="tight")
    plt.close()
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "full_model_time_dependent_auc.csv", index=False)
    return out


def km_event_probability(time: np.ndarray, event: np.ndarray, horizon: int) -> float:
    xs, ys, _ = kaplan_meier(time, event)
    survival = 1.0
    for x, y in zip(xs, ys):
        if x <= horizon:
            survival = y
        else:
            break
    return float(1.0 - survival)


def survival_calibration_analysis(
    df: pd.DataFrame,
    features: list[str],
    model_label: str = "CoxPH_Top50",
    n_bins: int = 5,
) -> pd.DataFrame:
    risk, baseline_hazards = train_full_cox_risk(df, features)
    time = df["time_to_event_months"].to_numpy(float)
    event = df["event"].to_numpy(int)
    rows = []

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes = axes.ravel()
    for ax, horizon in zip(axes, TIME_HORIZONS):
        predicted_event = 1.0 - survival_probability_from_risk(risk, baseline_hazards[horizon])
        bins = pd.qcut(
            predicted_event,
            q=min(n_bins, pd.Series(predicted_event).nunique()),
            labels=False,
            duplicates="drop",
        )
        plot_x = []
        plot_y = []
        for bin_id in sorted(pd.Series(bins).dropna().unique()):
            mask = np.asarray(bins == bin_id)
            if mask.sum() < 10:
                continue
            mean_pred = float(np.mean(predicted_event[mask]))
            observed = km_event_probability(time[mask], event[mask], horizon)
            rows.append(
                {
                    "model": model_label,
                    "horizon_months": horizon,
                    "risk_bin": int(bin_id) + 1,
                    "n_subjects": int(mask.sum()),
                    "mean_predicted_event_probability": mean_pred,
                    "observed_event_probability_km": observed,
                    "calibration_error": observed - mean_pred,
                    "absolute_calibration_error": abs(observed - mean_pred),
                }
            )
            plot_x.append(mean_pred)
            plot_y.append(observed)
        ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
        ax.plot(plot_x, plot_y, marker="o", color="#4477AA")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean predicted conversion probability")
        ax.set_ylabel("Observed conversion probability")
        ax.set_title(f"{horizon}-month calibration")
    fig.suptitle(f"Survival Calibration Curves: {model_label}", y=1.02)
    fig.tight_layout()
    fig.savefig(CALIBRATION_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    out = pd.DataFrame(rows)
    out.to_csv(CALIBRATION_PATH, index=False)
    return out


def categorize_feature(feature: str) -> str:
    upper = feature.upper()
    if any(token in upper for token in ["ADAS", "MMSE", "FAQ", "CDR", "NEUROBAT", "MOCA"]):
        return "Clinical/Cognitive"
    if any(token in upper for token in ["AMY", "PET", "SUVR", "CENTILOIDS", "FDG", "TAU"]):
        return "PET"
    if any(token in upper for token in ["UCSF", "MRI", "HIPPO", "ENTORHINAL", "VENTRICLE", "ST", "SV", "TA", "CV"]):
        return "MRI"
    if any(token in upper for token in ["AGE", "PTGENDER", "SEX", "PTEDUCAT", "DEMOG"]):
        return "Demographic"
    if any(token in upper for token in ["APOE", "GENETIC"]):
        return "Genetic"
    return "Other"


def write_reports(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    cox_table: pd.DataFrame,
    cv_summary: pd.DataFrame,
    km_stats: dict,
    td_auc: pd.DataFrame,
    calibration_table: pd.DataFrame,
) -> None:
    best_discrimination = cv_summary.iloc[0]
    best_ibs = cv_summary.sort_values(
        ["integrated_brier_score_ipcw_mean", "c_index_mean"],
        ascending=[True, False],
    ).iloc[0]
    top_cox = cox_table.head(20).copy()
    top_cox["category"] = top_cox["feature"].map(categorize_feature)
    early_auc = td_auc[td_auc["horizon_months"].isin([24, 36])].copy()
    late_auc = td_auc[td_auc["horizon_months"].isin([60, 84])].copy()

    biomarker_lines = [
        "Survival Biomarker Report",
        "",
        "Top prognostic biomarkers by univariate Cox p-value:",
        top_cox.to_string(index=False),
        "",
        "Top clinical/cognitive biomarkers:",
        top_cox[top_cox["category"] == "Clinical/Cognitive"].head(10).to_string(index=False),
        "",
        "Top PET biomarkers:",
        top_cox[top_cox["category"] == "PET"].head(10).to_string(index=False),
        "",
        "Top MRI biomarkers:",
        top_cox[top_cox["category"] == "MRI"].head(10).to_string(index=False),
        "",
        "Interpretation:",
        "Higher hazard ratios indicate shorter time to AD conversion. Clinical/cognitive impairment features usually reflect symptomatic proximity to dementia, while MRI/PET biomarkers may reflect underlying neurodegeneration or amyloid burden.",
    ]
    BIOMARKER_REPORT_PATH.write_text("\n".join(biomarker_lines), encoding="utf-8")

    final_lines = [
        "ADNI Survival Prediction Experiment Report",
        "",
        "Dataset",
        f"Subjects: {len(df)}",
        f"Events/converters: {int(df['event'].sum())}",
        f"Censored: {int((df['event'] == 0).sum())}",
        f"Median observed time: {df['time_to_event_months'].median():.2f} months",
        "",
        "Best survival model by discrimination",
        best_discrimination.to_string(),
        "",
        "Best survival model by Integrated Brier Score",
        best_ibs.to_string(),
        "",
        "Kaplan-Meier analysis",
        json.dumps(km_stats, indent=2, default=str),
        "",
        "Time-dependent AUC from full Top50 XGBoost survival risk",
        td_auc.to_string(index=False),
        "",
        "Survival calibration",
        calibration_table.to_string(index=False),
        "",
        "Comparison with classification",
        "The current 36-month binary RF model reported AUC=0.896. Survival models additionally use censored subjects and estimate risk over continuous time. Classification is simpler for a fixed clinical horizon, while survival analysis is more informative for prognosis and individualized follow-up planning.",
        "",
        "XGBoost survival regularization",
        json.dumps(XGB_SURVIVAL_PARAMS, indent=2, default=str),
        "",
        "Integrated Brier Score note",
        "The script reports an IPCW-adjusted Integrated Brier Score estimated on the validation folds across the prespecified 24-, 36-, 60-, and 84-month time grid. Because scikit-survival is not installed, Random Survival Forest was not run.",
        "",
        "Publication recommendation",
        "Use the survival analysis as a complementary prognostic experiment. Report C-index and time-dependent AUC at 24, 36, 60, and 84 months alongside the fixed-horizon classification results.",
    ]
    FINAL_REPORT_PATH.write_text("\n".join(final_lines), encoding="utf-8")

    JSON_REPORT_PATH.write_text(
        json.dumps(
            {
                "n_subjects": len(df),
                "n_events": int(df["event"].sum()),
                "n_censored": int((df["event"] == 0).sum()),
                "best_discrimination_model": best_discrimination.to_dict(),
                "best_ibs_model": best_ibs.to_dict(),
                "xgb_survival_params": XGB_SURVIVAL_PARAMS,
                "km_stats": km_stats,
                "outputs": {
                    "survival_dataset": str(SURVIVAL_DATA_PATH),
                    "audit": str(AUDIT_PATH),
                    "km_plot": str(KM_PLOT_PATH),
                    "cox_importance": str(COX_IMPORTANCE_PATH),
                    "cv_results": str(CV_RESULTS_PATH),
                    "time_dependent_auc": str(TD_AUC_PATH),
                    "ipcw_brier_scores": str(BRIER_PATH),
                    "survival_calibration": str(CALIBRATION_PATH),
                    "survival_calibration_plot": str(CALIBRATION_PLOT_PATH),
                    "biomarker_report": str(BIOMARKER_REPORT_PATH),
                    "final_report": str(FINAL_REPORT_PATH),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


def main() -> None:
    log("====================================================")
    log("ADNI SURVIVAL ANALYSIS: MCI TO AD TIME-TO-CONVERSION")
    log("====================================================")
    df = build_survival_dataset()
    write_survival_audit(df)

    feature_sets = {}
    for name, path in TOP_FILES.items():
        features = load_feature_set(path)
        features = [f for f in features if f in df.columns and not is_leakage_feature(f)]
        feature_sets[name] = features
        log(f"{name}: {len(features)} usable baseline features")

    # Use Top50 XGBoost survival risk for Kaplan-Meier risk groups and full-cohort time ROC.
    risk_features = feature_sets.get("Top50") or next(iter(feature_sets.values()))
    log("\nTraining full Top50 XGBoost survival model for risk stratification...")
    full_risk = train_full_xgb_risk(df, risk_features)
    km_stats = plot_km(df, full_risk)
    td_auc = plot_time_dependent_roc(df, full_risk)

    log("\nRunning univariate Cox hazard-ratio analysis...")
    cox_table = univariate_cox_table(df, feature_sets)

    log("\nRunning 5-fold x 10-repeat survival cross-validation...")
    _, cv_summary = run_cross_validation(df, feature_sets)

    log("\nRunning survival calibration analysis for CoxPH Top50...")
    calibration_features = feature_sets.get("Top50") or risk_features
    calibration_table = survival_calibration_analysis(df, calibration_features)

    write_reports(df, feature_sets, cox_table, cv_summary, km_stats, td_auc, calibration_table)

    log("\n====================================================")
    log("SURVIVAL MODEL COMPARISON")
    log("====================================================")
    log(cv_summary.to_string(index=False))

    log("\n====================================================")
    log("TOP PROGNOSTIC BIOMARKERS")
    log("====================================================")
    log(cox_table.head(20).to_string(index=False))

    best = cv_summary.iloc[0]
    best_ibs = cv_summary.sort_values(
        ["integrated_brier_score_ipcw_mean", "c_index_mean"],
        ascending=[True, False],
    ).iloc[0]
    log("\n====================================================")
    log("FINAL SURVIVAL RECOMMENDATION")
    log("====================================================")
    log(f"Best discrimination model: {best['model']} with {best['feature_set']}")
    log(f"C-index: {best['c_index_mean']:.4f} +/- {best['c_index_std']:.4f}")
    log(f"Mean time-dependent AUC: {best['mean_time_dependent_auc_mean']:.4f} +/- {best['mean_time_dependent_auc_std']:.4f}")
    log("")
    log(f"Best IBS model: {best_ibs['model']} with {best_ibs['feature_set']}")
    log(
        "IBS: "
        f"{best_ibs['integrated_brier_score_ipcw_mean']:.4f} +/- "
        f"{best_ibs['integrated_brier_score_ipcw_std']:.4f}"
    )
    log("Use the lowest-IBS model when the goal is calibrated absolute survival probability.")
    log("Use survival modeling as a complementary prognostic analysis to fixed-horizon classification.")
    log(f"Calibration table: {CALIBRATION_PATH}")
    log(f"Calibration plot: {CALIBRATION_PLOT_PATH}")
    log(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
