"""Refit and interpret the twelve recorded primary configurations."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
RANKING_DIR = PROCESSED_DIR / "publication_feature_ranking"
OUT_DIR = PROCESSED_DIR / "shap_interpretation" / "primary_model_shap_continuation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FULL_DATA_PATH = PROCESSED_DIR / "final_mci_to_ad_publication_dataset.csv"
HORIZON_DIR = PROCESSED_DIR / "prediction_horizon_analysis"
SURVIVAL_DATA_PATH = PROCESSED_DIR / "survival_analysis" / "results" / "adni_survival_dataset.csv"

RANDOM_STATE = 42
TIME_HORIZONS = [24, 36, 60, 84]
TOP_FILES = {
    20: RANKING_DIR / "top20_features.csv",
    50: RANKING_DIR / "top50_features.csv",
    100: RANKING_DIR / "top100_features.csv",
}
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
    r"VISCODE",
    r"EXAMDATE",
    r"SCANDATE",
]

FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8
MAX_DISPLAY = 20
MAX_BACKGROUND = 200


@dataclass(frozen=True)
class ShapConfig:
    analysis: str
    dataset_kind: str
    model_name: str
    top_k: int
    target: str
    horizon: int | None = None

    @property
    def config_id(self) -> str:
        parts = [self.analysis, self.model_name, f"Top{self.top_k}"]
        if self.horizon is not None:
            parts.insert(1, f"{self.horizon}m")
        return sanitize_name("_".join(parts))


def log(message: str = "") -> None:
    print(message, flush=True)


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def is_leakage_feature(feature: str) -> bool:
    upper = feature.upper()
    return any(re.search(pattern, upper) for pattern in LEAKAGE_PATTERNS)


def coerce_numeric_like(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    for col in out.columns:
        if out[col].dtype == object:
            numeric = pd.to_numeric(out[col], errors="coerce")
            non_missing = out[col].notna().sum()
            if non_missing and numeric.notna().sum() / non_missing >= 0.8:
                out[col] = numeric
    return out


def load_feature_list(top_k: int) -> list[str]:
    path = TOP_FILES[top_k]
    df = pd.read_csv(path)
    if "feature" not in df.columns:
        raise ValueError(f"{path} must contain a feature column.")
    features = df["feature"].dropna().astype(str).tolist()
    if len(features) < top_k:
        raise ValueError(f"{path} contains only {len(features)} features, but Top-{top_k} was requested.")
    return features[:top_k]


def target_column(df: pd.DataFrame) -> str:
    for col in ["label", "conversion_label", "converter_label", "converter"]:
        if col in df.columns:
            return col
    raise ValueError("No target column found.")


def make_preprocessor(x: pd.DataFrame, scale_numeric: bool) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", one_hot_encoder()),
        ]
    )
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric),
            ("cat", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return fallback


def readable_label(feature: str) -> str:
    labels = {
        "ADAS_TOTAL13": "ADAS total score",
        "ADAS_TOTSCORE": "ADAS total score",
        "FAQ_FAQTOTAL": "FAQ total score",
        "CDR_CDRSB": "CDR sum of boxes",
        "CDR_CDJUDGE": "CDR judgment and problem solving",
        "CDR_CDMEMORY": "CDR memory",
        "MMSE_MMSCORE": "MMSE total score",
        "AMY_SUMMARY_SUVR": "Amyloid PET summary SUVR",
        "AMY_CENTILOIDS": "Amyloid PET Centiloids",
        "VITALS_BMI": "Body mass index",
        "PTDEMOG_PTEDUCAT": "Years of education",
        "NEUROBAT_AVTOT5": "Auditory verbal learning trial 5",
        "NEUROBAT_LDELTOTAL": "Logical memory delayed recall",
        "NEUROBAT_AVTOT6": "Auditory verbal learning trial 6",
        "NEUROBAT_AVDEL30MIN": "Auditory verbal delayed recall",
    }
    if feature in labels:
        return labels[feature]
    label = feature
    label = label.replace("UCBERKELEY_AMY_6MM_", "Amyloid PET ")
    label = label.replace("SUMMARY_SUVR", "summary SUVR")
    label = label.replace("CENTILOIDS", "Centiloids")
    label = label.replace("FAQ_", "FAQ ")
    label = label.replace("CDR_", "CDR ")
    label = label.replace("ADAS_", "ADAS ")
    label = label.replace("MMSE_", "MMSE ")
    label = label.replace("NEUROBAT_", "Neuropsychological test ")
    label = label.replace("PTDEMOG_", "Demographic ")
    label = label.replace("VITALS_", "Vitals ")
    label = label.replace("_", " ")
    return " ".join(label.split())


def set_ieee_style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE,
            "axes.labelsize": FONT_SIZE,
            "axes.titlesize": FONT_SIZE,
            "xtick.labelsize": FONT_SIZE,
            "ytick.labelsize": FONT_SIZE,
            "legend.fontsize": FONT_SIZE,
            "figure.titlesize": FONT_SIZE,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )


def make_model(model_name: str) -> tuple[object, bool, str]:
    if model_name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE), True, "linear"
    if model_name == "RandomForest":
        return (
            RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
            "tree",
        )
    if model_name == "XGBoost":
        return (
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
            "tree",
        )
    if model_name == "CatBoost":
        return (
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
            "tree",
        )
    raise ValueError(f"Unsupported classification model: {model_name}")


def make_xgb_survival() -> XGBRegressor:
    return XGBRegressor(
        objective="survival:cox",
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def load_classification_data(config: ShapConfig) -> tuple[pd.DataFrame, np.ndarray]:
    if config.dataset_kind == "anyfuture":
        df = pd.read_csv(FULL_DATA_PATH)
        target = target_column(df)
        features = load_feature_list(config.top_k)
        missing = [feature for feature in features if feature not in df.columns]
        if missing:
            raise ValueError(f"Missing features for {config.config_id}: {missing[:10]}")
        leakage = [feature for feature in features if is_leakage_feature(feature)]
        if leakage:
            raise ValueError(f"Leakage-like features in {config.config_id}: {leakage}")
        x = df[features].copy()
        y = df[target].astype(int).to_numpy()
        return coerce_numeric_like(x), y

    if config.dataset_kind == "horizon":
        if config.horizon is None:
            raise ValueError("Horizon configuration requires horizon month.")
        path = HORIZON_DIR / f"top{config.top_k}_{config.horizon}m_dataset.csv"
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError(f"{path} must contain label.")
        x = df.drop(columns=["label"])
        y = df["label"].astype(int).to_numpy()
        return coerce_numeric_like(x), y

    raise ValueError(f"Unsupported dataset kind: {config.dataset_kind}")


def load_survival_data(config: ShapConfig) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    df = pd.read_csv(SURVIVAL_DATA_PATH)
    features = load_feature_list(config.top_k)
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing survival features for {config.config_id}: {missing[:10]}")
    x = coerce_numeric_like(df[features].copy())
    duration = df["time_to_event_months"].astype(float).to_numpy()
    event = df["event"].astype(int).to_numpy()
    return x, duration, event


def fit_classification_model(config: ShapConfig, x: pd.DataFrame, y: np.ndarray) -> tuple[Pipeline, str]:
    model, scale_numeric, explainer_kind = make_model(config.model_name)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(x, scale_numeric)),
            ("model", model),
        ]
    )
    pipe.fit(x, y)
    return pipe, explainer_kind


def fit_xgb_survival_model(x: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> Pipeline:
    y_xgb = np.where(event == 1, duration, -duration)
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(x, scale_numeric=False)),
            ("model", make_xgb_survival()),
        ]
    )
    pipe.fit(x, y_xgb)
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


def fit_penalized_cox_processed(x_proc: pd.DataFrame, duration: np.ndarray, event: np.ndarray) -> np.ndarray:
    from scipy.optimize import minimize

    beta0 = np.zeros(x_proc.shape[1])
    result = minimize(
        lambda b: cox_neg_loglik_and_grad(b, x_proc.to_numpy(float), duration, event, alpha=0.1),
        beta0,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300},
    )
    return result.x


def transformed_frame(pipe: Pipeline, x: pd.DataFrame) -> pd.DataFrame:
    x_t = pipe.named_steps["preprocess"].transform(x)
    names = get_feature_names(pipe.named_steps["preprocess"], x.columns.tolist())
    return pd.DataFrame(x_t, columns=names, index=x.index)


def normalize_shap_values(values: object) -> np.ndarray:
    if isinstance(values, list):
        values = values[-1]
    if hasattr(values, "values"):
        values = values.values
    arr = np.asarray(values)
    if arr.ndim == 3:
        arr = arr[:, :, -1]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")
    return arr


def tree_shap_values(model: object, x_proc: pd.DataFrame) -> np.ndarray:
    explainer = shap.TreeExplainer(model)
    try:
        values = explainer.shap_values(x_proc)
    except Exception:
        values = explainer(x_proc)
    return normalize_shap_values(values)


def linear_contributions(beta: np.ndarray, x_proc: pd.DataFrame) -> np.ndarray:
    centered = x_proc.to_numpy(float) - x_proc.to_numpy(float).mean(axis=0, keepdims=True)
    return centered * beta.reshape(1, -1)


def summarize_values(values: np.ndarray, x_proc: pd.DataFrame, config: ShapConfig, method: str) -> pd.DataFrame:
    rows = []
    for idx, feature in enumerate(x_proc.columns):
        svals = values[:, idx]
        fvals = pd.to_numeric(x_proc[feature], errors="coerce")
        rows.append(
            {
                "analysis": config.analysis,
                "dataset_kind": config.dataset_kind,
                "model": config.model_name,
                "top_k": config.top_k,
                "horizon": config.horizon,
                "method": method,
                "feature": feature,
                "feature_label": readable_label(feature),
                "mean_abs_contribution": float(np.mean(np.abs(svals))),
                "mean_contribution": float(np.mean(svals)),
                "median_contribution": float(np.median(svals)),
                "feature_min": float(np.nanmin(fvals)),
                "feature_max": float(np.nanmax(fvals)),
                "feature_median": float(np.nanmedian(fvals)),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_contribution", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[
        [
            "analysis",
            "dataset_kind",
            "model",
            "top_k",
            "horizon",
            "method",
            "rank",
            "feature",
            "feature_label",
            "mean_abs_contribution",
            "mean_contribution",
            "median_contribution",
            "feature_min",
            "feature_max",
            "feature_median",
        ]
    ]


def relabel_frame(x_proc: pd.DataFrame) -> pd.DataFrame:
    out = x_proc.copy()
    out.columns = [readable_label(col) for col in out.columns]
    return out


def save_summary_plot(values: np.ndarray, x_proc: pd.DataFrame, path: Path, xlabel: str) -> None:
    set_ieee_style()
    x_plot = relabel_frame(x_proc)
    shap.summary_plot(
        values,
        x_plot,
        show=False,
        max_display=MAX_DISPLAY,
        plot_size=(7.0, 4.8),
        color_bar_label="Feature value",
    )
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.2)
    if fig.axes:
        fig.axes[0].set_xlabel(xlabel, fontsize=FONT_SIZE, fontname=FONT_FAMILY)
    if len(fig.axes) > 1:
        cax = fig.axes[-1]
        cax.set_xlabel("")
        cax.set_title("")
        cax.set_ylabel("Feature value", fontsize=FONT_SIZE, fontname=FONT_FAMILY, rotation=90, labelpad=14)
        cax.yaxis.set_label_position("right")
        cax.yaxis.tick_right()
        y_min, y_max = cax.get_ylim()
        cax.set_yticks([y_min, y_max])
        cax.set_yticklabels(["Low", "High"], fontname=FONT_FAMILY, fontsize=FONT_SIZE)
        cax.tick_params(axis="y", length=0, pad=4)
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(FONT_FAMILY)
            label.set_fontsize(FONT_SIZE)
    plt.subplots_adjust(left=0.34, right=0.90, bottom=0.12, top=0.98)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_bar_plot(values: np.ndarray, x_proc: pd.DataFrame, path: Path, xlabel: str) -> None:
    set_ieee_style()
    x_plot = relabel_frame(x_proc)
    shap.summary_plot(values, x_plot, plot_type="bar", show=False, max_display=MAX_DISPLAY, plot_size=(7.0, 4.8))
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.2)
    if fig.axes:
        fig.axes[0].set_xlabel(xlabel, fontsize=FONT_SIZE, fontname=FONT_FAMILY)
    for ax in fig.axes:
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(FONT_FAMILY)
            label.set_fontsize(FONT_SIZE)
    plt.subplots_adjust(left=0.34, right=0.96, bottom=0.12, top=0.98)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def save_outputs(config: ShapConfig, values: np.ndarray, x_proc: pd.DataFrame, method: str, out_dir: Path) -> pd.DataFrame:
    summary = summarize_values(values, x_proc, config, method)
    summary.to_csv(out_dir / "feature_contribution_ranking.csv", index=False)
    np.save(out_dir / "contribution_values.npy", values)
    x_proc.to_csv(out_dir / "processed_feature_matrix.csv", index=False)

    if method == "tree_shap":
        xlabel = "SHAP value contribution to predicted conversion risk"
        if config.dataset_kind == "survival":
            xlabel = "SHAP value contribution to predicted conversion hazard"
    else:
        xlabel = "Cox linear contribution to predicted conversion hazard"

    save_summary_plot(values, x_proc, out_dir / "summary_plot_ieee.png", xlabel)
    save_bar_plot(values, x_proc, out_dir / "bar_plot_ieee.png", "Mean absolute contribution")
    return summary


def config_done(config_dir: Path, force: bool) -> bool:
    required = [
        config_dir / "feature_contribution_ranking.csv",
        config_dir / "summary_plot_ieee.png",
        config_dir / "bar_plot_ieee.png",
    ]
    return (not force) and all(path.exists() for path in required)


def run_config(config: ShapConfig, force: bool) -> pd.DataFrame | None:
    config_dir = OUT_DIR / config.config_id
    config_dir.mkdir(parents=True, exist_ok=True)
    if config_done(config_dir, force):
        log(f"Skip completed: {config.config_id}")
        return pd.read_csv(config_dir / "feature_contribution_ranking.csv")

    log(f"Run SHAP continuation: {config.config_id}")
    if config.dataset_kind in {"anyfuture", "horizon"}:
        x, y = load_classification_data(config)
        pipe, explainer_kind = fit_classification_model(config, x, y)
        x_proc = transformed_frame(pipe, x)
        if explainer_kind == "tree":
            values = tree_shap_values(pipe.named_steps["model"], x_proc)
            method = "tree_shap"
        else:
            model = pipe.named_steps["model"]
            values = linear_contributions(model.coef_.reshape(-1), x_proc)
            method = "linear_contribution"
    elif config.dataset_kind == "survival":
        x, duration, event = load_survival_data(config)
        if config.model_name == "XGBoostSurvival":
            pipe = fit_xgb_survival_model(x, duration, event)
            x_proc = transformed_frame(pipe, x)
            values = tree_shap_values(pipe.named_steps["model"], x_proc)
            method = "tree_shap"
        elif config.model_name == "CoxPH":
            preprocessor = make_preprocessor(x, scale_numeric=True)
            x_mat = preprocessor.fit_transform(x)
            names = get_feature_names(preprocessor, x.columns.tolist())
            x_proc = pd.DataFrame(x_mat, columns=names, index=x.index)
            beta = fit_penalized_cox_processed(x_proc, duration, event)
            values = linear_contributions(beta, x_proc)
            method = "cox_linear_contribution"
        else:
            raise ValueError(f"Unsupported survival model: {config.model_name}")
    else:
        raise ValueError(f"Unsupported dataset kind: {config.dataset_kind}")

    summary = save_outputs(config, values, x_proc, method, config_dir)
    (config_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2), encoding="utf-8")
    return summary


def default_configs() -> list[ShapConfig]:
    configs = [
        ShapConfig("primary_anyfuture", "anyfuture", "XGBoost", 20, "conversion_label"),
        ShapConfig("primary_anyfuture", "anyfuture", "XGBoost", 50, "conversion_label"),
        ShapConfig("primary_anyfuture", "anyfuture", "XGBoost", 100, "conversion_label"),
        ShapConfig("primary_anyfuture_best", "anyfuture", "CatBoost", 100, "conversion_label"),
        ShapConfig("primary_anyfuture_best", "anyfuture", "RandomForest", 50, "conversion_label"),
        ShapConfig("fixed_horizon_best", "horizon", "RandomForest", 50, "label", horizon=24),
        ShapConfig("fixed_horizon_best", "horizon", "RandomForest", 50, "label", horizon=36),
        ShapConfig("fixed_horizon_best", "horizon", "CatBoost", 50, "label", horizon=60),
        ShapConfig("primary_survival", "survival", "XGBoostSurvival", 20, "event"),
        ShapConfig("primary_survival", "survival", "XGBoostSurvival", 50, "event"),
        ShapConfig("primary_survival", "survival", "XGBoostSurvival", 100, "event"),
        ShapConfig("primary_survival", "survival", "CoxPH", 50, "event"),
    ]
    return configs


def filter_configs(configs: list[ShapConfig], ids: list[str] | None, scopes: list[str] | None) -> list[ShapConfig]:
    out = configs
    if scopes:
        wanted = set(scopes)
        out = [config for config in out if config.analysis in wanted or config.dataset_kind in wanted]
    if ids:
        wanted = set(ids)
        out = [config for config in out if config.config_id in wanted]
    return out


def write_manifest(configs: list[ShapConfig]) -> None:
    manifest = {
        "analysis_name": "primary_model_shap_continuation",
        "purpose": "Continue SHAP interpretation on previous primary results without modifying original outputs.",
        "output_dir": str(OUT_DIR),
        "input_dataset": str(FULL_DATA_PATH),
        "horizon_dataset_dir": str(HORIZON_DIR),
        "survival_dataset": str(SURVIVAL_DATA_PATH),
        "feature_lists": {f"Top{k}": str(path) for k, path in TOP_FILES.items()},
        "random_state": RANDOM_STATE,
        "configs": [config.__dict__ | {"config_id": config.config_id} for config in configs],
        "notes": [
            "Tree SHAP is used for RandomForest, XGBoost, CatBoost, and XGBoost Survival.",
            "Cox PH is reported as additive Cox linear risk contributions, not Tree SHAP.",
            "Models are refit on the full corresponding primary dataset only for final interpretation.",
            "This script uses previous primary datasets and feature rankings, not robustness/sensitivity outputs.",
        ],
    }
    (OUT_DIR / "shap_continuation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def combine_outputs() -> None:
    rows = []
    for path in sorted(OUT_DIR.glob("*/feature_contribution_ranking.csv")):
        df = pd.read_csv(path)
        df.insert(0, "config_id", path.parent.name)
        rows.append(df.head(30))
    if rows:
        combined = pd.concat(rows, ignore_index=True)
        combined.to_csv(OUT_DIR / "combined_top30_feature_contributions.csv", index=False)
        index = (
            combined.groupby(["config_id", "analysis", "dataset_kind", "model", "top_k", "horizon", "method"], dropna=False)
            .size()
            .reset_index(name="top_features_listed")
        )
        index.to_csv(OUT_DIR / "completed_shap_config_index.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continue SHAP interpretation for previous primary ADNI models.")
    parser.add_argument("--list-configs", action="store_true", help="List planned configuration IDs and exit.")
    parser.add_argument("--only-id", nargs="+", help="Run only selected config IDs.")
    parser.add_argument("--scope", nargs="+", help="Run only selected analyses/dataset kinds.")
    parser.add_argument("--force", action="store_true", help="Recompute outputs even if config files already exist.")
    args = parser.parse_args()

    configs = filter_configs(default_configs(), args.only_id, args.scope)
    write_manifest(configs)

    if args.list_configs:
        for config in configs:
            print(config.config_id)
        return

    all_summaries = []
    failures = []
    for config in configs:
        try:
            summary = run_config(config, args.force)
            if summary is not None:
                all_summaries.append(summary.head(30))
        except Exception as exc:
            failures.append({"config_id": config.config_id, "error": repr(exc)})
            log(f"FAILED {config.config_id}: {exc!r}")
            pd.DataFrame(failures).to_csv(OUT_DIR / "failed_shap_configs.csv", index=False)

    if all_summaries:
        pd.concat(all_summaries, ignore_index=True).to_csv(OUT_DIR / "combined_top30_feature_contributions.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(OUT_DIR / "failed_shap_configs.csv", index=False)
    combine_outputs()
    log(f"Done. SHAP continuation outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
