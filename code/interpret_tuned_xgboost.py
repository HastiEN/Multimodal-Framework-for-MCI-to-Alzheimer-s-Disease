"""Reproduce the legacy tuned interpretation; dataset caveats are documented."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
DATA_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed.csv"
OPTUNA_REPORT_JSON = PROCESSED_DIR / "optuna_xgb_report.json"

OUT_DIR = PROCESSED_DIR / "shap_interpretation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = PROCESSED_DIR / "shap_interpretation_report.md"
TOP20_IMPORTANCE_PATH = OUT_DIR / "top20_shap_importance.csv"
EFFECT_PATH = OUT_DIR / "top20_shap_effect_direction.csv"
THRESHOLD_PATH = OUT_DIR / "risk_transition_thresholds.csv"

RANDOM_STATE = 42
DEPENDENCE_FEATURES = [
    "ADAS_TOTAL13",
    "FAQ_FAQTOTAL",
    "CDR_CDRSB",
    "AMY_SUMMARY_SUVR",
    "AMY_CENTILOIDS",
    "VITALS_BMI",
    "PTDEMOG_PTEDUCAT",
]
THRESHOLD_FEATURES = [
    "ADAS_TOTAL13",
    "FAQ_FAQTOTAL",
    "CDR_CDRSB",
    "AMY_SUMMARY_SUVR",
    "AMY_CENTILOIDS",
]


def log(message: str = "") -> None:
    print(message, flush=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_best_params() -> dict:
    if not OPTUNA_REPORT_JSON.exists():
        raise FileNotFoundError(f"Missing Optuna report JSON: {OPTUNA_REPORT_JSON}")
    report = json.loads(OPTUNA_REPORT_JSON.read_text(encoding="utf-8"))
    params = dict(report["best_params"])
    params.update(
        {
            "random_state": RANDOM_STATE,
            "eval_metric": "logloss",
            "n_jobs": -1,
        }
    )
    return params


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return ColumnTransformer(
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


def get_feature_names(preprocessor: ColumnTransformer, fallback: list[str]) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return fallback


def train_final_model(x: pd.DataFrame, y: pd.Series, params: dict) -> Pipeline:
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(x)),
            ("model", XGBClassifier(**params)),
        ]
    )
    pipe.fit(x, y)
    return pipe


def compute_shap(pipe: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    x_transformed = pipe.named_steps["preprocess"].transform(x)
    feature_names = get_feature_names(pipe.named_steps["preprocess"], x.columns.tolist())
    x_transformed_df = pd.DataFrame(x_transformed, columns=feature_names, index=x.index)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(x_transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values), x_transformed_df, feature_names


def summarize_shap(shap_values: np.ndarray, x_transformed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, feature in enumerate(x_transformed_df.columns):
        values = x_transformed_df[feature]
        svals = shap_values[:, idx]
        rows.append(
            {
                "feature": feature,
                "mean_shap": float(np.mean(svals)),
                "mean_abs_shap": float(np.mean(np.abs(svals))),
                "median_shap": float(np.median(svals)),
                "feature_min": float(np.nanmin(values)),
                "feature_max": float(np.nanmax(values)),
                "feature_median": float(np.nanmedian(values)),
            }
        )
    out = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out[["rank", "feature", "mean_abs_shap", "mean_shap", "median_shap", "feature_min", "feature_max", "feature_median"]]


def direction_of_effect(feature_values: pd.Series, shap_feature_values: np.ndarray) -> tuple[str, float, str]:
    valid = feature_values.notna() & pd.Series(shap_feature_values, index=feature_values.index).notna()
    if valid.sum() < 20:
        return "insufficient data", np.nan, "Too few observed values for stable direction inference."

    x = feature_values.loc[valid].astype(float)
    s = pd.Series(shap_feature_values, index=feature_values.index).loc[valid].astype(float)
    corr = x.corr(s, method="spearman")
    low = s[x <= x.quantile(0.25)].mean()
    high = s[x >= x.quantile(0.75)].mean()
    delta = high - low

    # Nonlinearity heuristic: middle quantiles differ from a monotonic low-high trend.
    q = pd.qcut(x.rank(method="first"), q=5, labels=False)
    bin_means = pd.DataFrame({"bin": q, "shap": s}).groupby("bin")["shap"].mean().to_numpy()
    sign_changes = int(np.sum(np.diff(np.sign(np.diff(bin_means))) != 0)) if len(bin_means) >= 3 else 0

    if abs(corr) < 0.15 or sign_changes >= 2:
        direction = "nonlinear or weak monotonic effect"
    elif corr > 0 and delta > 0:
        direction = "higher values increase AD conversion risk"
    elif corr < 0 and delta < 0:
        direction = "higher values decrease AD conversion risk"
    else:
        direction = "mixed or interaction-dependent effect"
    rationale = f"Spearman(feature, SHAP)={corr:.3f}; high-vs-low SHAP delta={delta:.3f}."
    return direction, float(corr), rationale


def estimate_threshold(feature_values: pd.Series, shap_feature_values: np.ndarray) -> dict:
    valid = feature_values.notna() & pd.Series(shap_feature_values, index=feature_values.index).notna()
    feature = feature_values.name
    if valid.sum() < 30:
        return {
            "feature": feature,
            "approx_transition_value": np.nan,
            "method": "insufficient observed values",
            "note": "No stable threshold estimated.",
        }
    x = feature_values.loc[valid].astype(float)
    s = pd.Series(shap_feature_values, index=feature_values.index).loc[valid].astype(float)
    order = np.argsort(x.to_numpy())
    xs = x.to_numpy()[order]
    ss = s.to_numpy()[order]

    n_bins = min(20, max(5, int(np.sqrt(len(xs)))))
    bins = pd.qcut(pd.Series(xs).rank(method="first"), q=n_bins, labels=False)
    binned = pd.DataFrame({"x": xs, "shap": ss, "bin": bins}).groupby("bin", as_index=False).agg(
        x_median=("x", "median"),
        shap_mean=("shap", "mean"),
    )
    candidates = binned.iloc[(binned["shap_mean"].abs()).argsort()]
    transition = float(candidates.iloc[0]["x_median"])
    note = "Approximate SHAP zero-crossing/lowest absolute mean SHAP bin; use as exploratory threshold, not a clinical cutoff."
    return {
        "feature": feature,
        "approx_transition_value": transition,
        "method": "binned SHAP zero-transition estimate",
        "note": note,
    }


def clinical_interpretation(feature: str, direction: str) -> tuple[str, str, str]:
    f = feature.upper()
    if "ADAS" in f:
        stat = "ADAS contributes strongly to model output, with larger impairment scores generally shifting predictions toward conversion."
        clinical = "Higher ADAS-Cog burden reflects worse baseline cognitive impairment."
        ad = "This is consistent with prodromal AD progression, where baseline cognitive severity predicts conversion."
    elif "FAQ" in f:
        stat = "FAQ features quantify functional impairment and carry substantial SHAP contribution."
        clinical = "Higher FAQ scores indicate greater difficulty in daily activities."
        ad = "Functional decline in MCI is clinically associated with imminent AD dementia conversion."
    elif "CDR" in f:
        stat = "CDR features capture clinical staging information at baseline."
        clinical = "Higher CDRSB or judgment impairment indicates more severe clinical symptoms."
        ad = "Greater baseline clinical dementia rating burden is expected in MCI subjects closer to AD conversion."
    elif "AMY" in f or "CENTILOIDS" in f or "SUVR" in f:
        stat = "Amyloid PET features contribute biomarker evidence beyond clinical testing."
        clinical = "Higher amyloid SUVR/Centiloids indicate greater amyloid plaque burden."
        ad = "Amyloid positivity is a core AD pathophysiological marker and supports biological plausibility."
    elif "NEUROBAT" in f:
        stat = "Neuropsychological test performance contributes domain-specific cognitive information."
        clinical = "Naming, memory, and executive task performance reflect vulnerable cognitive networks."
        ad = "Deficits in episodic memory, language, and executive function often precede AD dementia."
    elif "MOCA" in f:
        stat = "MoCA delayed recall/domain performance contributes additional cognitive signal."
        clinical = "Lower MoCA recall performance suggests memory consolidation impairment."
        ad = "Memory decline is a hallmark of prodromal AD."
    elif "PTEDUCAT" in f:
        stat = "Education contributes as a reserve/demographic modifier."
        clinical = "Education is commonly interpreted as a proxy for cognitive reserve."
        ad = "Cognitive reserve can alter the clinical expression of AD pathology."
    elif "BMI" in f or "VITALS" in f:
        stat = "Vital-sign features show model contribution but require cautious interpretation."
        clinical = "BMI, blood pressure, and weight may reflect vascular/metabolic health or frailty."
        ad = "Systemic health can modify cognitive trajectory, but these associations may be cohort-dependent."
    else:
        stat = "Feature has measurable SHAP contribution."
        clinical = "Clinical meaning should be interpreted with domain-specific context."
        ad = "Relationship to AD progression may be indirect or interaction-dependent."

    stat = f"{stat} Direction: {direction}."
    return stat, clinical, ad


def save_plots(shap_values: np.ndarray, x_transformed_df: pd.DataFrame) -> None:
    shap.summary_plot(shap_values, x_transformed_df, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(values=shap_values, data=x_transformed_df.values, feature_names=x_transformed_df.columns.tolist())
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_beeswarm_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x_transformed_df, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "shap_bar_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    for feature in DEPENDENCE_FEATURES:
        if feature not in x_transformed_df.columns:
            continue
        shap.dependence_plot(feature, shap_values, x_transformed_df, show=False, interaction_index=None)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"dependence_{feature}.png", dpi=300, bbox_inches="tight")
        plt.close()


def df_to_markdown(df: pd.DataFrame) -> str:
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy().tolist()]
    return "\n".join([header, sep] + rows)


def write_report(
    shap_summary: pd.DataFrame,
    effect_table: pd.DataFrame,
    threshold_table: pd.DataFrame,
    best_params: dict,
) -> None:
    top20 = shap_summary.head(20)
    lines = [
        "# SHAP Interpretation Report: Optimized XGBoost",
        "",
        "## Model And Dataset",
        f"- Dataset: `{DATA_PATH}`",
        "- Cohort: 1,255 leakage-fixed baseline MCI subjects.",
        "- Target: future conversion to AD.",
        "- Model: Optuna-optimized XGBoost using the final Top-20 leakage-free features.",
        "",
        "## Best Hyperparameters",
        "```json",
        json.dumps({k: v for k, v in best_params.items() if k not in {'random_state', 'eval_metric', 'n_jobs'}}, indent=2),
        "```",
        "",
        "## Top 20 Features By Mean Absolute SHAP",
        df_to_markdown(top20),
        "",
        "## Direction Of Effect",
        df_to_markdown(effect_table[["feature", "direction_of_effect", "spearman_feature_shap", "rationale"]]),
        "",
        "## Approximate Risk Transition Thresholds",
        df_to_markdown(threshold_table),
        "",
        "## Biomedical Interpretation",
    ]
    for row in effect_table.itertuples(index=False):
        stat, clinical, ad = clinical_interpretation(row.feature, row.direction_of_effect)
        lines.extend(
            [
                f"### {row.feature}",
                f"- Statistical interpretation: {stat}",
                f"- Clinical interpretation: {clinical}",
                f"- Relation to AD progression: {ad}",
                "",
            ]
        )
    lines.extend(
        [
            "## Suggested Results Text",
            "In the optimized XGBoost model, SHAP analysis identified cognitive severity, amyloid PET burden, functional impairment, CDR clinical staging, neuropsychological performance, education, and vital-sign measures as the dominant predictors of future MCI-to-AD conversion. The highest-ranked features included ADAS_TOTAL13, FAQ_FAQTOTAL, CDR_CDRSB, AMY_SUMMARY_SUVR, and AMY_CENTILOIDS, supporting a multimodal interpretation in which baseline cognitive-functional impairment and amyloid pathology jointly inform conversion risk.",
            "",
            "## Suggested Discussion Text",
            "The SHAP profile is clinically plausible for prodromal Alzheimer's disease. Cognitive and functional measures capture the current clinical severity of MCI, while amyloid SUVR and Centiloids provide biological evidence of AD pathology. CDR and neuropsychological variables further refine risk by measuring domain-specific deficits. Vitals and education should be interpreted cautiously: they may represent systemic health, frailty, vascular/metabolic burden, or cognitive reserve rather than direct AD pathology.",
            "",
            "## Publication Recommendation",
            "Use the SHAP interpretation as the primary explainability analysis for the optimized XGBoost model, while emphasizing that thresholds are exploratory and should not be treated as clinical decision cutoffs without external validation.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    y = data["conversion_label"].astype(int)
    x = data.drop(columns=["conversion_label"])
    best_params = load_best_params()

    log("====================================================")
    log("FINAL OPTIMIZED XGBOOST SHAP INTERPRETATION")
    log("====================================================")
    log(f"Subjects: {len(data)}")
    log(f"Converters: {int((y == 1).sum())}")
    log(f"Stable MCI: {int((y == 0).sum())}")
    log(f"Features: {x.shape[1]}")

    pipe = train_final_model(x, y, best_params)
    shap_values, x_transformed_df, _ = compute_shap(pipe, x)
    shap_summary = summarize_shap(shap_values, x_transformed_df)
    shap_summary.to_csv(TOP20_IMPORTANCE_PATH, index=False)

    effect_rows = []
    feature_to_index = {feature: idx for idx, feature in enumerate(x_transformed_df.columns)}
    for _, row in shap_summary.iterrows():
        feature = row["feature"]
        shap_idx = feature_to_index[feature]
        direction, corr, rationale = direction_of_effect(x_transformed_df[feature], shap_values[:, shap_idx])
        stat, clinical, ad = clinical_interpretation(feature, direction)
        effect_rows.append(
            {
                "feature": feature,
                "rank": row["rank"],
                "mean_abs_shap": row["mean_abs_shap"],
                "direction_of_effect": direction,
                "spearman_feature_shap": corr,
                "rationale": rationale,
                "statistical_interpretation": stat,
                "clinical_interpretation": clinical,
                "ad_progression_interpretation": ad,
            }
        )
    effect_table = pd.DataFrame(effect_rows)
    effect_table.to_csv(EFFECT_PATH, index=False)

    threshold_rows = []
    for feature in THRESHOLD_FEATURES:
        if feature in feature_to_index:
            threshold_rows.append(estimate_threshold(x_transformed_df[feature], shap_values[:, feature_to_index[feature]]))
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(THRESHOLD_PATH, index=False)

    save_plots(shap_values, x_transformed_df)
    write_report(shap_summary, effect_table, threshold_table, best_params)

    def first_matching(prefixes: list[str]) -> str:
        for feature in shap_summary["feature"]:
            upper = feature.upper()
            if any(prefix in upper for prefix in prefixes):
                return feature
        return "None"

    log("\nTop 20 features:")
    log(shap_summary.head(20).to_string(index=False))
    log("\nMost important biomarker:")
    log(first_matching(["AMY", "CENTILOIDS", "SUVR"]))
    log("Most important cognitive feature:")
    log(first_matching(["ADAS", "CDR", "FAQ", "NEUROBAT", "MOCA"]))
    log("Most important PET feature:")
    log(first_matching(["AMY", "CENTILOIDS", "SUVR"]))
    log("Most important demographic feature:")
    log(first_matching(["PTDEMOG", "PTEDUCAT", "AGE", "SEX"]))
    log("Most important vital-sign feature:")
    log(first_matching(["VITALS", "BMI"]))
    log(f"\nSaved SHAP interpretation report: {REPORT_PATH}")
    log(f"Saved plots and tables to: {OUT_DIR}")


if __name__ == "__main__":
    main()
