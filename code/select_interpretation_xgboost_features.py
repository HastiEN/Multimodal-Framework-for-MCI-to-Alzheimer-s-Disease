"""Reproduce the early XGBoost SHAP-based subset-selection experiment."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
DATA_PATH = PROCESSED_DIR / "baseline_mci_to_ad_selected_features_dataset.csv"
PLOT_DIR = PROCESSED_DIR / "shap_plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_PATH = PROCESSED_DIR / "shap_feature_ranking.csv"
SUBSET_PERFORMANCE_PATH = PROCESSED_DIR / "shap_subset_performance.csv"
TOP20_PATH = PROCESSED_DIR / "shap_top20_features.csv"
REPORT_PATH = PROCESSED_DIR / "shap_feature_selection_report.md"
JSON_REPORT_PATH = PROCESSED_DIR / "shap_feature_selection_report.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
SUBSET_SIZES = [10, 20, 30, 50, 75, 100]
LEAKAGE_COLUMNS = {
    "PTID",
    "RID",
    "conversion_date",
    "months_to_conversion",
    "last_followup_date",
    "n_followup_visits",
    "baseline_date",
    "baseline_visit",
}


def log(message: str = "") -> None:
    print(message, flush=True)


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


def split_feature_types(x: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = x.select_dtypes(include=[np.number]).columns.tolist()
    categorical = [col for col in x.columns if col not in numeric]
    return numeric, categorical


def make_preprocessor(x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric, categorical = split_feature_types(x)
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
    return preprocessor, numeric, categorical


def transformed_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        names = []
        for _, _, cols in preprocessor.transformers_:
            names.extend(list(cols))
        return names


def map_transformed_to_original(name: str, original_features: list[str], categorical_features: list[str]) -> str:
    if name in original_features:
        return name
    for col in sorted(categorical_features, key=len, reverse=True):
        if name.startswith(f"{col}_"):
            return col
    return name


def evaluate(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float | str]:
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
        "confusion_matrix": json.dumps([[int(tn), int(fp)], [int(fn), int(tp)]]),
    }


def train_xgb_raw_subset(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    features: list[str],
) -> tuple[Pipeline, dict[str, float | str]]:
    preprocessor, _, _ = make_preprocessor(x_train[features])
    pipe = Pipeline([("preprocess", preprocessor), ("model", make_xgb())])
    pipe.fit(x_train[features], y_train)
    y_score = pipe.predict_proba(x_test[features])[:, 1]
    return pipe, evaluate(y_test, y_score)


def get_shap_matrix(pipe: Pipeline, x: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[str], list[str]]:
    preprocessor = pipe.named_steps["preprocess"]
    x_transformed = preprocessor.transform(x)
    names = transformed_names(preprocessor)
    x_transformed_df = pd.DataFrame(x_transformed, columns=names, index=x.index)
    _, categorical = split_feature_types(x)
    explainer = shap.TreeExplainer(pipe.named_steps["model"])
    shap_values = explainer.shap_values(x_transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return np.asarray(shap_values), x_transformed_df, names, categorical


def aggregate_shap_to_original(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    original_features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    rows = []
    for idx, transformed in enumerate(transformed_feature_names):
        rows.append(
            {
                "feature": map_transformed_to_original(transformed, original_features, categorical_features),
                "transformed_feature": transformed,
                "mean_abs_shap": float(np.abs(shap_values[:, idx]).mean()),
            }
        )
    transformed_rank = pd.DataFrame(rows)
    ranking = (
        transformed_rank.groupby("feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking[["rank", "feature", "mean_abs_shap"]]


def save_shap_plots(
    shap_values: np.ndarray,
    x_transformed_df: pd.DataFrame,
    transformed_feature_names: list[str],
    ranking: pd.DataFrame,
) -> None:
    shap.summary_plot(shap_values, x_transformed_df, show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_summary_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x_transformed_df, plot_type="bar", show=False, max_display=30)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_bar_top30.png", dpi=300, bbox_inches="tight")
    plt.close()

    explanation = shap.Explanation(
        values=shap_values,
        data=x_transformed_df.values,
        feature_names=transformed_feature_names,
    )
    shap.plots.beeswarm(explanation, max_display=30, show=False)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / "shap_beeswarm_plot.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_features = ranking["feature"].head(10).tolist()
    for feature in top_features:
        matching = [name for name in transformed_feature_names if name == feature or name.startswith(f"{feature}_")]
        if not matching:
            continue
        transformed = matching[0]
        try:
            shap.dependence_plot(
                transformed,
                shap_values,
                x_transformed_df,
                show=False,
                interaction_index=None,
            )
            plt.tight_layout()
            safe_name = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in feature)
            plt.savefig(PLOT_DIR / f"shap_dependence_{safe_name}.png", dpi=300, bbox_inches="tight")
            plt.close()
        except Exception as exc:
            log(f"Dependence plot skipped for {feature}: {exc}")


def cv_best_subset(dataset: pd.DataFrame, features: list[str]) -> dict[str, float]:
    x = dataset[features].copy()
    y = dataset["conversion_label"].astype(int).to_numpy()
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    metric_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        _, metrics = train_xgb_raw_subset(x_train, x_test, y_train, y_test, features)
        metrics["fold"] = fold_idx
        metric_rows.append(metrics)
    folds = pd.DataFrame(metric_rows)
    folds.to_csv(PROCESSED_DIR / "shap_best_subset_repeated_cv_folds.csv", index=False)
    return {
        "cv_roc_auc_mean": float(folds["roc_auc"].mean()),
        "cv_roc_auc_std": float(folds["roc_auc"].std(ddof=1)),
        "cv_pr_auc_mean": float(folds["pr_auc"].mean()),
        "cv_f1_mean": float(folds["f1"].mean()),
        "cv_recall_mean": float(folds["recall"].mean()),
        "cv_specificity_mean": float(folds["specificity"].mean()),
        "cv_balanced_accuracy_mean": float(folds["balanced_accuracy"].mean()),
    }


def interpret_feature(feature: str) -> str:
    upper = feature.upper()
    if "ADAS" in upper or "TOTSCORE" in upper or "TOTAL13" in upper:
        return "Higher impairment on ADAS-Cog generally indicates greater Alzheimer-type cognitive burden and higher conversion risk."
    if "FAQ" in upper:
        return "Greater functional impairment is clinically consistent with increased probability of future AD conversion."
    if "CDR" in upper:
        return "Higher CDR severity or domain impairment reflects worse clinical staging and usually increases conversion risk."
    if "MMSE" in upper or "MMSCORE" in upper:
        return "Lower global cognition is expected to increase conversion risk; direction can be inverse depending on score coding."
    if "AMY" in upper or "CENTILOIDS" in upper or "SUVR" in upper:
        return "Higher amyloid burden is biologically consistent with increased future AD conversion risk."
    if "NEUROBAT" in upper:
        return "Neuropsychological memory, naming, and executive scores capture baseline cognitive vulnerability to conversion."
    if "MOCA" in upper:
        return "MoCA domain performance reflects global and domain-specific cognitive impairment."
    if "VITALS" in upper or "BMI" in upper:
        return "Vital-sign associations may reflect systemic health, frailty, or cohort effects and should be interpreted cautiously."
    if "PTDEMOG" in upper or "AGE" in upper or "SEX" in upper or "EDUC" in upper:
        return "Demographic risk and reserve factors may modify observed conversion risk."
    return "Potential nonlinear or interaction effect; inspect dependence plots before assigning a directional clinical interpretation."


def df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    text_df = df.copy()
    for col in text_df.columns:
        if pd.api.types.is_float_dtype(text_df[col]):
            text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
        else:
            text_df[col] = text_df[col].astype(str)
    header = "| " + " | ".join(text_df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy().tolist()]
    return "\n".join([header, separator] + rows)


def write_markdown_report(
    dataset: pd.DataFrame,
    baseline_metrics: dict[str, float | str],
    ranking: pd.DataFrame,
    subset_results: pd.DataFrame,
    best_row: pd.Series,
    within_001: pd.Series,
    within_0005: pd.Series,
    cv_summary: dict[str, float],
) -> None:
    top20 = ranking.head(20).copy()
    lines = [
        "# SHAP-Based XGBoost Feature Selection Report",
        "",
        "## Dataset Summary",
        f"- Subjects: {len(dataset)}",
        f"- Converters: {int((dataset['conversion_label'] == 1).sum())}",
        f"- Stable MCI: {int((dataset['conversion_label'] == 0).sum())}",
        f"- Candidate baseline features: {len([c for c in dataset.columns if c not in LEAKAGE_COLUMNS and c != 'conversion_label'])}",
        "",
        "## Baseline XGBoost Holdout Performance",
        f"- ROC-AUC: {baseline_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {baseline_metrics['pr_auc']:.4f}",
        f"- Accuracy: {baseline_metrics['accuracy']:.4f}",
        f"- Precision: {baseline_metrics['precision']:.4f}",
        f"- Recall/Sensitivity: {baseline_metrics['recall']:.4f}",
        f"- Specificity: {baseline_metrics['specificity']:.4f}",
        f"- F1: {baseline_metrics['f1']:.4f}",
        "",
        "## Top 20 SHAP Features",
        df_to_markdown(top20),
        "",
        "## Feature Count Comparison",
        df_to_markdown(subset_results),
        "",
        "## Optimal Feature Subsets",
        f"- Highest ROC-AUC subset: Top {int(best_row['n_features'])} features, ROC-AUC={best_row['roc_auc']:.4f}.",
        f"- Smallest subset within 0.01 ROC-AUC of best: Top {int(within_001['n_features'])} features, ROC-AUC={within_001['roc_auc']:.4f}.",
        f"- Smallest subset within 0.005 ROC-AUC of best: Top {int(within_0005['n_features'])} features, ROC-AUC={within_0005['roc_auc']:.4f}.",
        "",
        "## Repeated CV for Best SHAP Subset",
        f"- ROC-AUC: {cv_summary['cv_roc_auc_mean']:.4f} +/- {cv_summary['cv_roc_auc_std']:.4f}",
        f"- PR-AUC: {cv_summary['cv_pr_auc_mean']:.4f}",
        f"- F1: {cv_summary['cv_f1_mean']:.4f}",
        f"- Recall/Sensitivity: {cv_summary['cv_recall_mean']:.4f}",
        f"- Specificity: {cv_summary['cv_specificity_mean']:.4f}",
        "",
        "## Biomedical Interpretation",
    ]
    for row in top20.itertuples(index=False):
        lines.append(f"- {row.feature}: {interpret_feature(row.feature)}")
    lines.extend(
        [
            "",
            "## Recommended Final Model",
            f"Use XGBoost with the Top {int(within_0005['n_features'])} SHAP-ranked features if the goal is parsimony with near-best discrimination. Use the Top {int(best_row['n_features'])} subset if maximizing holdout ROC-AUC is the priority.",
            "",
            "All imputation and encoding were fitted inside model pipelines. The original dataset was not modified.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    dataset = pd.read_csv(DATA_PATH)
    y = dataset["conversion_label"].astype(int).to_numpy()
    feature_cols = [col for col in dataset.columns if col not in LEAKAGE_COLUMNS and col != "conversion_label"]
    x = dataset[feature_cols].copy()

    log("====================================================")
    log("STEP 1 - DATASET")
    log("====================================================")
    log(f"Subjects: {len(dataset)}")
    log(f"Converters: {int((dataset['conversion_label'] == 1).sum())}")
    log(f"Stable MCI: {int((dataset['conversion_label'] == 0).sum())}")
    log(f"Features: {len(feature_cols)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    log("\n====================================================")
    log("STEP 2 - BASELINE XGBOOST")
    log("====================================================")
    baseline_pipe, baseline_metrics = train_xgb_raw_subset(x_train, x_test, y_train, y_test, feature_cols)
    log(json.dumps(baseline_metrics, indent=2))

    log("\n====================================================")
    log("STEP 3 - SHAP ANALYSIS")
    log("====================================================")
    shap_values, x_transformed_df, transformed_feature_names, categorical_features = get_shap_matrix(
        baseline_pipe,
        x_train,
    )
    ranking = aggregate_shap_to_original(
        shap_values,
        transformed_feature_names,
        feature_cols,
        categorical_features,
    )
    ranking.to_csv(RANKING_PATH, index=False)
    ranking.head(20).to_csv(TOP20_PATH, index=False)

    log("\nTop 50 SHAP features:")
    log(ranking.head(50).to_string(index=False))

    log("\n====================================================")
    log("STEP 4 - SHAP PLOTS")
    log("====================================================")
    save_shap_plots(shap_values, x_transformed_df, transformed_feature_names, ranking)
    log(f"Saved SHAP plots to: {PLOT_DIR}")

    log("\n====================================================")
    log("STEP 5 - FEATURE SELECTION EXPERIMENT")
    log("====================================================")
    subset_rows = []
    actual_sizes = [size for size in SUBSET_SIZES if size < len(feature_cols)] + [len(feature_cols)]
    for size in actual_sizes:
        selected = ranking["feature"].head(size).tolist()
        _, metrics = train_xgb_raw_subset(x_train, x_test, y_train, y_test, selected)
        metrics.update(
            {
                "subset": f"Top {size}" if size < len(feature_cols) else "All features",
                "n_features": len(selected),
                "features": json.dumps(selected),
            }
        )
        subset_rows.append(metrics)
        log(
            f"Top {len(selected):>3}: ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, F1={metrics['f1']:.4f}"
        )

    subset_results = pd.DataFrame(subset_rows).sort_values(["roc_auc", "balanced_accuracy"], ascending=False)
    subset_results.to_csv(SUBSET_PERFORMANCE_PATH, index=False)

    best_auc = subset_results["roc_auc"].max()
    best_row = subset_results.iloc[0]
    within_001 = subset_results[subset_results["roc_auc"] >= best_auc - 0.01].sort_values("n_features").iloc[0]
    within_0005 = subset_results[subset_results["roc_auc"] >= best_auc - 0.005].sort_values("n_features").iloc[0]

    log("\n====================================================")
    log("STEP 6 - OPTIMAL FEATURE COUNT")
    log("====================================================")
    log(f"Highest ROC-AUC: Top {int(best_row['n_features'])}, ROC-AUC={best_row['roc_auc']:.4f}")
    log(
        f"Smallest within 0.01: Top {int(within_001['n_features'])}, "
        f"ROC-AUC={within_001['roc_auc']:.4f}"
    )
    log(
        f"Smallest within 0.005: Top {int(within_0005['n_features'])}, "
        f"ROC-AUC={within_0005['roc_auc']:.4f}"
    )

    log("\n====================================================")
    log("STEP 7 - REPEATED CV FOR BEST SHAP SUBSET")
    log("====================================================")
    best_features = json.loads(best_row["features"])
    cv_summary = cv_best_subset(dataset, best_features)
    log(json.dumps(cv_summary, indent=2))

    write_markdown_report(
        dataset=dataset,
        baseline_metrics=baseline_metrics,
        ranking=ranking,
        subset_results=subset_results.drop(columns=["features"]),
        best_row=best_row,
        within_001=within_001,
        within_0005=within_0005,
        cv_summary=cv_summary,
    )

    JSON_REPORT_PATH.write_text(
        json.dumps(
            {
                "baseline_metrics": baseline_metrics,
                "best_subset": best_row.to_dict(),
                "smallest_within_0_01": within_001.to_dict(),
                "smallest_within_0_005": within_0005.to_dict(),
                "cv_summary_best_subset": cv_summary,
                "outputs": {
                    "ranking": str(RANKING_PATH),
                    "subset_performance": str(SUBSET_PERFORMANCE_PATH),
                    "top20": str(TOP20_PATH),
                    "report": str(REPORT_PATH),
                    "plots": str(PLOT_DIR),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    log("\n====================================================")
    log("RECOMMENDED FINAL FEATURE SUBSET FOR PUBLICATION")
    log("====================================================")
    recommended_features = json.loads(within_0005["features"])
    log(f"Recommended subset size: {len(recommended_features)}")
    for idx, feature in enumerate(recommended_features, start=1):
        log(f"{idx:>2}. {feature}")
    log(f"\nSaved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
