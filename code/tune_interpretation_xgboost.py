"""Reproduce the separate Optuna tuning experiment for interpretation."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))
import project_paths as _paths

import json
import os
import pickle

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


try:
    import optuna
except ImportError as exc:
    raise SystemExit(
        "Optuna is not installed in this Python environment. Install it first with:\n"
        "  python -m pip install optuna\n"
        "or:\n"
        "  conda install -c conda-forge optuna\n"
        "Then rerun this script."
    ) from exc


PROJECT_DIR = _paths.PACKAGE_ROOT
PROCESSED_DIR = _paths.processed_dir()
DATA_PATH = PROCESSED_DIR / "baseline_mci_to_ad_top20_leakage_fixed_bmi_corrected.csv"

STUDY_PATH = PROCESSED_DIR / "optuna_xgb_study.pkl"
PLOTS_DIR = PROCESSED_DIR / "optuna_xgb_plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = PROCESSED_DIR / "optuna_xgb_report.md"
RESULTS_PATH = PROCESSED_DIR / "optuna_xgb_best_model_results.csv"
CV_FOLDS_PATH = PROCESSED_DIR / "optuna_xgb_repeated_cv_folds.csv"
COMPARISON_PATH = PROCESSED_DIR / "optuna_xgb_before_after_comparison.csv"
IMPORTANCE_PATH = PROCESSED_DIR / "optuna_xgb_feature_importance.csv"
SHAP_IMPORTANCE_PATH = PROCESSED_DIR / "optuna_xgb_shap_importance.csv"
TRIALS_PATH = PROCESSED_DIR / "optuna_xgb_trials.csv"
JSON_REPORT_PATH = PROCESSED_DIR / "optuna_xgb_report.json"

RANDOM_STATE = 42
N_TRIALS = int(os.getenv("OPTUNA_XGB_N_TRIALS", "300"))
N_JOBS = int(os.getenv("OPTUNA_XGB_N_JOBS", "1"))

BASELINE_PARAMS = {
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "eval_metric": "logloss",
    "n_jobs": -1,
}

LEAKAGE_COLUMNS = {
    "PTID",
    "RID",
    "baseline_date",
    "baseline_visit",
    "conversion_date",
    "months_to_conversion",
    "last_followup_date",
    "n_followup_visits",
}


def log(message: str = "") -> None:
    print(message, flush=True)


def one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    df = pd.read_csv(DATA_PATH)
    if "conversion_label" not in df.columns:
        raise ValueError("Dataset must contain conversion_label target column.")
    feature_cols = [col for col in df.columns if col not in LEAKAGE_COLUMNS and col != "conversion_label"]
    x = df[feature_cols].copy()
    y = df["conversion_label"].astype(int)
    return x, y, feature_cols


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


def make_pipeline(x: pd.DataFrame, params: dict) -> Pipeline:
    params = dict(params)
    params.setdefault("random_state", RANDOM_STATE)
    params.setdefault("eval_metric", "logloss")
    params.setdefault("n_jobs", -1)
    return Pipeline(
        [
            ("preprocess", make_preprocessor(x)),
            ("model", XGBClassifier(**params)),
        ]
    )


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


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.30, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "gamma": trial.suggest_float("gamma", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 20.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 20.0),
        "max_delta_step": trial.suggest_int("max_delta_step", 0, 10),
        "random_state": RANDOM_STATE,
        "eval_metric": "logloss",
        "n_jobs": -1,
    }


def optimize(x: pd.DataFrame, y: pd.Series) -> optuna.Study:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
            x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
            pipe = make_pipeline(x_train, params)
            pipe.fit(x_train, y_train)
            y_score = pipe.predict_proba(x_valid)[:, 1]
            scores.append(roc_auc_score(y_valid, y_score))
            trial.report(float(np.mean(scores)), step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(scores))

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, show_progress_bar=True)
    with STUDY_PATH.open("wb") as f:
        pickle.dump(study, f)
    study.trials_dataframe().to_csv(TRIALS_PATH, index=False)
    return study


def holdout_eval(x: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipe = make_pipeline(x_train, params)
    pipe.fit(x_train, y_train)
    y_score = pipe.predict_proba(x_test)[:, 1]
    metrics = evaluate(y_test.to_numpy(), y_score)
    return {"pipeline": pipe, "metrics": metrics, "x_train": x_train, "x_test": x_test, "y_train": y_train, "y_test": y_test}


def repeated_cv_eval(x: pd.DataFrame, y: pd.Series, params: dict) -> tuple[dict, pd.DataFrame]:
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
    rows = []
    for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x, y), start=1):
        x_train, x_valid = x.iloc[train_idx], x.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]
        pipe = make_pipeline(x_train, params)
        pipe.fit(x_train, y_train)
        y_score = pipe.predict_proba(x_valid)[:, 1]
        metrics = evaluate(y_valid.to_numpy(), y_score)
        metrics["fold"] = fold_idx
        rows.append(metrics)
    folds = pd.DataFrame(rows)
    folds.to_csv(CV_FOLDS_PATH, index=False)
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


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return preprocessor.get_feature_names_out().tolist()
    except Exception:
        return []


def save_importance_and_plots(pipe: Pipeline, x_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = pipe.named_steps["model"]
    names = get_feature_names(pipe.named_steps["preprocess"])
    if not names:
        names = x_train.columns.tolist()

    importance = pd.DataFrame(
        {
            "feature": names[: len(model.feature_importances_)],
            "weight_importance": model.feature_importances_,
        }
    ).sort_values("weight_importance", ascending=False)

    booster = model.get_booster()
    gain_dict = booster.get_score(importance_type="gain")
    importance["gain_importance"] = importance["feature"].map(gain_dict).fillna(0.0)
    importance = importance.sort_values("gain_importance", ascending=False)
    importance.to_csv(IMPORTANCE_PATH, index=False)

    top_weight = importance.sort_values("weight_importance", ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top_weight["feature"][::-1], top_weight["weight_importance"][::-1])
    plt.xlabel("XGBoost Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_feature_importance_weight.png", dpi=300, bbox_inches="tight")
    plt.close()

    top_gain = importance.sort_values("gain_importance", ascending=False).head(20)
    plt.figure(figsize=(8, 6))
    plt.barh(top_gain["feature"][::-1], top_gain["gain_importance"][::-1])
    plt.xlabel("XGBoost Gain Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "xgb_feature_importance_gain.png", dpi=300, bbox_inches="tight")
    plt.close()

    x_transformed = pipe.named_steps["preprocess"].transform(x_train)
    x_transformed_df = pd.DataFrame(x_transformed, columns=names, index=x_train.index)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_transformed_df)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_importance = pd.DataFrame(
        {
            "feature": names,
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(SHAP_IMPORTANCE_PATH, index=False)

    shap.summary_plot(shap_values, x_transformed_df, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, x_transformed_df, plot_type="bar", show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close()

    return importance, shap_importance


def comparison_table(before: dict, after: dict) -> pd.DataFrame:
    metrics = ["roc_auc", "pr_auc", "f1", "recall", "specificity"]
    rows = []
    for metric in metrics:
        rows.append(
            {
                "Metric": metric,
                "Before": before[metric],
                "After": after[metric],
                "Delta": after[metric] - before[metric],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(COMPARISON_PATH, index=False)
    return out


def df_to_markdown(df: pd.DataFrame) -> str:
    text = df.copy()
    for col in text.columns:
        if pd.api.types.is_float_dtype(text[col]):
            text[col] = text[col].map(lambda x: f"{x:.4f}")
        else:
            text[col] = text[col].astype(str)
    header = "| " + " | ".join(text.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text.to_numpy().tolist()]
    return "\n".join([header, sep] + rows)


def write_report(
    x: pd.DataFrame,
    y: pd.Series,
    study: optuna.Study,
    baseline_metrics: dict,
    optimized_metrics: dict,
    cv_summary: dict,
    comparison: pd.DataFrame,
    importance: pd.DataFrame,
    shap_importance: pd.DataFrame,
) -> None:
    lines = [
        "# Optuna XGBoost Top-20 Tuning Report",
        "",
        "## Dataset Summary",
        f"- Dataset: `{DATA_PATH}`",
        f"- Subjects: {len(y)}",
        f"- Stable MCI: {int((y == 0).sum())}",
        f"- Converters: {int((y == 1).sum())}",
        f"- Features: {x.shape[1]}",
        "",
        "## Leakage-Free Validation Summary",
        "- Dataset is the leakage-fixed Top-20 SHAP feature table.",
        "- Unsafe source records beyond baseline + 180 days were removed before tuning.",
        "- Imputation and one-hot encoding occur inside sklearn pipelines.",
        "",
        "## Optuna Settings",
        f"- Trials: {N_TRIALS}",
        "- Sampler: TPESampler(seed=42)",
        "- Pruner: MedianPruner",
        "- Objective CV: RepeatedStratifiedKFold, 5 folds x 3 repeats",
        "- Optimization metric: mean ROC-AUC",
        "",
        "## Best Hyperparameters",
        "```json",
        json.dumps(study.best_params, indent=2),
        "```",
        f"- Best Optuna CV ROC-AUC: {study.best_value:.4f}",
        "",
        "## Test Performance",
        f"- ROC-AUC: {optimized_metrics['roc_auc']:.4f}",
        f"- PR-AUC: {optimized_metrics['pr_auc']:.4f}",
        f"- Accuracy: {optimized_metrics['accuracy']:.4f}",
        f"- Precision: {optimized_metrics['precision']:.4f}",
        f"- Recall: {optimized_metrics['recall']:.4f}",
        f"- Specificity: {optimized_metrics['specificity']:.4f}",
        f"- F1: {optimized_metrics['f1']:.4f}",
        f"- Balanced Accuracy: {optimized_metrics['balanced_accuracy']:.4f}",
        "",
        "## Repeated CV Performance",
        f"- ROC-AUC: {cv_summary['cv_roc_auc_mean']:.4f} +/- {cv_summary['cv_roc_auc_std']:.4f}",
        f"- PR-AUC: {cv_summary['cv_pr_auc_mean']:.4f}",
        f"- F1: {cv_summary['cv_f1_mean']:.4f}",
        f"- Recall: {cv_summary['cv_recall_mean']:.4f}",
        f"- Specificity: {cv_summary['cv_specificity_mean']:.4f}",
        "",
        "## Performance Improvement",
        df_to_markdown(comparison),
        "",
        "## Top Feature Importance",
        df_to_markdown(importance.head(20)),
        "",
        "## Top SHAP Importance",
        df_to_markdown(shap_importance.head(20)),
        "",
        "## Publication Recommendation",
        "Use optimized XGBoost only if it improves repeated-CV ROC-AUC or offers a clinically meaningful holdout gain without recall/specificity trade-offs. Otherwise, keep the simpler leakage-fixed Top-20 XGBoost as the primary model and report Optuna tuning as a robustness analysis.",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    x, y, feature_cols = load_data()
    log("====================================================")
    log("OPTUNA XGBOOST TOP-20 TUNING")
    log("====================================================")
    log(f"Subjects: {len(y)}")
    log(f"Converters: {int((y == 1).sum())}")
    log(f"Stable MCI: {int((y == 0).sum())}")
    log(f"Features: {len(feature_cols)}")
    log(f"Trials: {N_TRIALS}")

    log("\nTraining baseline XGBoost for before/after comparison...")
    baseline_eval = holdout_eval(x, y, BASELINE_PARAMS)
    baseline_metrics = baseline_eval["metrics"]

    log("\nStarting Optuna optimization...")
    study = optimize(x, y)
    best_params = dict(study.best_params)
    best_params.update({"random_state": RANDOM_STATE, "eval_metric": "logloss", "n_jobs": -1})

    log("\nTraining optimized holdout model...")
    optimized_eval = holdout_eval(x, y, best_params)
    optimized_metrics = optimized_eval["metrics"]

    log("\nRunning repeated CV with optimized parameters...")
    cv_summary, _ = repeated_cv_eval(x, y, best_params)

    comparison = comparison_table(baseline_metrics, optimized_metrics)
    importance, shap_importance = save_importance_and_plots(optimized_eval["pipeline"], optimized_eval["x_train"])

    pd.DataFrame(
        [
            {"model": "baseline_xgboost", **baseline_metrics},
            {"model": "optimized_xgboost", **optimized_metrics},
        ]
    ).to_csv(RESULTS_PATH, index=False)

    JSON_REPORT_PATH.write_text(
        json.dumps(
            {
                "best_params": study.best_params,
                "best_cv_roc_auc": study.best_value,
                "baseline_metrics": baseline_metrics,
                "optimized_metrics": optimized_metrics,
                "cv_summary": cv_summary,
                "outputs": {
                    "study": str(STUDY_PATH),
                    "report": str(REPORT_PATH),
                    "plots": str(PLOTS_DIR),
                    "results": str(RESULTS_PATH),
                    "comparison": str(COMPARISON_PATH),
                },
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    write_report(
        x=x,
        y=y,
        study=study,
        baseline_metrics=baseline_metrics,
        optimized_metrics=optimized_metrics,
        cv_summary=cv_summary,
        comparison=comparison,
        importance=importance,
        shap_importance=shap_importance,
    )

    log("\n====================================================")
    log("FINAL RECOMMENDATION")
    log("====================================================")
    log("Best Parameters:")
    log(json.dumps(study.best_params, indent=2))
    log(f"Best Optuna CV ROC-AUC: {study.best_value:.4f}")
    log(f"Best Test ROC-AUC: {optimized_metrics['roc_auc']:.4f}")
    replace = optimized_metrics["roc_auc"] > baseline_metrics["roc_auc"] and cv_summary["cv_roc_auc_mean"] >= 0.8396
    if replace:
        log("Expected publication model: Optimized XGBoost should replace the current model.")
    else:
        log("Expected publication model: Keep current leakage-fixed XGBoost unless optimized CV performance is clearly superior.")
    log(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
