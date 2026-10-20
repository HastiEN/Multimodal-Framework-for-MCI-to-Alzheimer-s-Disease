"""Evaluate matched semantic and common-cohort conditions with fixed models."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

import argparse
import json
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from validate_nested_sensitivity import COHORT_PATH, DATA_PATH, FALLBACK_DATA_PATH, HORIZONS, MISSING_THRESHOLD, OUTER_FOLDS, OUTER_SEEDS, PROCESSED_DIR, SEMANTIC_FAMILIES, TOP_K_VALUES, build_pipeline, coerce_numeric_like, compute_metrics, is_leakage_feature, months_between, semantic_family, target_column


OUT_DIR = PROCESSED_DIR / "robustness_analysis_fast_no_nested"
SEMANTIC_DIR = OUT_DIR / "semantic_sensitivity"
COMMON_DIR = OUT_DIR / "common_cohort"
SUMMARY_DIR = OUT_DIR / "summary"

DEFAULT_SEMANTIC_MODEL = "CatBoost"
DEFAULT_SEMANTIC_TOP_K = 100
DEFAULT_COMMON_MODEL = "RandomForest"
DEFAULT_COMMON_TOP_K = 50


def log(message: str = "") -> None:
    print(message, flush=True)


def ensure_dirs() -> None:
    for path in [SEMANTIC_DIR, COMMON_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def load_modeling_data() -> tuple[pd.DataFrame, np.ndarray, pd.Series, pd.DataFrame, str]:
    path = DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH
    df = pd.read_csv(path, low_memory=False)
    target = target_column(df)
    if "PTID" not in df.columns:
        raise ValueError("Dataset must contain PTID for subject-level validation.")
    if df["PTID"].duplicated().any():
        raise ValueError("Dataset contains duplicated PTID values.")
    df["PTID"] = df["PTID"].astype(str)
    feature_cols = [col for col in df.columns if col != target and col != "PTID" and not is_leakage_feature(col)]
    x = coerce_numeric_like(df[feature_cols].copy())
    y = df[target].astype(int).to_numpy()
    ptid = df["PTID"].copy()
    return x, y, ptid, df, target


def semantic_exclusion_table(columns: list[str]) -> pd.DataFrame:
    rows = []
    for feature in columns:
        family = semantic_family(feature)
        if family:
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "exclusion_reason": SEMANTIC_FAMILIES[family],
                }
            )
    out = pd.DataFrame(rows).sort_values(["family", "feature"]).reset_index(drop=True)
    out.to_csv(SEMANTIC_DIR / "excluded_semantic_features.csv", index=False)
    return out


def fixed_repeated_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    ptid: pd.Series,
    model_name: str,
    top_k: int,
    condition: str,
    analysis_type: str,
    semantic_exclusions: tuple[str, ...] = (),
    horizon_months: int | None = None,
    split_y: np.ndarray | None = None,
    max_outer_folds: int | None = None,
) -> dict[str, pd.DataFrame]:
    predictions = []
    fold_rows = []
    selected_rows = []
    audit_rows = []
    split_rows = []
    split_labels = y if split_y is None else split_y
    fold_counter = 0

    for seed in OUTER_SEEDS:
        cv = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(x, split_labels), start=1):
            fold_counter += 1
            if max_outer_folds is not None and fold_counter > max_outer_folds:
                break
            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            train_ids = set(ptid.iloc[train_idx])
            test_ids = set(ptid.iloc[test_idx])
            assert train_ids.isdisjoint(test_ids)

            fit_seed = seed * 100 + fold
            pipe = build_pipeline(model_name, top_k, fit_seed, semantic_exclusions)
            pipe.fit(x_train, y_train)
            y_score = pipe.predict_proba(x_test)[:, 1]
            metrics = compute_metrics(y_test, y_score)
            selected_features = pipe.named_steps["feature_selector"].selected_features_

            base = {
                "analysis_type": analysis_type,
                "condition": condition,
                "horizon_months": horizon_months,
                "model": model_name,
                "top_k": top_k,
                "seed": seed,
                "fold": fold,
                "global_fold": fold_counter,
            }
            fold_rows.append({**base, **metrics, "n_selected_features": len(selected_features)})

            audit = pipe.named_steps["feature_selector"].audit_row()
            audit_rows.append({**base, **audit})

            for rank, feature in enumerate(selected_features, start=1):
                selected_rows.append({**base, "rank": rank, "feature": feature})

            for sid in sorted(train_ids):
                split_rows.append({**base, "PTID": sid, "split": "train"})
            for sid, true_label, probability in zip(ptid.iloc[test_idx], y_test, y_score):
                split_rows.append({**base, "PTID": sid, "split": "test"})
                predictions.append(
                    {
                        **base,
                        "PTID": sid,
                        "y_true": int(true_label),
                        "y_probability": float(probability),
                    }
                )
        if max_outer_folds is not None and fold_counter >= max_outer_folds:
            break

    return {
        "predictions": pd.DataFrame(predictions),
        "fold_results": pd.DataFrame(fold_rows),
        "selected_features": pd.DataFrame(selected_rows),
        "filter_audit": pd.DataFrame(audit_rows),
        "split_manifest": pd.DataFrame(split_rows),
    }


def summarize_folds(folds: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "accuracy",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "precision",
        "recall",
        "f1",
    ]
    summary = folds.groupby(group_cols)[metrics].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    return summary


def feature_frequency(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame()
    n_folds = selected[["seed", "fold"]].drop_duplicates().shape[0]
    out = (
        selected.groupby(["condition", "feature"], as_index=False)
        .agg(selection_count=("feature", "size"), mean_rank=("rank", "mean"))
        .sort_values(["condition", "selection_count", "mean_rank"], ascending=[True, False, True])
    )
    out["selection_frequency"] = out["selection_count"] / n_folds
    return out


def paired_semantic_outputs(full: dict[str, pd.DataFrame], restricted: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    full_pred = full["predictions"].rename(
        columns={
            "y_probability": "full_probability",
            "model": "full_model",
            "top_k": "full_top_k",
        }
    )
    restricted_pred = restricted["predictions"].rename(
        columns={
            "y_probability": "restricted_probability",
            "model": "restricted_model",
            "top_k": "restricted_top_k",
        }
    )
    paired = full_pred[
        ["PTID", "seed", "fold", "y_true", "full_probability", "full_model", "full_top_k"]
    ].merge(
        restricted_pred[
            ["PTID", "seed", "fold", "y_true", "restricted_probability", "restricted_model", "restricted_top_k"]
        ],
        on=["PTID", "seed", "fold", "y_true"],
        how="inner",
        validate="one_to_one",
    )

    rows = []
    for (seed, fold), group in paired.groupby(["seed", "fold"]):
        y_true = group["y_true"].to_numpy()
        full_metrics = compute_metrics(y_true, group["full_probability"].to_numpy())
        restricted_metrics = compute_metrics(y_true, group["restricted_probability"].to_numpy())
        row = {"seed": seed, "fold": fold}
        for metric in ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity", "precision", "recall", "f1"]:
            row[f"{metric}_full"] = full_metrics[metric]
            row[f"{metric}_restricted"] = restricted_metrics[metric]
            row[f"delta_{metric}"] = restricted_metrics[metric] - full_metrics[metric]
        rows.append(row)

    fold_delta = pd.DataFrame(rows)
    delta_cols = [col for col in fold_delta.columns if col.startswith("delta_")]
    summary = fold_delta[delta_cols].agg(["mean", "std"]).T.reset_index()
    summary.columns = ["metric", "mean_delta_restricted_minus_full", "std_delta"]
    return paired, fold_delta, summary


def run_semantic_sensitivity(
    x: pd.DataFrame,
    y: np.ndarray,
    ptid: pd.Series,
    model_name: str,
    top_k: int,
    max_outer_folds: int | None,
) -> pd.DataFrame:
    log("\nSEMANTIC SENSITIVITY")
    excluded = semantic_exclusion_table(x.columns.tolist())
    exclusions = tuple(excluded["feature"].tolist())
    log(f"Excluded semantic features: {len(exclusions)}")
    log(f"Fixed model/config: {model_name} Top-{top_k}")

    full = fixed_repeated_cv(
        x,
        y,
        ptid,
        model_name=model_name,
        top_k=top_k,
        condition="FULL",
        analysis_type="semantic_sensitivity",
        semantic_exclusions=(),
        max_outer_folds=max_outer_folds,
    )
    restricted = fixed_repeated_cv(
        x,
        y,
        ptid,
        model_name=model_name,
        top_k=top_k,
        condition="SEMANTIC_RESTRICTED",
        analysis_type="semantic_sensitivity",
        semantic_exclusions=exclusions,
        max_outer_folds=max_outer_folds,
    )

    for prefix, outputs in [("full", full), ("restricted", restricted)]:
        for name, df in outputs.items():
            df.to_csv(SEMANTIC_DIR / f"{prefix}_{name}.csv", index=False)

    paired, fold_delta, sensitivity_summary = paired_semantic_outputs(full, restricted)
    paired.to_csv(SEMANTIC_DIR / "semantic_paired_predictions.csv", index=False)
    fold_delta.to_csv(SEMANTIC_DIR / "semantic_sensitivity_fold_results.csv", index=False)
    sensitivity_summary.to_csv(SEMANTIC_DIR / "semantic_sensitivity_summary.csv", index=False)

    all_folds = pd.concat([full["fold_results"], restricted["fold_results"]], ignore_index=True)
    all_selected = pd.concat([full["selected_features"], restricted["selected_features"]], ignore_index=True)
    summary = summarize_folds(all_folds, ["condition", "model", "top_k"])
    summary.to_csv(SEMANTIC_DIR / "semantic_condition_summary.csv", index=False)
    feature_frequency(all_selected).to_csv(SEMANTIC_DIR / "semantic_feature_frequency.csv", index=False)
    log(summary.to_string(index=False))
    log("\nDelta restricted minus full:")
    log(sensitivity_summary.to_string(index=False))
    return summary


def build_common_cohort_dataset(full_df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort = pd.read_csv(COHORT_PATH)
    cohort["PTID"] = cohort["PTID"].astype(str)
    cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"], errors="coerce")
    cohort["conversion_date"] = pd.to_datetime(cohort["conversion_date"], errors="coerce")
    cohort["last_followup_date"] = pd.to_datetime(cohort["last_followup_date"], errors="coerce")
    cohort["conversion_label"] = pd.to_numeric(cohort["conversion_label"], errors="coerce").astype(int)
    cohort["months_to_conversion"] = pd.to_numeric(cohort["months_to_conversion"], errors="coerce")
    cohort["followup_months"] = months_between(cohort["last_followup_date"], cohort["baseline_date"])

    cohort["eligible_60m"] = (
        ((cohort["conversion_label"] == 1) & cohort["months_to_conversion"].notna())
        | ((cohort["conversion_label"] == 0) & (cohort["followup_months"] >= 60))
    )
    common = cohort[cohort["eligible_60m"]].copy()
    for horizon in HORIZONS:
        common[f"label_{horizon}m"] = np.where(
            (common["conversion_label"] == 1) & (common["months_to_conversion"] <= horizon),
            1,
            0,
        )

    members = common[
        [
            "PTID",
            "baseline_date",
            "conversion_date",
            "last_followup_date",
            "months_to_conversion",
            "followup_months",
            "eligible_60m",
            "label_24m",
            "label_36m",
            "label_60m",
        ]
    ].copy()
    members.to_csv(COMMON_DIR / "common_cohort_members.csv", index=False)

    distribution_rows = []
    for horizon in HORIZONS:
        y = common[f"label_{horizon}m"].astype(int)
        distribution_rows.append(
            {
                "horizon_months": horizon,
                "n_total": len(y),
                "positive": int((y == 1).sum()),
                "negative": int((y == 0).sum()),
                "prevalence": float((y == 1).mean()),
            }
        )
    distribution = pd.DataFrame(distribution_rows)
    assert distribution["n_total"].nunique() == 1
    distribution.to_csv(COMMON_DIR / "common_cohort_label_distribution.csv", index=False)

    dataset = full_df.merge(common[["PTID", "label_24m", "label_36m", "label_60m"]], on="PTID", how="inner")
    assert dataset["PTID"].nunique() == len(dataset)
    assert set(dataset["PTID"]) == set(common["PTID"])
    return dataset, distribution


def run_common_cohort(
    full_df: pd.DataFrame,
    target: str,
    model_name: str,
    top_k: int,
    max_outer_folds: int | None,
) -> pd.DataFrame:
    log("\nCOMMON-COHORT HORIZON ANALYSIS")
    common_df, distribution = build_common_cohort_dataset(full_df, target)
    base_features = [
        col
        for col in full_df.columns
        if col not in {"PTID", target, "label_24m", "label_36m", "label_60m"} and not is_leakage_feature(col)
    ]
    x = coerce_numeric_like(common_df[base_features].copy())
    ptid = common_df["PTID"].astype(str)
    split_y = common_df["label_60m"].astype(int).to_numpy()
    all_folds = []
    all_predictions = []
    all_selected = []

    log(f"Common cohort subjects: {len(common_df)}")
    log(distribution.to_string(index=False))
    log(f"Fixed model/config for all horizons: {model_name} Top-{top_k}")

    for horizon in HORIZONS:
        y = common_df[f"label_{horizon}m"].astype(int).to_numpy()
        outputs = fixed_repeated_cv(
            x,
            y,
            ptid,
            model_name=model_name,
            top_k=top_k,
            condition="COMMON_COHORT",
            analysis_type=f"common_{horizon}m",
            semantic_exclusions=(),
            horizon_months=horizon,
            split_y=split_y,
            max_outer_folds=max_outer_folds,
        )
        for name, df in outputs.items():
            df.to_csv(COMMON_DIR / f"common_{horizon}m_{name}.csv", index=False)
        all_folds.append(outputs["fold_results"])
        all_predictions.append(outputs["predictions"])
        all_selected.append(outputs["selected_features"])

    folds = pd.concat(all_folds, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    selected = pd.concat(all_selected, ignore_index=True)
    folds.to_csv(COMMON_DIR / "common_horizon_metrics.csv", index=False)
    predictions.to_csv(COMMON_DIR / "common_horizon_predictions.csv", index=False)
    selected.to_csv(COMMON_DIR / "common_horizon_selected_features.csv", index=False)
    summary = summarize_folds(folds, ["horizon_months", "model", "top_k"])
    summary = distribution.merge(summary, on="horizon_months", how="left")
    summary.to_csv(COMMON_DIR / "common_horizon_comparison.csv", index=False)
    feature_frequency(selected).to_csv(COMMON_DIR / "common_horizon_feature_frequency.csv", index=False)
    log(summary.to_string(index=False))
    return summary


def write_manifest(args: argparse.Namespace, x: pd.DataFrame, y: np.ndarray, target: str) -> None:
    try:
        import catboost
        import lightgbm
        import sklearn
        import xgboost

        versions: dict[str, Any] = {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "lightgbm": lightgbm.__version__,
            "catboost": catboost.__version__,
        }
    except Exception as exc:
        versions = {"version_collection_error": str(exc)}

    manifest = {
        "analysis_name": "fast_no_nested_semantic_and_common_cohort_sensitivity",
        "purpose": "Fast sensitivity analysis without nested CV; previous primary results remain frozen.",
        "input_dataset": str(DATA_PATH if DATA_PATH.exists() else FALLBACK_DATA_PATH),
        "cohort_file": str(COHORT_PATH),
        "target": target,
        "subjects": int(len(y)),
        "converters": int((y == 1).sum()),
        "stable_mci": int((y == 0).sum()),
        "candidate_features_before_fold_specific_filtering": int(x.shape[1]),
        "outer_seeds": OUTER_SEEDS,
        "outer_folds": OUTER_FOLDS,
        "cv_design": "Repeated 5-fold StratifiedKFold using seeds 42-51; no inner CV.",
        "semantic_model": args.semantic_model,
        "semantic_top_k": args.semantic_top_k,
        "common_model": args.common_model,
        "common_top_k": args.common_top_k,
        "semantic_restricted_families": SEMANTIC_FAMILIES,
        "top_k_values_available": TOP_K_VALUES,
        "missing_threshold_percent": MISSING_THRESHOLD,
        "hyperparameter_policy": "fixed pre-specified hyperparameters; no Optuna",
        "max_outer_folds_debug": args.max_outer_folds,
        "outputs": str(OUT_DIR),
        "package_versions": versions,
    }
    (SUMMARY_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_readme(semantic_summary: pd.DataFrame, common_summary: pd.DataFrame) -> None:
    lines = [
        "FAST ROBUSTNESS ANALYSIS WITHOUT NESTED CV",
        "",
        "This script leaves all previous analyses untouched.",
        "It focuses on two fast sensitivity questions:",
        "1. Does performance drop when CDR, ADAS, and FAQ feature families are removed?",
        "2. Does the 24/36/60-month horizon pattern remain when the same 60-month ascertainable subjects are used?",
        "",
        "Unlike the nested robustness pipeline, this version does not perform inner model selection.",
        "It uses fixed model configurations and training-fold-only preprocessing/feature selection.",
        "",
        "Semantic sensitivity summary:",
        semantic_summary.to_string(index=False) if not semantic_summary.empty else "Skipped or unavailable.",
        "",
        "Common-cohort summary:",
        common_summary.to_string(index=False) if not common_summary.empty else "Skipped or unavailable.",
    ]
    (SUMMARY_DIR / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast no-nested robustness sensitivity analysis.")
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-common", action="store_true")
    parser.add_argument("--semantic-model", default=DEFAULT_SEMANTIC_MODEL)
    parser.add_argument("--semantic-top-k", type=int, default=DEFAULT_SEMANTIC_TOP_K)
    parser.add_argument("--common-model", default=DEFAULT_COMMON_MODEL)
    parser.add_argument("--common-top-k", type=int, default=DEFAULT_COMMON_TOP_K)
    parser.add_argument("--max-outer-folds", type=int, default=None, help="Debug only. Omit for full run.")
    args = parser.parse_args()

    ensure_dirs()
    x, y, ptid, full_df, target = load_modeling_data()

    log("=" * 72)
    log("FAST ROBUSTNESS ANALYSIS WITHOUT NESTED CV")
    log("=" * 72)
    log(f"Subjects: {len(y)}")
    log(f"Converters: {int((y == 1).sum())}")
    log(f"Stable MCI: {int((y == 0).sum())}")
    log(f"Candidate features: {x.shape[1]}")
    log(f"Output directory: {OUT_DIR}")

    semantic_summary = pd.DataFrame()
    if not args.skip_semantic:
        semantic_summary = run_semantic_sensitivity(
            x,
            y,
            ptid,
            args.semantic_model,
            args.semantic_top_k,
            args.max_outer_folds,
        )

    common_summary = pd.DataFrame()
    if not args.skip_common:
        common_summary = run_common_cohort(
            full_df,
            target,
            args.common_model,
            args.common_top_k,
            args.max_outer_folds,
        )

    write_manifest(args, x, y, target)
    write_readme(semantic_summary, common_summary)
    log("\nFinished fast robustness analysis.")
    log(f"Saved outputs to: {OUT_DIR}")


if __name__ == "__main__":
    main()
