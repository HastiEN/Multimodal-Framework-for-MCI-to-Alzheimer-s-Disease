"""Checkpoint and resume the reduced follow-up and Cox-ranking comparison."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import validate_followup_and_survival_ranking as base


OUT_DIR = base.PROCESSED_DIR / "robustness_min_followup_survival_coxrank_FAST_RESUME_1to2h"
ANY_DIR = OUT_DIR / "anyfuture_min_followup"
SURV_DIR = OUT_DIR / "survival_aware_feature_selection"
RANK_DIR = SURV_DIR / "survival_rankings"
SUMMARY_DIR = OUT_DIR / "summary"

DEFAULT_SEEDS = [42, 43]
DEFAULT_TOP_K = [50, 100]
DEFAULT_ANY_MODELS = ["RandomForest", "CatBoost"]


def log(message: str = "") -> None:
    print(message, flush=True)


def configure_outputs() -> None:
    base.OUT_DIR = OUT_DIR
    base.ANYFUTURE_DIR = ANY_DIR
    base.SURVIVAL_DIR = SURV_DIR
    base.SURVIVAL_RANKING_DIR = RANK_DIR
    base.SUMMARY_DIR = SUMMARY_DIR
    for path in [OUT_DIR, ANY_DIR, SURV_DIR, RANK_DIR, SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def append_df(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def append_row(path: Path, row: dict) -> None:
    append_df(path, pd.DataFrame([row]))


def completed_keys(path: Path, keys: list[str]) -> set[tuple]:
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    if df.empty or not set(keys).issubset(df.columns):
        return set()
    return set(map(tuple, df[keys].drop_duplicates().to_numpy().tolist()))


def summarize_classification(path: Path, out_path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    folds = pd.read_csv(path)
    if folds.empty:
        return pd.DataFrame()
    metric_cols = [
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
    summary = folds.groupby(["model", "top_k", "n_selected_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["roc_auc_mean", "balanced_accuracy_mean"], ascending=False)
    summary.to_csv(out_path, index=False)
    return summary


def summarize_survival(path: Path, out_path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    folds = pd.read_csv(path)
    if folds.empty:
        return pd.DataFrame()
    metric_cols = [
        "c_index",
        "integrated_brier_score_ipcw",
        "mean_time_dependent_auc",
        *[f"auc_{h}m" for h in base.TIME_HORIZONS],
        *[f"ipcw_brier_{h}m" for h in base.TIME_HORIZONS],
    ]
    summary = folds.groupby(["model", "top_k", "n_selected_features"])[metric_cols].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in summary.columns]
    summary = summary.sort_values(["c_index_mean", "mean_time_dependent_auc_mean"], ascending=False)
    summary.to_csv(out_path, index=False)
    return summary


def write_anyfuture_cohort(feature_df: pd.DataFrame, target: str) -> pd.DataFrame:
    modeling, _, summary = base.build_anyfuture_min_followup_cohort(feature_df, target)
    log("\nCorrected Any-Future cohort:")
    log(summary.to_string(index=False))
    return modeling


def run_anyfuture_resume(
    modeling: pd.DataFrame,
    seeds: list[int],
    top_k_values: list[int],
    models: list[str],
    max_folds: int | None,
) -> pd.DataFrame:
    fold_path = ANY_DIR / "anyfuture_min24_fold_results.csv"
    pred_path = ANY_DIR / "anyfuture_min24_predictions.csv"
    selected_path = ANY_DIR / "anyfuture_min24_selected_features.csv"
    split_path = ANY_DIR / "anyfuture_min24_split_manifest.csv"
    summary_path = ANY_DIR / "anyfuture_min24_model_summary.csv"

    y = modeling["target"].astype(int).to_numpy()
    ptid = modeling["PTID"].astype(str).reset_index(drop=True)
    x = modeling[
        [col for col in modeling.columns if col not in {"PTID", "target"} and not base.is_leakage_feature(col)]
    ].copy()
    x = base.coerce_numeric_like(x).reset_index(drop=True)

    done = completed_keys(fold_path, ["model", "top_k", "seed", "fold"])
    fold_counter = 0
    for top_k in top_k_values:
        for model_name in models:
            for seed in seeds:
                cv = StratifiedKFold(n_splits=base.OUTER_FOLDS, shuffle=True, random_state=seed)
                for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                    fold_counter += 1
                    if max_folds is not None and fold_counter > max_folds:
                        return summarize_classification(fold_path, summary_path)
                    key = (model_name, top_k, seed, fold)
                    if key in done:
                        log(f"Skip completed Any-Future | {model_name} Top-{top_k} seed {seed} fold {fold}")
                        continue
                    log(f"Run Any-Future | {model_name} Top-{top_k} seed {seed} fold {fold}")
                    assert set(ptid.iloc[train_idx]).isdisjoint(set(ptid.iloc[test_idx]))

                    fit_seed = seed * 100 + fold
                    pipe = base.classification_pipeline(model_name, top_k, fit_seed)
                    pipe.fit(x.iloc[train_idx], y[train_idx])
                    y_score = pipe.predict_proba(x.iloc[test_idx])[:, 1]
                    metrics = base.compute_classification_metrics(y[test_idx], y_score)
                    selected = pipe.named_steps["feature_selector"].selected_features_

                    base_row = {
                        "analysis": "anyfuture_min24_fast_resume",
                        "model": model_name,
                        "top_k": top_k,
                        "seed": seed,
                        "fold": fold,
                        "n_selected_features": len(selected),
                    }
                    append_row(fold_path, {**base_row, **metrics})
                    append_df(
                        pred_path,
                        pd.DataFrame(
                            [
                                {
                                    **base_row,
                                    "PTID": sid,
                                    "y_true": int(yt),
                                    "y_probability": float(ys),
                                }
                                for sid, yt, ys in zip(ptid.iloc[test_idx], y[test_idx], y_score)
                            ]
                        ),
                    )
                    append_df(
                        selected_path,
                        pd.DataFrame(
                            [
                                {**base_row, "rank": rank, "feature": feature}
                                for rank, feature in enumerate(selected, start=1)
                            ]
                        ),
                    )
                    append_df(
                        split_path,
                        pd.DataFrame(
                            [{**base_row, "PTID": sid, "split": "train"} for sid in ptid.iloc[train_idx]]
                            + [{**base_row, "PTID": sid, "split": "test"} for sid in ptid.iloc[test_idx]]
                        ),
                    )
                    summarize_classification(fold_path, summary_path)
    return summarize_classification(fold_path, summary_path)


def survival_feature_stability(selected_path: Path, out_path: Path, top_k_values: list[int]) -> pd.DataFrame:
    if not selected_path.exists():
        return pd.DataFrame()
    selected = pd.read_csv(selected_path)
    if selected.empty:
        return pd.DataFrame()
    total_folds = selected[["seed", "fold"]].drop_duplicates().shape[0]
    rows = []
    for feature, group in selected.groupby("feature"):
        row = {"feature": feature, "mean_cox_rank": float(group["rank"].mean())}
        for top_k in top_k_values:
            row[f"top{top_k}_frequency"] = group.loc[group["top_k"] == top_k, ["seed", "fold"]].drop_duplicates().shape[0] / total_folds
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = [f"top{k}_frequency" for k in top_k_values] + ["mean_cox_rank"]
    out = out.sort_values(sort_cols, ascending=[False] * len(top_k_values) + [True])
    out.to_csv(out_path, index=False)
    return out


def compare_old_new_survival(new_summary: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    old_path = base.PROCESSED_DIR / "survival_analysis" / "results" / "survival_model_comparison.csv"
    if new_summary.empty or not old_path.exists():
        return pd.DataFrame()
    old = pd.read_csv(old_path)
    old["top_k"] = old["feature_set"].astype(str).str.extract(r"(\d+)").astype(int)
    rows = []
    for _, new_row in new_summary.iterrows():
        match = old[(old["model"] == new_row["model"]) & (old["top_k"] == int(new_row["top_k"]))]
        if match.empty:
            continue
        old_row = match.iloc[0]
        rows.append(
            {
                "model": new_row["model"],
                "top_k": int(new_row["top_k"]),
                "old_c_index": old_row["c_index_mean"],
                "new_c_index": new_row["c_index_mean"],
                "delta_c_index": new_row["c_index_mean"] - old_row["c_index_mean"],
                "old_mean_time_dependent_auc": old_row["mean_time_dependent_auc_mean"],
                "new_mean_time_dependent_auc": new_row["mean_time_dependent_auc_mean"],
                "delta_mean_time_dependent_auc": new_row["mean_time_dependent_auc_mean"] - old_row["mean_time_dependent_auc_mean"],
                "old_ipcw_ibs": old_row["integrated_brier_score_ipcw_mean"],
                "new_ipcw_ibs": new_row["integrated_brier_score_ipcw_mean"],
                "delta_ipcw_ibs": new_row["integrated_brier_score_ipcw_mean"] - old_row["integrated_brier_score_ipcw_mean"],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    return out


def run_survival_resume(
    survival_df: pd.DataFrame,
    seeds: list[int],
    top_k_values: list[int],
    max_folds: int | None,
) -> pd.DataFrame:
    fold_path = SURV_DIR / "survival_coxrank_fold_results.csv"
    pred_path = SURV_DIR / "survival_coxrank_predictions.csv"
    selected_path = SURV_DIR / "survival_coxrank_selected_features.csv"
    split_path = SURV_DIR / "survival_coxrank_split_manifest.csv"
    audit_path = SURV_DIR / "survival_coxrank_ranking_audit.csv"
    summary_path = SURV_DIR / "survival_coxrank_model_summary.csv"
    stability_path = SURV_DIR / "survival_feature_stability.csv"
    comparison_path = SURV_DIR / "survival_feature_selection_comparison.csv"

    event = survival_df["event"].astype(int).to_numpy()
    duration = survival_df["time_to_event_months"].astype(float).to_numpy()
    ptid = survival_df["PTID"].astype(str).reset_index(drop=True)
    x = survival_df[
        [col for col in survival_df.columns if col not in {"PTID", "time_to_event_months", "event"} and not base.is_leakage_feature(col)]
    ].copy()
    x = base.coerce_numeric_like(x).reset_index(drop=True)

    done = completed_keys(fold_path, ["model", "top_k", "seed", "fold"])
    written_splits = completed_keys(split_path, ["seed", "fold"])
    fold_counter = 0
    for seed in seeds:
        cv = StratifiedKFold(n_splits=base.OUTER_FOLDS, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(x, event), start=1):
            fold_counter += 1
            if max_folds is not None and fold_counter > max_folds:
                summary = summarize_survival(fold_path, summary_path)
                survival_feature_stability(selected_path, stability_path, top_k_values)
                compare_old_new_survival(summary, comparison_path)
                return summary

            ranking_path = RANK_DIR / f"seed{seed}_fold{fold}_cox_ranking.csv"
            if ranking_path.exists():
                ranking = pd.read_csv(ranking_path)
                # Rebuild processed matrices because they are not stored, but avoid reranking.
                x_train, x_test = base.preprocess_survival_fold(x.iloc[train_idx], x.iloc[test_idx])
                log(f"Reuse Cox ranking | seed {seed} fold {fold}")
            else:
                log(f"Rank survival features | seed {seed} fold {fold}")
                x_train, x_test = base.preprocess_survival_fold(x.iloc[train_idx], x.iloc[test_idx])
                ranking = base.rank_features_univariate_cox(x_train, duration[train_idx], event[train_idx])
                ranking.to_csv(ranking_path, index=False)
                append_row(
                    audit_path,
                    {
                        "seed": seed,
                        "fold": fold,
                        "n_processed_features": x_train.shape[1],
                        "n_valid_cox_ranked_features": int((ranking["status"] == "ok").sum()),
                        "n_failed_or_constant_features": int((ranking["status"] != "ok").sum()),
                    },
                )

            valid = ranking[ranking["status"] == "ok"]
            if len(valid) < max(top_k_values):
                raise RuntimeError(f"Seed {seed} fold {fold} has only {len(valid)} valid Cox-ranked features.")
            t_train, t_test = duration[train_idx], duration[test_idx]
            e_train, e_test = event[train_idx], event[test_idx]
            assert set(ptid.iloc[train_idx]).isdisjoint(set(ptid.iloc[test_idx]))

            if (seed, fold) not in written_splits:
                append_df(
                    split_path,
                    pd.DataFrame(
                        [{"seed": seed, "fold": fold, "PTID": sid, "split": "train"} for sid in ptid.iloc[train_idx]]
                        + [{"seed": seed, "fold": fold, "PTID": sid, "split": "test"} for sid in ptid.iloc[test_idx]]
                    ),
                )
                written_splits.add((seed, fold))

            for top_k in top_k_values:
                selected = valid.head(top_k)["feature"].tolist()
                xtr = x_train[selected]
                xte = x_test[selected]
                for model_name in ["CoxPH", "XGBoostSurvival"]:
                    key = (model_name, top_k, seed, fold)
                    if key in done:
                        log(f"Skip completed survival | {model_name} Top-{top_k} seed {seed} fold {fold}")
                        continue
                    log(f"Run survival | {model_name} Top-{top_k} seed {seed} fold {fold}")
                    if model_name == "CoxPH":
                        beta = base.fit_cox_processed(xtr, t_train, e_train)
                        train_risk = xtr.to_numpy(float) @ beta
                        risk = xte.to_numpy(float) @ beta
                    else:
                        model = base.fit_xgb_survival_processed(xtr, t_train, e_train, seed=seed * 100 + fold)
                        train_risk = model.predict(xtr)
                        risk = model.predict(xte)
                    metrics = base.evaluate_survival_fold(model_name, risk, train_risk, t_train, e_train, t_test, e_test)
                    row = {
                        "seed": seed,
                        "fold": fold,
                        "model": model_name,
                        "top_k": top_k,
                        "n_selected_features": len(selected),
                        **metrics,
                    }
                    append_row(fold_path, row)
                    append_df(
                        pred_path,
                        pd.DataFrame(
                            [
                                {
                                    "seed": seed,
                                    "fold": fold,
                                    "model": model_name,
                                    "top_k": top_k,
                                    "PTID": sid,
                                    "time_to_event_months": float(tt),
                                    "event": int(ee),
                                    "risk_score": float(rr),
                                }
                                for sid, tt, ee, rr in zip(ptid.iloc[test_idx], t_test, e_test, risk)
                            ]
                        ),
                    )
                existing_selected = completed_keys(selected_path, ["top_k", "seed", "fold"])
                if (top_k, seed, fold) not in existing_selected:
                    append_df(
                        selected_path,
                        pd.DataFrame(
                            [
                                {"seed": seed, "fold": fold, "top_k": top_k, "rank": rank, "feature": feature}
                                for rank, feature in enumerate(selected, start=1)
                            ]
                        ),
                    )
            summary = summarize_survival(fold_path, summary_path)
            survival_feature_stability(selected_path, stability_path, top_k_values)
            compare_old_new_survival(summary, comparison_path)
    summary = summarize_survival(fold_path, summary_path)
    survival_feature_stability(selected_path, stability_path, top_k_values)
    compare_old_new_survival(summary, comparison_path)
    return summary


def write_manifest(args: argparse.Namespace) -> None:
    manifest = {
        "analysis_name": "FAST_RESUME_anyfuture_min24_and_survival_coxrank",
        "purpose": "Checkpointed fast robustness analysis. Reruns skip completed folds.",
        "output_dir": str(OUT_DIR),
        "input_dataset": str(base.DATA_PATH),
        "cohort_file": str(base.COHORT_PATH),
        "seeds": args.seeds,
        "folds_per_seed": base.OUTER_FOLDS,
        "total_planned_validation_folds": len(args.seeds) * base.OUTER_FOLDS,
        "top_k_values": args.top_k,
        "anyfuture_models": args.anyfuture_models,
        "survival_models": ["CoxPH", "XGBoostSurvival"],
        "minimum_followup_months_for_anyfuture_nonconverters": base.MIN_FOLLOWUP_MONTHS,
        "resume_policy": "Each fold/config writes immediately. Existing fold/config rows are skipped on rerun.",
        "nested_cv": False,
        "optuna": False,
        "package_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (SUMMARY_DIR / "resume_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def write_readme(any_summary: pd.DataFrame, survival_summary: pd.DataFrame) -> None:
    lines = [
        "FAST RESUME ROBUSTNESS ANALYSIS",
        "",
        "This version saves every fold/configuration immediately.",
        "If the run crashes, rerun the same command and completed folds will be skipped.",
        "",
        "Any-Future current summary:",
        any_summary.to_string(index=False) if not any_summary.empty else "Skipped or no completed folds.",
        "",
        "Survival current summary:",
        survival_summary.to_string(index=False) if not survival_summary.empty else "Skipped or no completed folds.",
    ]
    (SUMMARY_DIR / "README_RESUME.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Checkpointed fast robustness sensitivity analysis.")
    parser.add_argument("--skip-anyfuture", action="store_true")
    parser.add_argument("--skip-survival", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--top-k", nargs="+", type=int, choices=[20, 50, 100], default=DEFAULT_TOP_K)
    parser.add_argument(
        "--anyfuture-models",
        nargs="+",
        choices=list(base.CLASSIFICATION_MODELS),
        default=DEFAULT_ANY_MODELS,
    )
    parser.add_argument("--max-folds", type=int, default=None, help="Debug limit across folds. Omit for planned run.")
    args = parser.parse_args()

    configure_outputs()
    base.TOP_K_VALUES = args.top_k

    feature_df, target = base.load_feature_dataset()
    log("=" * 72)
    log("FAST RESUME ROBUSTNESS ANALYSIS")
    log("=" * 72)
    log(f"Outputs: {OUT_DIR}")
    log(f"Seeds: {args.seeds}")
    log(f"Top-k: {args.top_k}")
    log(f"Any-Future models: {args.anyfuture_models}")
    log("Checkpointing is ON. Reruns skip completed fold/configuration rows.")

    any_summary = pd.DataFrame()
    if not args.skip_anyfuture:
        modeling = write_anyfuture_cohort(feature_df, target)
        any_summary = run_anyfuture_resume(modeling, args.seeds, args.top_k, args.anyfuture_models, args.max_folds)
        log("\nAny-Future current summary:")
        log(any_summary.head(12).to_string(index=False) if not any_summary.empty else "No completed Any-Future folds.")

    survival_summary = pd.DataFrame()
    if not args.skip_survival:
        survival_df = base.build_survival_dataset_no_write(feature_df, target)
        log(
            f"\nSurvival cohort: N={len(survival_df)}, events={int(survival_df['event'].sum())}, "
            f"censored={int((survival_df['event'] == 0).sum())}"
        )
        survival_summary = run_survival_resume(survival_df, args.seeds, args.top_k, args.max_folds)
        log("\nSurvival current summary:")
        log(survival_summary.head(12).to_string(index=False) if not survival_summary.empty else "No completed survival folds.")

    write_manifest(args)
    write_readme(any_summary, survival_summary)
    log(f"\nDone. Resume outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
