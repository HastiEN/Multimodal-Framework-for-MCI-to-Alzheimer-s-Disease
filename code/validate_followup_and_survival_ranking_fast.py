"""Run the reduced two-repeat follow-up and Cox-ranking comparison."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

import argparse
import json
import platform
import sys

import numpy as np
import pandas as pd

import validate_followup_and_survival_ranking as base


FAST_OUT_DIR = base.PROCESSED_DIR / "robustness_min_followup_survival_coxrank_FAST_1to2h"
FAST_ANYFUTURE_DIR = FAST_OUT_DIR / "anyfuture_min_followup"
FAST_SURVIVAL_DIR = FAST_OUT_DIR / "survival_aware_feature_selection"
FAST_SURVIVAL_RANKING_DIR = FAST_SURVIVAL_DIR / "survival_rankings"
FAST_SUMMARY_DIR = FAST_OUT_DIR / "summary"

FAST_SEEDS = [42, 43]
FAST_TOP_K = [50, 100]
FAST_ANYFUTURE_MODELS = ["RandomForest", "CatBoost"]


def log(message: str = "") -> None:
    print(message, flush=True)


def configure_fast_globals(seeds: list[int], top_k_values: list[int]) -> None:
    """Redirect outputs and reduce repeated-CV work without editing the original module."""
    base.OUT_DIR = FAST_OUT_DIR
    base.ANYFUTURE_DIR = FAST_ANYFUTURE_DIR
    base.SURVIVAL_DIR = FAST_SURVIVAL_DIR
    base.SURVIVAL_RANKING_DIR = FAST_SURVIVAL_RANKING_DIR
    base.SUMMARY_DIR = FAST_SUMMARY_DIR
    base.OUTER_SEEDS = seeds
    base.TOP_K_VALUES = top_k_values
    for path in [FAST_ANYFUTURE_DIR, FAST_SURVIVAL_DIR, FAST_SURVIVAL_RANKING_DIR, FAST_SUMMARY_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_fast_manifest(args: argparse.Namespace, any_summary: pd.DataFrame, survival_summary: pd.DataFrame) -> None:
    manifest = {
        "analysis_name": "FAST_anyfuture_min24_and_survival_coxrank_sensitivity",
        "purpose": "Time-limited 1-2 hour robustness check; previous outputs untouched.",
        "important_note": (
            "This is a rapid sensitivity analysis using 2 repeats x 5 folds = 10 validation folds. "
            "It is scientifically useful for checking direction and robustness, but it should not replace "
            "the full 50-fold primary validation."
        ),
        "input_dataset": str(base.DATA_PATH),
        "cohort_file": str(base.COHORT_PATH),
        "output_dir": str(FAST_OUT_DIR),
        "minimum_followup_months_for_anyfuture_nonconverters": base.MIN_FOLLOWUP_MONTHS,
        "seeds": args.seeds,
        "folds_per_seed": base.OUTER_FOLDS,
        "total_validation_folds": len(args.seeds) * base.OUTER_FOLDS,
        "top_k_values": args.top_k,
        "anyfuture_models": args.anyfuture_models,
        "survival_models": ["CoxPH", "XGBoostSurvival"],
        "survival_feature_selection": "fold-specific univariate Cox ranking by absolute Wald z statistic",
        "hyperparameter_policy": "fixed pre-specified parameters; no Optuna and no nested CV",
        "skip_anyfuture": args.skip_anyfuture,
        "skip_survival": args.skip_survival,
        "max_outer_folds_debug": args.max_outer_folds,
        "package_versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    (FAST_SUMMARY_DIR / "fast_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    lines = [
        "FAST 1-2 HOUR ROBUSTNESS ANALYSIS",
        "",
        f"This run uses {len(args.seeds)} repeats x {base.OUTER_FOLDS} folds = "
        f"{len(args.seeds) * base.OUTER_FOLDS} validation folds.",
        "It is intended as a rapid sensitivity analysis, not a replacement for the full 50-fold validation.",
        "",
        "Any-Future min24 summary:",
        any_summary.to_string(index=False) if not any_summary.empty else "Skipped.",
        "",
        "Survival Cox-rank summary:",
        survival_summary.head(12).to_string(index=False) if not survival_summary.empty else "Skipped.",
    ]
    (FAST_SUMMARY_DIR / "README_FAST.txt").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast 1-2 hour robustness sensitivity run.")
    parser.add_argument("--skip-anyfuture", action="store_true")
    parser.add_argument("--skip-survival", action="store_true")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=FAST_SEEDS,
        help="Repeat seeds. Default: 42 43, giving 10 validation folds.",
    )
    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=FAST_TOP_K,
        choices=[20, 50, 100],
        help="Top-k feature set sizes. Default: 50 100.",
    )
    parser.add_argument(
        "--anyfuture-models",
        nargs="+",
        default=FAST_ANYFUTURE_MODELS,
        choices=list(base.CLASSIFICATION_MODELS),
        help="Models for the corrected Any-Future branch. Default: RandomForest CatBoost.",
    )
    parser.add_argument(
        "--max-outer-folds",
        type=int,
        default=None,
        help="Debug option. Limit folds per branch. Omit for real fast analysis.",
    )
    args = parser.parse_args()

    configure_fast_globals(args.seeds, args.top_k)
    feature_df, target = base.load_feature_dataset()

    log("=" * 72)
    log("FAST 1-2 HOUR ROBUSTNESS ANALYSIS")
    log("=" * 72)
    log(f"Output directory: {FAST_OUT_DIR}")
    log(f"Seeds: {args.seeds}")
    log(f"Validation folds: {len(args.seeds) * base.OUTER_FOLDS}")
    log(f"Top-k values: {args.top_k}")
    log(f"Any-Future models: {args.anyfuture_models}")
    log("No nested CV. No Optuna. Previous outputs are not modified.")

    any_cohort_summary = pd.DataFrame()
    any_model_summary = pd.DataFrame()
    if not args.skip_anyfuture:
        anyfuture_modeling, _, any_cohort_summary = base.build_anyfuture_min_followup_cohort(feature_df, target)
        log("\nCorrected Any-Future cohort:")
        log(any_cohort_summary.to_string(index=False))
        any_model_summary = base.run_anyfuture_models(
            anyfuture_modeling,
            args.anyfuture_models,
            max_outer_folds=args.max_outer_folds,
        )
        log("\nFast Any-Future min24 model summary:")
        log(any_model_summary.head(10).to_string(index=False))

    survival_summary = pd.DataFrame()
    if not args.skip_survival:
        survival_df = base.build_survival_dataset_no_write(feature_df, target)
        log(
            f"\nSurvival cohort: N={len(survival_df)}, events={int(survival_df['event'].sum())}, "
            f"censored={int((survival_df['event'] == 0).sum())}"
        )
        survival_summary = base.run_survival_coxrank(survival_df, max_outer_folds=args.max_outer_folds)
        log("\nFast survival Cox-rank summary:")
        log(survival_summary.head(10).to_string(index=False))

    write_fast_manifest(args, any_cohort_summary, survival_summary)
    if not any_model_summary.empty:
        any_model_summary.to_csv(FAST_ANYFUTURE_DIR / "anyfuture_min24_FAST_model_summary.csv", index=False)

    log("\nFinished fast robustness analysis.")
    log(f"Saved outputs to: {FAST_OUT_DIR}")


if __name__ == "__main__":
    main()
