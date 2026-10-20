"""Run a named analysis in a writable workspace; bundled results stay frozen."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STAGES = {
    "scan": "scan_diagnosis_tables.py",
    "cohort": "construct_mci_cohort.py",
    "extract": "extract_baseline_predictors.py",
    "publication-ranking": "rank_publication_predictors.py",
    "any-future": "classify_any_future.py",
    "publication-classification": "classify_publication_subsets.py",
    "fixed-horizons": "classify_fixed_horizons.py",
    "survival": "model_survival.py",
    "nested-sensitivity": "validate_nested_sensitivity.py",
    "fixed-model-sensitivity": "validate_fixed_model_sensitivity.py",
    "followup-survival-sensitivity": "validate_followup_and_survival_ranking.py",
    "fast-sensitivity": "validate_followup_and_survival_ranking_fast.py",
    "resume-sensitivity": "validate_followup_and_survival_ranking_resume.py",
    "tune-shap": "tune_interpretation_xgboost.py",
    "shap-tuned": "interpret_tuned_xgboost.py",
    "shap-extended": "interpret_primary_models.py",
    "shap-consistency": "summarize_attribution_consistency.py",
    "shap-figures": "plot_publication_shap.py",
    "discrimination-figures": "plot_classification_discrimination.py",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List stages without training.")
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--run-name", default="default", help="Folder name under runs; reuse it to resume.")
    parser.add_argument("--fresh", action="store_true", help="Start a new workspace without saved sensitivity or SHAP checkpoints.")
    args, forwarded = parser.parse_known_args()
    if args.list:
        for stage, script in STAGES.items():
            print(f"{stage:30s} code/{script}")
        return 0
    if not args.stage:
        parser.error("Choose --stage or --list.")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        parser.error("--run-name must be a single folder name.")
    env = os.environ.copy()
    env["ICBME_RUN_DIR"] = str(ROOT / "runs" / args.run_name)
    if args.fresh:
        if (ROOT / "runs" / args.run_name).exists():
            parser.error("--fresh requires a new --run-name; existing workspaces are retained.")
        env["ICBME_FRESH_WORKSPACE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["MPLBACKEND"] = "Agg"
    script = ROOT / "code" / STAGES[args.stage]
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    print(f"Run workspace: {env['ICBME_RUN_DIR']}", flush=True)
    return subprocess.run([sys.executable, str(script), *forwarded], cwd=ROOT, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
