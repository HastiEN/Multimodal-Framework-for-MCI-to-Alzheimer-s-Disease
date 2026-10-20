"""Resolve frozen inputs and writable rerun paths for the packaged study."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def run_root() -> Path:
    root = Path(os.environ.get("ICBME_RUN_DIR", str(PACKAGE_ROOT / "runs" / "default"))).resolve()
    frozen = [(PACKAGE_ROOT / name).resolve() for name in ("data", "outputs", "final_result", "manuscript")]
    if any(root == item or item in root.parents or root in item.parents for item in frozen):
        raise ValueError("ICBME_RUN_DIR must be separate from the frozen input and manuscript folders.")
    root.mkdir(parents=True, exist_ok=True)
    return root


def seed_directory(name: str) -> Path:
    destination = run_root() / name
    marker = run_root() / ("." + name.replace("/", "_") + "_seeded")
    if not marker.exists():
        ignored = None
        if os.environ.get("ICBME_FRESH_WORKSPACE") == "1":
            ignored = shutil.ignore_patterns(
                "robustness_analysis",
                "robustness_analysis_fast_no_nested",
                "robustness_min_followup_survival_coxrank_FAST_RESUME_1to2h",
                "primary_model_shap_continuation",
            )
        shutil.copytree(PACKAGE_ROOT / name, destination, dirs_exist_ok=True, ignore=ignored)
        if name == "data/processed":
            aliases = {
                "mci_to_ad_labels_from_all_csv_scan.csv": "1-mci_to_ad_labels_from_all_csv_scan.csv",
                "strict_mci_to_ad_publication_cohort.csv": "2-strict_mci_to_ad_publication_cohort.csv",
                "final_mci_to_ad_publication_dataset.csv": "3-final_mci_to_ad_publication_dataset.csv",
            }
            for canonical, historical in aliases.items():
                if not (destination / canonical).exists() and (destination / historical).exists():
                    shutil.copy2(destination / historical, destination / canonical)
        marker.touch()
    return destination


def processed_dir() -> Path:
    return seed_directory("data/processed")


def outputs_dir() -> Path:
    return seed_directory("outputs")


def project_path(relative: str) -> Path:
    normalized = relative.replace("\\", "/")
    if normalized == "data/processed" or normalized.startswith("data/processed/"):
        return processed_dir() / normalized[len("data/processed"):].lstrip("/")
    if normalized == "outputs" or normalized.startswith("outputs/"):
        return outputs_dir() / normalized[len("outputs"):].lstrip("/")
    if normalized == "data/final" or normalized.startswith("data/final/"):
        return seed_directory("data/final") / normalized[len("data/final"):].lstrip("/")
    return PACKAGE_ROOT / normalized
