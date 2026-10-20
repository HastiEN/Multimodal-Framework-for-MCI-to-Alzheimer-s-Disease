"""Reproduce legacy publication SHAP figures and their value-color legend."""
from __future__ import annotations
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'code'))

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from interpret_tuned_xgboost import (
    DATA_PATH,
    OUT_DIR,
    compute_shap,
    load_best_params,
    train_final_model,
)


PUBLICATION_DIR = OUT_DIR / "publication_figures"
PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_PATH = PUBLICATION_DIR / "figure_shap_summary_ieee.png"
SUMMARY_LEGEND_PATH = PUBLICATION_DIR / "figure_shap_summary_ieee_feature_value_legend.png"
BAR_PATH = PUBLICATION_DIR / "figure_shap_bar_ieee.png"

FONT_FAMILY = "Times New Roman"
FONT_SIZE = 8
MAX_DISPLAY = 20


FEATURE_LABELS = {
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
    "NEUROBAT_AVTOT2": "Auditory verbal learning trial 2",
    "NEUROBAT_AVTOT3": "Auditory verbal learning trial 3",
    "NEUROBAT_AVTOT4": "Auditory verbal learning trial 4",
}


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


def readable_label(feature: str) -> str:
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
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


def relabel_features(x_transformed_df: pd.DataFrame) -> pd.DataFrame:
    renamed = [readable_label(col) for col in x_transformed_df.columns]
    out = x_transformed_df.copy()
    out.columns = renamed
    return out


def tune_current_figure(path: Path, xlabel: str) -> None:
    fig = plt.gcf()
    fig.set_size_inches(7.4, 5.2)
    main_ax = fig.axes[0] if fig.axes else None
    colorbar_ax = fig.axes[-1] if len(fig.axes) > 1 else None
    for ax in fig.axes:
        if ax is main_ax:
            ax.set_xlabel(xlabel, fontsize=FONT_SIZE, fontname=FONT_FAMILY)
        elif ax is colorbar_ax:
            ax.set_xlabel("", fontsize=FONT_SIZE, fontname=FONT_FAMILY)
            ax.set_title("")
            ax.set_ylabel(
                "Feature value",
                fontsize=FONT_SIZE,
                fontname=FONT_FAMILY,
                rotation=90,
                labelpad=14,
            )
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            y_min, y_max = ax.get_ylim()
            ax.set_yticks([y_min, y_max])
            ax.set_yticklabels(["Low", "High"], fontname=FONT_FAMILY, fontsize=FONT_SIZE)
            ax.tick_params(axis="y", length=0, pad=4)
        else:
            ax.set_xlabel("", fontsize=FONT_SIZE, fontname=FONT_FAMILY)
            ax.set_ylabel(ax.get_ylabel(), fontsize=FONT_SIZE, fontname=FONT_FAMILY)
        ax.tick_params(axis="both", labelsize=FONT_SIZE)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(FONT_FAMILY)
            label.set_fontsize(FONT_SIZE)
    plt.subplots_adjust(left=0.34, right=0.90, bottom=0.12, top=0.98)
    plt.savefig(path, dpi=600, bbox_inches="tight")
    if path == SUMMARY_PATH:
        plt.savefig(SUMMARY_LEGEND_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)


def make_summary_plot(shap_values: np.ndarray, x_plot: pd.DataFrame) -> None:
    set_ieee_style()
    shap.summary_plot(
        shap_values,
        x_plot,
        show=False,
        max_display=MAX_DISPLAY,
        plot_size=(7.0, 4.8),
        color_bar_label="Feature value",
    )
    ax = plt.gca()
    ax.set_xlabel(
        "SHAP value contribution to predicted conversion risk",
        fontsize=FONT_SIZE,
        fontname=FONT_FAMILY,
    )
    tune_current_figure(SUMMARY_PATH, "SHAP value contribution to predicted conversion risk")


def make_bar_plot(shap_values: np.ndarray, x_plot: pd.DataFrame) -> None:
    set_ieee_style()
    shap.summary_plot(
        shap_values,
        x_plot,
        plot_type="bar",
        show=False,
        max_display=MAX_DISPLAY,
        plot_size=(7.0, 4.8),
    )
    ax = plt.gca()
    ax.set_xlabel(
        "Mean absolute SHAP value contribution",
        fontsize=FONT_SIZE,
        fontname=FONT_FAMILY,
    )
    tune_current_figure(BAR_PATH, "Mean absolute SHAP value contribution")


def main() -> None:
    data = pd.read_csv(DATA_PATH)
    if "conversion_label" not in data.columns:
        raise ValueError("Dataset must contain conversion_label.")

    x = data.drop(columns=["conversion_label"])
    y = data["conversion_label"].astype(int)
    params = load_best_params()
    pipe = train_final_model(x, y, params)
    shap_values, x_transformed_df, _ = compute_shap(pipe, x)
    x_plot = relabel_features(x_transformed_df)

    make_summary_plot(shap_values, x_plot)
    make_bar_plot(shap_values, x_plot)

    print("Publication-ready SHAP figures saved:")
    print(SUMMARY_PATH)
    print(SUMMARY_LEGEND_PATH)
    print(BAR_PATH)


if __name__ == "__main__":
    main()
