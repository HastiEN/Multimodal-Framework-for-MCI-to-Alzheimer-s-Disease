"""Task 1: horizon figure with a full-scale y-axis and per-algorithm uncertainty (corrected resampled-t 95% CI of the mean over 50 folds).

Error bars are 95% confidence intervals of the mean AUC with the Nadeau-Bengio variance correction (1/50 + 1/4); they are NOT fold SDs.
Fold SDs are given in the CSV. The dashed series is the nested common-cohort analysis (n = 640 at every horizon), from its saved summary.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revisor_analysis" / "1_horizon_figure"; OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8, "axes.linewidth": .8})
f = pd.read_csv(ROOT / "data/processed/prediction_horizon_analysis/horizon_cv_fold_results.csv")
f = f[f.feature_set == "Top50"]
H = [24, 36, 60]; N = {24: 907, 36: 807, 60: 640}
order = ["LogisticRegression", "RandomForest", "XGBoost", "CatBoost", "LightGBM"]
lab = {"LogisticRegression": "Logistic Reg.", "RandomForest": "Random Forest", "XGBoost": "XGBoost", "CatBoost": "CatBoost", "LightGBM": "LightGBM"}
col = {"LogisticRegression": "#1b9e77", "RandomForest": "#d95f02", "XGBoost": "#7570b3", "CatBoost": "#e7298a", "LightGBM": "#66a61e"}
mk = {"LogisticRegression": "o", "RandomForest": "s", "XGBoost": "^", "CatBoost": "D", "LightGBM": "v"}


def ci(mean, sd, n=50, ratio=0.25):
    se = np.sqrt((1 / n + ratio) * sd ** 2); q = stats.t.ppf(.975, n - 1); return mean - q * se, mean + q * se


rows = []
for m in order:
    for h in H:
        a = f[(f.model == m) & (f.horizon == h)].roc_auc; lo, hi = ci(a.mean(), a.std()); rows.append(dict(series=m, horizon=h, n_folds=len(a), auc_mean=a.mean(), fold_sd=a.std(), ci95_lo=lo, ci95_hi=hi))
nest = pd.read_csv(ROOT / "audit/verified_tables/common_nested_summary.csv")
for _, r in nest.iterrows():
    lo, hi = ci(r.roc_auc_mean, r.roc_auc_std, int(r.roc_auc_count)); rows.append(dict(series="Nested, fixed cohort (n=640)", horizon=int(r.horizon_months), n_folds=int(r.roc_auc_count), auc_mean=r.roc_auc_mean, fold_sd=r.roc_auc_std, ci95_lo=lo, ci95_hi=hi))
D = pd.DataFrame(rows); D.to_csv(OUT / "figure_values.csv", index=False)


def draw(ax, ylim, zoom=False):
    off = np.linspace(-.5, .5, 6)
    for i, m in enumerate(order + ["nest"]):
        s = D[D.series == (m if m != "nest" else "Nested, fixed cohort (n=640)")].sort_values("horizon"); x = np.arange(3) + off[i] * 0.55
        kw = dict(color="k", ls="--", marker="x", ms=4.5) if m == "nest" else dict(color=col[m], marker=mk[m], ms=4)
        ax.errorbar(x, s.auc_mean, yerr=[s.auc_mean - s.ci95_lo, s.ci95_hi - s.auc_mean], capsize=2, lw=1, elinewidth=.8, label=("Nested, fixed cohort (n=640)" if m == "nest" else lab[m]), **kw)
    ax.axhline(.5, color="grey", lw=.8, ls=":"); ax.set_ylim(*ylim); ax.set_xlim(-.6, 2.6)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{h} mo\n(n={N[h]})" for h in H]); ax.set_ylabel("ROC-AUC (mean, 95% CI)"); ax.grid(alpha=.25)
    if zoom: ax.text(.02, .97, "zoomed axis (0.80-0.95)", transform=ax.transAxes, fontsize=7, style="italic", va="top")


fig, ax = plt.subplots(figsize=(3.5, 3.0)); draw(ax, (0.5, 1.0)); ax.text(.97, .52, "chance", transform=ax.get_yaxis_transform(), ha="right", fontsize=7, color="grey")
ax.legend(fontsize=6.5, loc="lower right", framealpha=.9); fig.tight_layout(); fig.savefig(OUT / "fig_horizon_fullscale.png", dpi=300); plt.close(fig)
fig, ax = plt.subplots(figsize=(3.5, 3.0)); draw(ax, (0.80, 0.95), zoom=True); ax.legend(fontsize=5.6, loc="lower right", ncol=2, framealpha=.95, columnspacing=0.8, handlelength=1.6, borderpad=0.3, labelspacing=0.25); fig.tight_layout(); fig.savefig(OUT / "fig_horizon_zoomed.png", dpi=300); plt.close(fig)
fig, ax = plt.subplots(2, 1, figsize=(3.5, 5.4), gridspec_kw=dict(height_ratios=[1, 1.15]))
draw(ax[0], (0.5, 1.0)); ax[0].set_title("(a) full scale", fontsize=8, loc="left"); ax[0].legend(fontsize=6, loc="lower right", framealpha=.9)
draw(ax[1], (0.80, 0.95), zoom=True); ax[1].set_title("(b) zoomed", fontsize=8, loc="left")
fig.tight_layout(); fig.savefig(OUT / "fig_horizon_two_panel.png", dpi=300); plt.close(fig)
print(D.round(3).to_string())