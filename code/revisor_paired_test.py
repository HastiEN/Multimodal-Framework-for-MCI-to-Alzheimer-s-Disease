"""Task 3: paired comparison of XGBoost Survival vs Cox PH from the saved per-fold results (no refitting).

Both models were evaluated on identical folds (same RepeatedStratifiedKFold(5 x 10, seed 42) split, fold k of each model uses the same test set).
Paired differences use the Nadeau-Bengio corrected resampled t (variance factor 1/50 + 1/4) and a sign-flip permutation check.
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revisor_analysis" / "3_paired_test"; OUT.mkdir(parents=True, exist_ok=True)
f = pd.read_csv(ROOT / "audit/verified_tables/primary_survival_folds.csv")
metrics = ["c_index", "mean_td_auc_24_36_60", "auc_24m", "auc_36m", "auc_60m"]
rng = np.random.default_rng(0)
rows = []
for fs_ in ("Top20", "Top50", "Top100"):
    x = f[(f.model == "XGBoostSurvival") & (f.feature_set == fs_)].sort_values("fold"); c = f[(f.model == "CoxPH") & (f.feature_set == fs_)].sort_values("fold")
    assert (x.fold.values == c.fold.values).all() and len(x) == 50
    for m in metrics:
        d = x[m].values - c[m].values; n = len(d); s2 = d.var(ddof=1)
        se = np.sqrt((1 / n + 0.25) * s2); t = d.mean() / se; p = 2 * (1 - stats.t.cdf(abs(t), n - 1)); q = stats.t.ppf(.975, n - 1)
        signs = rng.choice([-1, 1], size=(20000, n)); perm = np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-15
        rows.append(dict(feature_set=fs_, metric=m, xgb_mean=x[m].mean(), xgb_sd=x[m].std(), cox_mean=c[m].mean(), cox_sd=c[m].std(), mean_diff=d.mean(), sd_diff=d.std(ddof=1),
                         ci95_lo=d.mean() - q * se, ci95_hi=d.mean() + q * se, t_corrected=t, p_corrected_t=p, p_signflip_naive=perm.mean(),
                         share_folds_xgb_better=float((d > 0).mean()), n_folds=n))
R = pd.DataFrame(rows)
# Holm correction over the pre-specified family: C-index and mean AUC at the three list sizes (6 tests)
fam = R[R.metric.isin(["c_index", "mean_td_auc_24_36_60"])].copy().sort_values("p_corrected_t"); k = len(fam)
fam["p_holm"] = np.minimum(1, np.maximum.accumulate([(k - i) * p for i, p in enumerate(fam.p_corrected_t)]))
R = R.merge(fam[["feature_set", "metric", "p_holm"]], on=["feature_set", "metric"], how="left")
R.to_csv(OUT / "xgb_vs_cox_paired.csv", index=False)
print(R.round(4).to_string())