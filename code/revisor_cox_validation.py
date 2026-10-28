"""Task 4: corrected-ties Cox on the real survival cohort, same folds as the primary analysis, compared with the historical Cox."""
import os, sys, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ICBME_RUN_DIR", str(ROOT / "runs" / "strict_v1"))
sys.path.insert(0, str(ROOT / "code"))
import numpy as np, pandas as pd
from scipy import stats
warnings.filterwarnings("ignore")
import model_survival as MS
import cox_correct as cc
from sklearn.model_selection import RepeatedStratifiedKFold
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv

OUT = ROOT / "revisor_analysis" / "4_validated_cox"; OUT.mkdir(parents=True, exist_ok=True)
df = MS.build_survival_dataset()
time, event = df.time_to_event_months.to_numpy(float), df.event.to_numpy(int)
print("survival cohort:", len(df), "events:", int(event.sum()), "tied event times:", int(event.sum() - len(np.unique(time[event == 1]))), flush=True)
sets = {}
for name, path in MS.TOP_FILES.items():
    f = [c for c in MS.load_feature_set(path) if c in df.columns and not MS.is_leakage_feature(c)]
    sets[name] = f
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=MS.RANDOM_STATE)
rows = []
for name, feats in sets.items():
    x = df[feats]
    for fold, (tr, te) in enumerate(cv.split(x, event), start=1):
        pre = MS.make_preprocessor(x.iloc[tr], scale=True)
        xt = pre.fit_transform(x.iloc[tr]); xe = pre.transform(x.iloc[te])
        b = cc.fit_fast(xt, time[tr], event[tr], alpha=0.1, maxiter=500)
        r = xe @ b
        aucs = {f"auc_{h}m": MS.time_dependent_auc(time[te], event[te], r, h) for h in (24, 36, 60)}
        rows.append(dict(feature_set=name, fold=fold, c_index=MS.harrell_c_index(time[te], event[te], r), **aucs,
                         mean_td_auc_24_36_60=float(np.mean(list(aucs.values())))))
    print(name, "done", flush=True)
new = pd.DataFrame(rows); new.to_csv(OUT / "cox_corrected_folds.csv", index=False)
old = pd.read_csv(ROOT / "audit/verified_tables/primary_survival_folds.csv"); old = old[old.model == "CoxPH"]
xg = pd.read_csv(ROOT / "audit/verified_tables/primary_survival_folds.csv"); xg = xg[xg.model == "XGBoostSurvival"]
metrics = ["c_index", "auc_24m", "auc_36m", "auc_60m", "mean_td_auc_24_36_60"]


def corr_t(d, ratio=0.25):
    n = len(d); se = np.sqrt((1 / n + ratio) * d.var(ddof=1)); t = stats.t.ppf(.975, n - 1); m = d.mean()
    return m, m - t * se, m + t * se, 2 * (1 - stats.t.cdf(abs(m / se), n - 1)) if se > 0 else np.nan


summ, comp = [], []
for name in sets:
    n_ = new[new.feature_set == name].sort_values("fold"); o_ = old[old.feature_set == name].sort_values("fold"); x_ = xg[xg.feature_set == name].sort_values("fold")
    assert len(n_) == len(o_) == 50
    for m in metrics:
        summ.append(dict(feature_set=name, metric=m, corrected_mean=n_[m].mean(), corrected_sd=n_[m].std(), historical_mean=o_[m].mean(), historical_sd=o_[m].std(),
                         fold_correlation=np.corrcoef(n_[m], o_[m].values)[0, 1]))
        for lab, a, b_ in (("corrected Cox - historical Cox", n_[m].values, o_[m].values), ("XGBoost Survival - corrected Cox", x_[m].values, n_[m].values)):
            mean, lo, hi, p = corr_t(a - b_)
            comp.append(dict(feature_set=name, metric=m, contrast=lab, mean_diff=mean, ci95_lo=lo, ci95_hi=hi, p_corrected_t=p, share_folds_positive=float(((a - b_) > 0).mean())))
pd.DataFrame(summ).to_csv(OUT / "cox_corrected_vs_historical_summary.csv", index=False)
pd.DataFrame(comp).to_csv(OUT / "cox_corrected_contrasts.csv", index=False)
# real-data coefficient check: standardised Top-20, unpenalised, full cohort: custom vs scikit-survival (Breslow)
x = df[sets["Top20"]]; pre = MS.make_preprocessor(x, scale=True); xt = pre.fit_transform(x)
b = cc.fit_fast(xt, time, event, alpha=0.0)
ref = CoxPHSurvivalAnalysis(ties="breslow", alpha=0.0, tol=1e-12, n_iter=500).fit(xt, Surv.from_arrays(event.astype(bool), time))
chk = pd.DataFrame(dict(feature=MS.get_feature_names(pre, sets["Top20"]), custom=b, sksurv_breslow=ref.coef_)); chk["abs_diff"] = (chk.custom - chk.sksurv_breslow).abs()
chk.to_csv(OUT / "cox_coefficient_check_real_data.csv", index=False)
print("real-data max |coef diff| vs scikit-survival:", chk.abs_diff.max())
print(pd.DataFrame(summ).round(4).to_string()); print(pd.DataFrame(comp).round(4).to_string())