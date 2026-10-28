"""Task 5: corrected IBS vs Kaplan-Meier null on identical folds, plus out-of-fold calibration at 24/36/60 months.

Cohort/folds as in the primary survival analysis (1,235 participants, 5 x 10 repeated stratified CV, seed 42).
Models: Cox PH (corrected Breslow ties, ridge 0.1) and XGBoost Survival (survival:cox; predict() returns the hazard ratio exp(margin), so log() is the linear predictor).
S(t|x) = exp(-H0(t) * exp(lp)) with a Breslow baseline hazard estimated on the TRAINING fold.
Brier: corrected IPCW (survival_metrics.brier_ipcw, validated against scikit-survival); IBS over 24-60 months, grid step 3 months (the study ends at 60 months).
Null: training-fold KM, same for every test participant. Skill = 1 - IBS_model / IBS_null.
Calibration: repeat-averaged out-of-fold predicted risk per participant; quintile groups; observed risk = 1 - KM within group (censoring-aware); bootstrap CIs.
"""
import os, sys, warnings, time as _time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("ICBME_RUN_DIR", str(ROOT / "runs" / "strict_v1"))
sys.path.insert(0, str(ROOT / "code"))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
import model_survival as MS
import cox_correct as cc
import survival_metrics as sm
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = ROOT / "revisor_analysis" / "5_km_null_ibs_calibration"; OUT.mkdir(parents=True, exist_ok=True)
GRID = np.arange(24.0, 60.1, 3.0); I60 = GRID <= 60
HZ = [24, 36, 60]; HI = [int(np.where(GRID == h)[0][0]) for h in HZ]
df = MS.build_survival_dataset()
T, E = df.time_to_event_months.to_numpy(float), df.event.to_numpy(int)
sets = {n: [c for c in MS.load_feature_set(p) if c in df.columns and not MS.is_leakage_feature(c)] for n, p in MS.TOP_FILES.items()}
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=MS.RANDOM_STATE)
splits = list(cv.split(df, E))
print("cohort", len(df), "events", int(E.sum()), flush=True)


def fold_job(name, fold, tr, te):
    x = df[sets[name]]
    pre = MS.make_preprocessor(x.iloc[tr], scale=True); xt = pre.fit_transform(x.iloc[tr]); xe = pre.transform(x.iloc[te])
    b = cc.fit_fast(xt, T[tr], E[tr], alpha=0.1, maxiter=500)
    ts, H0 = cc.breslow_cumhaz(T[tr], E[tr], xt @ b)
    S = {"CoxPH": cc.survival_at(ts, H0, xe @ b, GRID)}
    pipe = MS.fit_xgb_survival(x.iloc[tr], T[tr], E[tr]); htr, hte = pipe.predict(x.iloc[tr]), pipe.predict(x.iloc[te])
    assert (htr > 0).all() and (hte > 0).all(), "XGB survival:cox output expected to be a hazard ratio (>0)"
    ts, H0 = cc.breslow_cumhaz(T[tr], E[tr], np.log(htr))
    S["XGBoostSurvival"] = cc.survival_at(ts, H0, np.log(hte), GRID)
    S["KM_null"] = np.tile(sm.km_null_survival(T[tr], E[tr], GRID), (len(te), 1))
    rows, oof = [], []
    for mname, Sm in S.items():
        br = np.array([sm.brier_ipcw(T[tr], E[tr], T[te], E[te], Sm[:, k], g) for k, g in enumerate(GRID)])
        for k, g in enumerate(GRID): rows.append(dict(feature_set=name, fold=fold, model=mname, t=g, brier=br[k]))
        rows.append(dict(feature_set=name, fold=fold, model=mname, t=-60, brier=sm.integrated_brier(GRID[I60], br[I60])))
        if mname != "KM_null":
            oof.append(pd.DataFrame(dict(feature_set=name, repeat=(fold - 1) // 5, idx=te, model=mname, **{f"surv_{h}": Sm[:, i] for h, i in zip(HZ, HI)})))
    return rows, pd.concat(oof)


t0 = _time.time()
jobs = [(n, f, tr, te) for n in sets for f, (tr, te) in enumerate(splits, start=1)]
res = Parallel(n_jobs=4, verbose=5)(delayed(fold_job)(*j) for j in jobs)
B = pd.DataFrame([r for rr, _ in res for r in rr]); O = pd.concat([o for _, o in res])
print(f"folds done in {(_time.time() - t0) / 60:.1f} min", flush=True)
B.to_csv(OUT / "brier_by_time_folds.csv", index=False)
I = B[B.t < 0].assign(window=lambda d: d.t.map({-60: "IBS 24-60 mo"})).drop(columns="t").rename(columns={"brier": "ibs"})
I.to_csv(OUT / "ibs_folds.csv", index=False)


def corr_t(d, ratio=0.25):
    n = len(d); se = np.sqrt((1 / n + ratio) * d.var(ddof=1)); q = stats.t.ppf(.975, n - 1); m = d.mean()
    return m, m - q * se, m + q * se, 2 * (1 - stats.t.cdf(abs(m / se), n - 1)) if se > 0 else np.nan


summ, paired = [], []
for (fs_, win), g in I.groupby(["feature_set", "window"]):
    w = g.pivot(index="fold", columns="model", values="ibs")
    skill = 1 - w.XGBoostSurvival / w.KM_null, 1 - w.CoxPH / w.KM_null
    for m in ("KM_null", "CoxPH", "XGBoostSurvival"):
        summ.append(dict(feature_set=fs_, window=win, model=m, ibs_mean=w[m].mean(), ibs_sd=w[m].std(),
                         skill_mean=(np.nan if m == "KM_null" else (1 - w[m] / w.KM_null).mean()), skill_sd=(np.nan if m == "KM_null" else (1 - w[m] / w.KM_null).std())))
    for a, b_ in (("CoxPH", "KM_null"), ("XGBoostSurvival", "KM_null"), ("XGBoostSurvival", "CoxPH")):
        mean, lo, hi, p = corr_t((w[a] - w[b_]).values)
        paired.append(dict(feature_set=fs_, window=win, contrast=f"IBS {a} - {b_}", mean_diff=mean, ci95_lo=lo, ci95_hi=hi, p_corrected_t=p, share_folds_negative=float(((w[a] - w[b_]) < 0).mean())))
pd.DataFrame(summ).to_csv(OUT / "ibs_summary.csv", index=False); pd.DataFrame(paired).to_csv(OUT / "ibs_paired_contrasts.csv", index=False)
H = B[(B.t > 0) & B.t.isin(HZ)].groupby(["feature_set", "model", "t"]).brier.agg(["mean", "std"]).reset_index().rename(columns={"t": "horizon"}); H.to_csv(OUT / "brier_at_24_36_60.csv", index=False)

# ---- calibration from repeat-averaged out-of-fold predictions ----
O.to_csv(OUT / "oof_predictions_private.csv", index=False)     # participant-level row indices; keep private
A = O.groupby(["feature_set", "model", "idx"])[[f"surv_{h}" for h in HZ]].mean().reset_index()
rng = np.random.default_rng(0)


def km_risk(t, e, h):
    return 1 - float(sm.km_at(h, t, e)[0])


def slope(pr, t, e):
    z = np.log(-np.log(np.clip(1 - pr, 1e-9, 1 - 1e-9)))     # cloglog of predicted risk
    return float(cc.fit_fast(((z - z.mean()) / z.std())[:, None], t, e, alpha=0.0)[0] / z.std())


cal, csum = [], []
for (fs_, mname), g in A.groupby(["feature_set", "model"]):
    idx = g.idx.to_numpy(); t, e = T[idx], E[idx]
    for h in HZ:
        pr = 1 - g[f"surv_{h}"].to_numpy(); known = int(((e == 1) & (t <= h)).sum() + (t > h).sum())
        obs = km_risk(t, e, h); ex = pr.mean(); sl = slope(pr, t, e)
        bo, bs = [], []
        for _ in range(300):
            bi = rng.integers(0, len(t), len(t)); bo.append(km_risk(t[bi], e[bi], h) / pr[bi].mean()); bs.append(slope(pr[bi], t[bi], e[bi]))
        csum.append(dict(feature_set=fs_, model=mname, horizon=h, n=len(t), n_known_status=known, observed_km_risk=obs, mean_predicted_risk=ex, obs_over_exp=obs / ex,
                         oe_ci_lo=np.percentile(bo, 2.5), oe_ci_hi=np.percentile(bo, 97.5), calibration_slope=sl, slope_ci_lo=np.percentile(bs, 2.5), slope_ci_hi=np.percentile(bs, 97.5)))
        q = pd.qcut(pr, 5, labels=False, duplicates="drop")
        for k in np.unique(q):
            m_ = q == k; tg, eg = t[m_], e[m_]; ob = km_risk(tg, eg, h)
            bb = [km_risk(tg[bi], eg[bi], h) for bi in (rng.integers(0, m_.sum(), m_.sum()) for _ in range(300))]
            cal.append(dict(feature_set=fs_, model=mname, horizon=h, group=int(k) + 1, n=int(m_.sum()), mean_predicted=pr[m_].mean(), observed_km=ob, obs_ci_lo=np.percentile(bb, 2.5), obs_ci_hi=np.percentile(bb, 97.5)))
pd.DataFrame(cal).to_csv(OUT / "calibration_table.csv", index=False); C = pd.DataFrame(csum); C.to_csv(OUT / "calibration_summary.csv", index=False)

# ---- figures ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})
cal = pd.DataFrame(cal); models = ["CoxPH", "XGBoostSurvival"]; nm = {"CoxPH": "Cox PH", "XGBoostSurvival": "XGBoost Survival"}
for fs_ in ["Top50"]:
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.8), sharex=False)
    for i, mname in enumerate(models):
        for j, h in enumerate(HZ):
            a = ax[i, j]; d = cal[(cal.feature_set == fs_) & (cal.model == mname) & (cal.horizon == h)]; s = C[(C.feature_set == fs_) & (C.model == mname) & (C.horizon == h)].iloc[0]
            hi = max(d.mean_predicted.max(), d.obs_ci_hi.max()) * 1.1
            a.plot([0, hi], [0, hi], "k--", lw=.8); a.errorbar(d.mean_predicted, d.observed_km, yerr=[d.observed_km - d.obs_ci_lo, d.obs_ci_hi - d.observed_km], fmt="o-", ms=4, capsize=2, lw=1, color="#2166ac")
            a.set_xlim(0, hi); a.set_ylim(0, hi); a.grid(alpha=.25); a.set_title(f"{nm[mname]}, {h} mo", fontsize=8)
            a.text(.04, .93, f"O/E {s.obs_over_exp:.2f}\nslope {s.calibration_slope:.2f}", transform=a.transAxes, va="top", fontsize=7)
            if i == 1: a.set_xlabel("Mean predicted risk (out-of-fold)")
            if j == 0: a.set_ylabel("Observed risk (Kaplan-Meier)")
    fig.suptitle("Calibration of out-of-fold predictions, Top-50, quintile groups (95% bootstrap CI)", fontsize=8); fig.tight_layout(); fig.savefig(OUT / "fig_calibration_24_36_60.png", dpi=250); plt.close(fig)
    fig, a = plt.subplots(figsize=(3.6, 3.0)); cols = {"KM_null": "k", "CoxPH": "#d95f02", "XGBoostSurvival": "#1b9e77"}
    for mname in ["KM_null", "CoxPH", "XGBoostSurvival"]:
        d = B[(B.feature_set == fs_) & (B.model == mname) & (B.t > 0) & (B.t <= 60)].groupby("t").brier.agg(["mean", "std"])
        a.plot(d.index, d["mean"], color=cols[mname], label={"KM_null": "Kaplan-Meier null"}.get(mname, nm.get(mname)), ls="--" if mname == "KM_null" else "-"); a.fill_between(d.index, d["mean"] - d["std"], d["mean"] + d["std"], color=cols[mname], alpha=.12)
    a.set_xlabel("Months from diagnostic origin"); a.set_ylabel("IPCW Brier score (mean over 50 folds, band = SD)"); a.legend(fontsize=7); a.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(OUT / "fig_brier_over_time_vs_null.png", dpi=250); plt.close(fig)
print(pd.DataFrame(summ).round(4).to_string()); print(pd.DataFrame(paired).round(4).to_string()); print(C.round(3).to_string())