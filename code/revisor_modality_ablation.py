"""Task 6: clinical-only and modality ablations with fold-internal feature selection.

Pool: the 1,072 candidate predictors of the primary analysis, grouped by source table:
  CLIN  cognitive/functional/demographic/genetic/vitals/other non-imaging, non-fluid tables
  MRI   FreeSurfer (UCSFFSX7) and WMH (UCD_WMH);  PET  UCBERKELEY amyloid ROI/summary;  LAB  fluid/blood/CSF biomarker and lab tables
  DXADJ = ADAS, CDR, FAQ, MMSE, MoCA items and logical-memory totals (diagnosis-adjacent); CLIN_noDx = CLIN minus DXADJ.
Protocol: repeated stratified 5-fold CV (4 repeats = 20 folds, strata = ADNI phase x any-future label) over all participants (or the PET-observed subset).
In every training part: filters + 4-way consensus ranking on the any-future label (fold_selection.py), Top-50 (or fewer if the pool is smaller);
then for each horizon the model is trained on training participants eligible for that horizon and scored on eligible test participants.
Models: paper-best per horizon (RF at 24/36 mo, CatBoost at 60 mo) and XGBoost at all horizons. Metric: ROC-AUC (PR-AUC also saved).
"""
import argparse, sys, time as _time, re, warnings
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
import fold_selection as fs
import selection_leakage_test as L
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = ROOT / "revisor_analysis" / "6_modality_ablation"; OUT.mkdir(parents=True, exist_ok=True)
CLIN_SRC = {"CDR", "FAQ", "ADAS", "MMSE", "MOCA", "NEUROBAT", "GDSCALE", "PTDEMOG", "VITALS", "PHYSICAL", "NEUROEXM", "MODHACH", "GENETIC", "RECCMEDS", "CLIELG", "BLSCHECK"}
MRI_SRC = {"UCSFFSX7", "UCD_WMH"}; PET_SRC = {"UCBERKELEY_AMY_6MM"}
LAB_SRC = {"LABDATA", "BIOMARK", "FOXLABBSI", "LOCLAB", "LABTESTS", "UPENNBIOMK_ROCHE_ELECSYS"}
DXADJ_SRC = {"ADAS", "CDR", "FAQ", "MMSE", "MOCA"}
AMY = "UCBERKELEY_AMY_6MM_SUMMARY_SUVR"
MODELS = lambda H: [L.PRIMARY[H], "XGBoost"] if L.PRIMARY[H] != "XGBoost" else ["XGBoost"]


def groups(cols):
    cat = pd.read_csv(ROOT / "data/processed/candidate_feature_catalog.csv")[["feature_name", "source_file"]]
    src = dict(zip(cat.feature_name, cat.source_file.str.replace("All_Subjects_", "").str.replace("_08Jun2026.csv", "")))
    g = {}
    for c in cols:
        s = src[c]
        g[c] = "CLIN" if s in CLIN_SRC else "MRI" if s in MRI_SRC else "PET" if s in PET_SRC else "LAB" if s in LAB_SRC else "META"
    dx = {c for c in cols if src[c] in DXADJ_SRC or re.search(r"LDEL|LIMM", c.upper())}
    return g, dx


def build_sets(cols):
    g, dx = groups(cols)
    pick = lambda *k: [c for c in cols if g[c] in k]
    clin = pick("CLIN")
    return g, {"CLIN": clin, "MRI": pick("MRI"), "PET": pick("PET"), "CLIN+MRI": pick("CLIN", "MRI"), "CLIN+PET": pick("CLIN", "PET"),
               "CLIN+MRI+PET": pick("CLIN", "MRI", "PET"), "ALL": list(cols), "CLIN_noDx+MRI+PET": [c for c in pick("CLIN", "MRI", "PET") if c not in dx]}


def unit(analysis, set_name, fold, tr, te, X, y_any, elig, lab, feats, g):
    sel, R = fs.select_features_in_fold(X.iloc[tr][feats], y_any[tr], ks=(50,), seed=42, n_jobs=2)
    top = sel[50]; rows = []
    for H in L.HORIZONS:
        de, tt = tr[elig[H][tr]], te[elig[H][te]]
        if len(np.unique(lab[H][tt])) < 2 or len(np.unique(lab[H][de])) < 2: continue
        for m in MODELS(H):
            r = L.fit_score(X.iloc[de], lab[H][de], X.iloc[tt], lab[H][tt], top, m)
            rows.append(dict(analysis=analysis, set=set_name, fold=fold, horizon=H, model=m, primary=(m == L.PRIMARY[H]), n_test=len(tt), n_test_pos=int(lab[H][tt].sum()), n_selected=len(top), **r))
    comp = pd.Series([g[c] for c in top]).value_counts().to_dict()
    return rows, dict(analysis=analysis, set=set_name, fold=fold, n_pool_after_filters=len(R), **{f"top50_{k}": v for k, v in comp.items()})


def corr_t(d, ratio=0.25):
    n = len(d); se = np.sqrt((1 / n + ratio) * d.var(ddof=1)); q = stats.t.ppf(.975, n - 1); m = d.mean()
    return m, m - q * se, m + q * se, (2 * (1 - stats.t.cdf(abs(m / se), n - 1)) if se > 0 else np.nan)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--repeats", type=int, default=4); ap.add_argument("--jobs", type=int, default=5); a = ap.parse_args()
    X, y_any, elig, lab, strata, paper, ptid = L.load()
    cols = X.columns.tolist(); g, sets = build_sets(cols)
    pd.DataFrame([dict(group=k, n_features=v) for k, v in pd.Series(g).value_counts().items()]).to_csv(OUT / "group_sizes.csv", index=False)
    pd.DataFrame([dict(set=k, n_candidate_features=len(v), median_missing=float(X[v].isna().mean().median())) for k, v in sets.items()]).to_csv(OUT / "set_definitions.csv", index=False)
    pet_obs = X[AMY].notna().to_numpy(); print("PET observed:", int(pet_obs.sum()), flush=True)
    jobs = []
    for analysis, mask, set_list in (("all_participants", np.ones(len(X), bool), list(sets)), ("PET_observed_only", pet_obs, ["CLIN", "PET", "CLIN+PET", "CLIN+MRI", "CLIN+MRI+PET", "ALL"])):
        idx = np.where(mask)[0]; cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=a.repeats, random_state=2026)
        for f, (tr, te) in enumerate(cv.split(idx, strata[idx]), start=1):
            for s in set_list: jobs.append((analysis, s, f, idx[tr], idx[te]))
    print("units:", len(jobs), flush=True); t0 = _time.time()
    res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(unit)(an, s, f, tr, te, X, y_any, elig, lab, sets[s], g) for an, s, f, tr, te in jobs)
    print(f"done in {(_time.time() - t0) / 60:.1f} min", flush=True)
    R = pd.DataFrame([r for rr, _ in res for r in rr]); R.to_csv(OUT / "ablation_long.csv", index=False)
    C = pd.DataFrame([c for _, c in res]).fillna(0); C.to_csv(OUT / "top50_composition_by_fold.csv", index=False)
    P = R[R.primary]
    S = P.groupby(["analysis", "horizon", "set"]).agg(model=("model", "first"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"), pr_auc_mean=("pr_auc", "mean"), n_test_mean=("n_test", "mean"), pos_mean=("n_test_pos", "mean"), n_folds=("auc", "size")).reset_index()
    q = stats.t.ppf(.975, S.n_folds - 1); se = np.sqrt((1 / S.n_folds + .25) * S.auc_sd ** 2); S["ci95_lo"], S["ci95_hi"] = S.auc_mean - q * se, S.auc_mean + q * se
    S.to_csv(OUT / "ablation_summary_primary_model.csv", index=False)
    X2 = R[R.model == "XGBoost"].groupby(["analysis", "horizon", "set"]).auc.agg(["mean", "std"]).reset_index(); X2.to_csv(OUT / "ablation_summary_xgboost.csv", index=False)
    pairs = [("CLIN+PET", "CLIN"), ("CLIN+MRI", "CLIN"), ("CLIN+MRI+PET", "CLIN"), ("CLIN+MRI+PET", "CLIN+MRI"), ("CLIN+MRI+PET", "CLIN+PET"), ("ALL", "CLIN+MRI+PET"), ("MRI", "CLIN"), ("PET", "CLIN"), ("CLIN_noDx+MRI+PET", "CLIN+MRI+PET")]
    rows = []
    for (an, H, mo), gdf in R[R.model.isin([*L.PRIMARY.values(), "XGBoost"])].groupby(["analysis", "horizon", "model"]):
        w = gdf.pivot(index="fold", columns="set", values="auc")
        for a_, b_ in pairs:
            if a_ in w and b_ in w:
                d = (w[a_] - w[b_]).dropna()
                if len(d) > 5:
                    m, lo, hi, p = corr_t(d.values); rows.append(dict(analysis=an, horizon=H, model=mo, contrast=f"{a_} - {b_}", mean_diff=m, ci95_lo=lo, ci95_hi=hi, p_corrected_t=p, share_folds_positive=float((d > 0).mean()), n_folds=len(d)))
    D = pd.DataFrame(rows); D.to_csv(OUT / "ablation_paired_contrasts.csv", index=False)
    # figure
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8})
    order = ["CLIN", "MRI", "PET", "CLIN+MRI", "CLIN+PET", "CLIN+MRI+PET", "CLIN_noDx+MRI+PET", "ALL"]; colr = {24: "#4c72b0", 36: "#dd8452", 60: "#55a868"}
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
    for k, an in enumerate(["all_participants", "PET_observed_only"]):
        sub = S[S.analysis == an]; sets_here = [s for s in order if s in set(sub.set)]
        for j, H in enumerate(L.HORIZONS):
            d = sub[sub.horizon == H].set_index("set").reindex(sets_here); x = np.arange(len(sets_here)) + (j - 1) * .26
            ax[k].errorbar(x, d.auc_mean, yerr=[d.auc_mean - d.ci95_lo, d.ci95_hi - d.auc_mean], fmt="o", ms=3.5, capsize=2, lw=.9, color=colr[H], label=f"{H} mo")
        ax[k].set_xticks(range(len(sets_here))); ax[k].set_xticklabels(sets_here, rotation=35, ha="right"); ax[k].grid(alpha=.25)
        ax[k].set_title(("All participants (n=1,250)" if k == 0 else f"PET-observed participants (n={int(pet_obs.sum())})"), fontsize=8)
    ax[0].set_ylabel("Held-out ROC-AUC (mean, 95% CI)"); ax[0].legend(fontsize=7, loc="lower right"); fig.tight_layout(); fig.savefig(OUT / "fig_modality_ablation.png", dpi=250); plt.close(fig)
    print(S.round(3).to_string()); print(D[D.model.isin(["RandomForest", "CatBoost"])].round(4).to_string())


if __name__ == "__main__":
    main()