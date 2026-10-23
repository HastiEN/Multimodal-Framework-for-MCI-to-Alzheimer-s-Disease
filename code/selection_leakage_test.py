"""Selection-leakage test: does ranking features on ALL participants inflate held-out performance?

Each repetition is a literal 1000/250 participant-level split (stratified by outcome and ADNI phase).
Arms (Top-50 lists, paper protocol = ranking on the any-future label):
  A  the paper's published Top-50 list (ranked on all 1,250 participants)
  B  same selector re-run on ALL 1,250 participants (global ranking; test participants influence selection)
  C  same selector run on the 1000 development participants only (honest; the 250 never touch selection)
Models are trained on the development participants eligible for horizon H and scored once on the 250's eligible subset.
Delta(B-C) isolates optimism due to selecting on the test participants.  --perms N runs a label-permutation null.
"""
import argparse, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
import fold_selection as fs
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

HORIZONS = (24, 36, 60)
PRIMARY = {24: "RandomForest", 36: "RandomForest", 60: "CatBoost"}   # best model per horizon in the paper (Top-50)
MARGIN = 0.03


def load():
    d = pd.read_csv(ROOT / "data/processed/final_mci_to_ad_publication_dataset.csv", low_memory=False)
    c = pd.read_csv(ROOT / "data/processed/strict_mci_to_ad_publication_cohort.csv")
    ph = pd.read_csv(ROOT / "audit/verified_tables/baseline_cohort_with_followup_audit.csv")[["PTID", "baseline_phase"]]
    assert (d.PTID.values == c.PTID.values).all()
    c = c.merge(ph, on="PTID", how="left")
    c["fm"] = (pd.to_datetime(c.last_followup_date) - pd.to_datetime(c.baseline_date)).dt.days / 30.4375
    X = d.drop(columns=["PTID", "conversion_label"])
    y_any = d.conversion_label.astype(int).values
    elig, lab = {}, {}
    for H in HORIZONS:
        conv = (c.conversion_label == 1) & c.months_to_conversion.notna()
        pos = conv & (c.months_to_conversion <= H)
        neg = (conv & (c.months_to_conversion > H)) | ((c.conversion_label == 0) & (c.fm >= H))
        elig[H] = (pos | neg).values
        lab[H] = pos.astype(int).values
    strata = (c.baseline_phase.astype(str) + "_" + c.conversion_label.astype(str)).values
    paper_top50 = pd.read_csv(ROOT / "data/processed/publication_feature_ranking/top50_features.csv").feature.tolist()
    return X, y_any, elig, lab, strata, paper_top50, c.PTID.values


def make_model(name, n_jobs=2):
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=n_jobs)
    if name == "XGBoost":
        return XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42, n_jobs=n_jobs)
    return CatBoostClassifier(iterations=1000, depth=6, learning_rate=0.03, loss_function="Logloss", verbose=False, random_state=42, thread_count=n_jobs, allow_writing_files=False)


def fit_score(Xtr, ytr, Xte, yte, feats, model):
    xtr, xte = fs.coerce_numeric_like(Xtr[feats]), fs.coerce_numeric_like(Xte[feats])
    num = xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in xtr.columns if c not in num]
    pre = ColumnTransformer([("n", SimpleImputer(strategy="median"), num),
                             ("c", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)])
    pipe = Pipeline([("p", pre), ("m", make_model(model))]).fit(xtr, ytr)
    p = pipe.predict_proba(xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    return dict(auc=roc_auc_score(yte, p), pr_auc=average_precision_score(yte, p), acc=accuracy_score(yte, pred), bal_acc=balanced_accuracy_score(yte, pred))


def one_rep(s, X, y_any, elig, lab, strata, lists, models, perm=None, arms=("A", "B", "C")):
    """lists: dict arm->top50 for A and B (global). C computed here on the dev part only."""
    seed = (5000 + s) if perm is not None else (1000 + s)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=250, random_state=seed)
    dev, test = next(sss.split(X, strata if perm is None else y_any))
    if perm is None:
        Cl, _ = fs.select_features_in_fold(X.iloc[dev], y_any[dev], ks=(50,), seed=42, n_jobs=2)
        lists = dict(lists, C=Cl[50])
    rows = []
    for H in HORIZONS:
        de, te = dev[elig[H][dev]], test[elig[H][test]]
        for arm in arms:
            for m in models(H):
                r = fit_score(X.iloc[de], lab[H][de], X.iloc[te], lab[H][te], lists[arm], m)
                rows.append(dict(rep=s, horizon=H, arm=arm, model=m, n_dev=len(de), n_test=len(te), n_test_pos=int(lab[H][te].sum()), **r))
    return rows, lists["C"] if perm is None else None, (test if perm is None else None)


def summarize(df, out, R):
    prim = df[df.apply(lambda r: r.model == PRIMARY[r.horizon], axis=1)]
    rows, diffs = [], []
    for H in HORIZONS:
        g = prim[prim.horizon == H]
        for arm in "ABC":
            a = g[g.arm == arm]
            rows.append(dict(horizon=H, model=PRIMARY[H], arm=arm, n_test_mean=a.n_test.mean(), pos_test_mean=a.n_test_pos.mean(),
                             auc_mean=a.auc.mean(), auc_sd=a.auc.std(), auc_min=a.auc.min(), auc_max=a.auc.max(), pr_auc_mean=a.pr_auc.mean(), acc_mean=a.acc.mean(), bal_acc_mean=a.bal_acc.mean()))
    pd.DataFrame(rows).to_csv(out / "summary_primary_model.csv", index=False)
    for m_name, sub in [("primary", prim), ("all_models", df)]:
        for (H, mo), g in sub.groupby(["horizon", "model"]):
            w = g.pivot(index="rep", columns="arm", values="auc")
            for lhs, rhs, label in (("B", "C", "B-C (selection on test participants)"), ("A", "C", "A-C (paper list vs honest)"), ("A", "B", "A-B")):
                d = (w[lhs] - w[rhs]).values
                Rr = len(d)
                se = np.sqrt((1 / Rr + 250 / 1000) * d.var(ddof=1))      # Nadeau-Bengio corrected resampled variance
                t95, t90 = stats.t.ppf(0.975, Rr - 1), stats.t.ppf(0.95, Rr - 1)
                diffs.append(dict(set=m_name, horizon=H, model=mo, contrast=label, mean_delta_auc=d.mean(), sd_across_splits=d.std(ddof=1),
                                  ci95_lo=d.mean() - t95 * se, ci95_hi=d.mean() + t95 * se, ci90_lo=d.mean() - t90 * se, ci90_hi=d.mean() + t90 * se,
                                  share_splits_positive=float((d > 0).mean()), equivalent_within_0p03=bool(d.mean() - t90 * se > -MARGIN and d.mean() + t90 * se < MARGIN), n_splits=Rr))
    D = pd.DataFrame(diffs)
    D.to_csv(out / "paired_differences.csv", index=False)
    return prim, D


def plots(prim, D, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    cols = {"A": "#c44e52", "B": "#dd8452", "C": "#4c72b0"}
    for i, H in enumerate(HORIZONS):
        for j, arm in enumerate("ABC"):
            v = prim[(prim.horizon == H) & (prim.arm == arm)].auc.values
            ax[0].boxplot(v, positions=[i * 4 + j], widths=0.8, patch_artist=True, boxprops=dict(facecolor=cols[arm], alpha=.75), medianprops=dict(color="k"), showfliers=False)
    ax[0].set_xticks([1 + 4 * i for i in range(3)]); ax[0].set_xticklabels([f"{H} mo" for H in HORIZONS]); ax[0].set_ylabel("Held-out ROC-AUC (250 participants)")
    ax[0].legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=cols[a], alpha=.75) for a in "ABC"], labels=["A paper list", "B re-ranked on all 1,250", "C ranked on 1,000 only (honest)"], fontsize=8, loc="upper left")
    ax[0].set_ylim(0.805, 0.99); ax[0].set_title("Repeated 1000/250 splits (30 splits)"); ax[0].grid(alpha=.25)
    d = D[(D.set == "primary") & (D.contrast.str.startswith("B-C"))].sort_values("horizon")
    ax[1].errorbar(range(3), d.mean_delta_auc, yerr=[d.mean_delta_auc - d.ci95_lo, d.ci95_hi - d.mean_delta_auc], fmt="o", capsize=5, color="#333")
    ax[1].axhline(0, color="k", lw=.8); ax[1].axhspan(-MARGIN, MARGIN, color="#9ecae1", alpha=.25, label="equivalence margin ±0.03")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels([f"{H} mo" for H in d.horizon]); ax[1].set_ylabel("ΔAUC (B − C), mean with 95% CI"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.25)
    ax[1].set_title("Optimism from selecting on test participants")
    fig.tight_layout(); fig.savefig(out / "fig_selection_leakage_test.png", dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--splits", type=int, default=30); ap.add_argument("--perms", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--out", default=str(ROOT / "Claude_Code_Outputs/B_selection_leakage_test"))
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    X, y_any, elig, lab, strata, paper, ptid = load()
    t0 = time.time()
    Bl, Brank = fs.select_features_in_fold(X, y_any, ks=(50,), seed=42, n_jobs=4)
    print(f"global ranking done {time.time()-t0:.0f}s; overlap(B, paper list) = {len(set(Bl[50]) & set(paper))}/50", flush=True)
    if a.perms == 0:
        models = lambda H: ["RandomForest", "XGBoost", "CatBoost"]
        res = Parallel(n_jobs=a.jobs, verbose=5)(delayed(one_rep)(s, X, y_any, elig, lab, strata, dict(A=paper, B=Bl[50]), models) for s in range(a.splits))
        df = pd.DataFrame([r for rr, _, _ in res for r in rr]); df.to_csv(out / "results_long.csv", index=False)
        C = [c for _, c, _ in res]
        ov = pd.DataFrame(dict(rep=range(len(C)), overlap_C_vs_B=[len(set(c) & set(Bl[50])) for c in C], overlap_C_vs_paper=[len(set(c) & set(paper)) for c in C],
                               overlap_B_vs_paper=len(set(Bl[50]) & set(paper))))
        ov.to_csv(out / "top50_overlap.csv", index=False)
        freq = pd.Series([f for c in C for f in c]).value_counts().rename("n_splits_selected").reset_index().rename(columns={"index": "feature"})
        freq["in_paper_top50"] = freq.iloc[:, 0].isin(paper); freq["in_B"] = freq.iloc[:, 0].isin(Bl[50]); freq.to_csv(out / "top50_selection_frequency.csv", index=False)
        pd.DataFrame(dict(rep=range(len(C)), test_ptids=[";".join(ptid[t]) for _, _, t in res])).to_csv(out / "test_split_ids.csv", index=False)
        prim, D = summarize(df, out, a.splits); plots(prim, D, out)
        print(pd.read_csv(out / "summary_primary_model.csv").round(3).to_string()); print(D[D.set == "primary"].round(4).to_string()); print(ov.describe().round(1))
    else:
        rows = []
        for p in range(a.perms):
            pi = np.random.default_rng(777 + p).permutation(len(y_any))
            yp = y_any[pi]; labp = {H: lab[H][pi] for H in HORIZONS}; elp = {H: elig[H][pi] for H in HORIZONS}
            Bp, _ = fs.select_features_in_fold(X, yp, ks=(50,), seed=42, n_jobs=4)
            sss = StratifiedShuffleSplit(n_splits=1, test_size=250, random_state=9000 + p); dev, test = next(sss.split(X, yp))
            Cp, _ = fs.select_features_in_fold(X.iloc[dev], yp[dev], ks=(50,), seed=42, n_jobs=4)
            for H in HORIZONS:
                de, te = dev[elp[H][dev]], test[elp[H][test]]
                if len(np.unique(labp[H][te])) < 2: continue
                for arm, L in (("B", Bp[50]), ("C", Cp[50])):
                    r = fit_score(X.iloc[de], labp[H][de], X.iloc[te], labp[H][te], L, "RandomForest")
                    rows.append(dict(perm=p, horizon=H, arm=arm, n_test=len(te), **r))
            print(f"perm {p+1}/{a.perms} done {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "permutation_null_long.csv", index=False)
        P = pd.DataFrame(rows)
        S = P.groupby(["horizon", "arm"]).auc.agg(["mean", "std", "count"]).reset_index(); S.to_csv(out / "permutation_null_summary.csv", index=False); print(S.round(4))


if __name__ == "__main__":
    main()