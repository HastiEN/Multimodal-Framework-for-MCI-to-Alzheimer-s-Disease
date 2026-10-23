"""B1: fold-internal feature selection. Pure functions: no file writes, no use of labels outside X_train/y_train.

Pipeline on the TRAINING part only (same thresholds/components as rank_publication_predictors.py):
  metadata/leakage-name removal (label-free) -> missingness > 40% -> variance < 0.001 or constant
  -> correlation > 0.95 (keeps higher coverage) -> four rankings (Welch-t/Mann-Whitney p, mutual information,
  XGBoost gain (inner K-fold), mean|SHAP| of an XGBoost fit) -> mean rank -> Top-k lists.
"""
import re
import numpy as np, pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

MISSING_THRESHOLD, VARIANCE_THRESHOLD, CORRELATION_THRESHOLD = 40.0, 0.001, 0.95
METADATA = [r"^PTID$", r"^RID$", r"IMAGEUID", r"IMAGE_ID", r"LONIUID", r"EXAMDATE", r"SCANDATE", r"VISCODE", r"VISIT", r"FIELD_STRENGTH", r"FSVER",
            r"IMAGE_RESOLUTION", r"MRFIELD", r"MRSEQUENCE", r"SCANNER", r"ACQUISITION", r"PROCESS", r"VERSION", r"SOFTWARE", r"FILENAME", r"FILE",
            r"SITEID", r"RECNO", r"USERDATE", r"UPDATE_STAMP", r"PHASE", r"LANGUAGE_CODE", r"QC", r"UNIT$", r"_UNIT", r"VSTMPUNT", r"VSWTUNIT",
            r"VSHTUNIT", r"MMDATE", r"MMYEAR", r"MMMONTH", r"MMDAY"]
LEAK = [r"DIAG", r"\bDX\b", r"CONVERT", r"CONVERSION", r"FOLLOWUP", r"FUTURE", r"LABEL", r"DEMENTIA", r"FINAL_DIAGNOSIS", r"BASELINE_DIAGNOSIS"]
BIO = ["CDRSB", "ADAS", "TOTSCORE", "TOTAL13", "MMSCORE", "MMSE", "FAQ", "MOCA", "NEUROBAT", "APOE", "GENOTYPE", "PTEDUCAT", "AGE", "HIPPOCAMPUS",
       "ENTORHINAL", "VENTRICLE", "CENTILOIDS", "SUMMARY_SUVR", "SUVR", "PTAU", "TAU", "ABETA", "BMI"]
XGB = dict(n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss")


def _has(name, pats):
    return any(re.search(p, name.upper()) for p in pats)


def coerce_numeric_like(x):
    out = x.copy()
    for c in out.columns:
        if out[c].dtype == object:
            n = pd.to_numeric(out[c], errors="coerce")
            nm = out[c].notna().sum()
            if nm and n.notna().sum() / nm >= 0.8:
                out[c] = n
    return out


def _prep(x):
    num = x.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in x.columns if c not in num]
    return ColumnTransformer(
        [("num", SimpleImputer(strategy="median"), num),
         ("cat", Pipeline([("i", SimpleImputer(strategy="most_frequent")), ("o", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat)],
        remainder="drop", verbose_feature_names_out=False)


def _map(names, cols, cat):
    m = {}
    for n in names:
        m[n] = n if n in cols else next((c for c in sorted(cat, key=len, reverse=True) if n.startswith(f"{c}_")), n)
    return m


def filter_features(x):
    x = coerce_numeric_like(x)
    x = x[[c for c in x.columns if not _has(c, METADATA) and not _has(c, LEAK)]]
    x = x.loc[:, x.isna().mean() * 100 <= MISSING_THRESHOLD]
    ok = []
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            v = x[c]
            var = v.var(skipna=True)
            if v.dropna().nunique() > 1 and not (pd.notna(var) and var < VARIANCE_THRESHOLD):
                ok.append(c)
        elif x[c].dropna().nunique() > 1:
            ok.append(c)
    x = x[ok]
    num = x.select_dtypes(include=[np.number]).columns.tolist()
    if len(num) > 1:
        imp = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(x[num]), columns=num)
        corr = imp.corr().abs().to_numpy()
        miss = x[num].isna().mean().to_numpy() * 100
        removed = set()
        for i in range(len(num)):
            if num[i] in removed:
                continue
            for j in range(i + 1, len(num)):
                if num[j] in removed or not corr[i, j] > CORRELATION_THRESHOLD:
                    continue
                sa = (100 - miss[i], -miss[i], sum(k in num[i].upper() for k in BIO))
                sb = (100 - miss[j], -miss[j], sum(k in num[j].upper() for k in BIO))
                removed.add(num[j] if sa >= sb else num[i])
        x = x.drop(columns=list(removed))
    return x


def _univariate(x, y):
    rows = []
    for c in x.columns:
        if pd.api.types.is_numeric_dtype(x[c]):
            a = x.loc[y == 0, c].dropna()
            b = x.loc[y == 1, c].dropna()
            if len(a) < 3 or len(b) < 3:
                p = np.nan
            else:
                try:
                    p = np.nanmin([stats.ttest_ind(a, b, equal_var=False)[1], stats.mannwhitneyu(a, b, alternative="two-sided")[1]])
                except Exception:
                    p = np.nan
        else:
            tab = pd.crosstab(x[c].fillna("MISSING"), y)
            try:
                p = stats.fisher_exact(tab)[1] if tab.shape == (2, 2) else stats.chi2_contingency(tab)[1]
            except Exception:
                p = np.nan
        rows.append((c, p))
    d = pd.DataFrame(rows, columns=["feature", "p"]).sort_values(["p", "feature"], na_position="last")
    return {f: i + 1 for i, f in enumerate(d.feature)}


def _agg(names, vals, cols, cat):
    mp = _map(names, cols, cat)
    return pd.Series(vals, index=[mp[n] for n in names]).groupby(level=0).sum()


def select_features_in_fold(X_train, y_train, ks=(20, 50, 100), seed=42, inner_splits=3, n_jobs=4):
    """Return ({k: [features]}, consensus DataFrame). Uses ONLY X_train / y_train."""
    import shap
    y = pd.Series(np.asarray(y_train).astype(int), index=X_train.index)
    x = filter_features(X_train)
    cols = x.columns.tolist()
    cat = [c for c in cols if not pd.api.types.is_numeric_dtype(x[c])]
    rk_u = _univariate(x, y)
    pre = _prep(x)
    xt = pd.DataFrame(pre.fit_transform(x), columns=pre.get_feature_names_out(), index=x.index)
    names = xt.columns.tolist()
    disc = np.array([n not in cols for n in names])
    mi = _agg(names, mutual_info_classif(xt, y, discrete_features=disc, random_state=seed), cols, cat).sort_values(ascending=False)
    gains = []
    for tr, _ in StratifiedKFold(inner_splits, shuffle=True, random_state=seed).split(xt, y):
        m = XGBClassifier(**XGB, random_state=seed, n_jobs=n_jobs).fit(xt.iloc[tr], y.iloc[tr])
        g = m.get_booster().get_score(importance_type="gain")
        gains.append(_agg(list(g), list(g.values()), cols, cat))
    gain = pd.concat(gains, axis=1).fillna(0).mean(axis=1).sort_values(ascending=False)
    m = XGBClassifier(**XGB, random_state=seed, n_jobs=n_jobs).fit(xt, y)
    sv = shap.TreeExplainer(m).shap_values(xt)
    sv = sv[1] if isinstance(sv, list) else sv
    sh = _agg(names, np.abs(sv).mean(0), cols, cat).sort_values(ascending=False)
    R = pd.DataFrame({"feature": cols})
    R["univariate"] = R.feature.map(rk_u)
    for nm, s in (("mi", mi), ("gain", gain), ("shap", sh)):
        R[nm] = R.feature.map({f: i + 1 for i, f in enumerate(s.index)})
    R = R.fillna(len(cols) + 1)
    R["avg_rank"] = R[["univariate", "mi", "gain", "shap"]].mean(axis=1)
    R = R.sort_values(["avg_rank", "feature"]).reset_index(drop=True)
    return {k: R.feature.head(k).tolist() for k in ks}, R