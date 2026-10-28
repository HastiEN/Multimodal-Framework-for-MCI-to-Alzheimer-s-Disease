"""A2: strict-date predictor extraction (D1 rule). The historical extractor is untouched.

Rule: origin = participant baseline_date (earliest sc/bl MCI diagnosis visit).
A source row is usable iff its own measurement date (first of EXAMDATE, VISDATE, SCANDATE, APTESTDT,
EXAM_DATE, Study Date; USERDATE is a data-entry date and is NOT accepted) satisfies
origin - LOOKBACK <= date <= origin.  One WHOLE row is selected per participant and table (nearest to
origin; tie -> exact baseline visit code, then later visit order). No fallback to later rows,
no column-wise first().  Usage: python extract_strict_predictors.py OUTDIR [LOOKBACK_DAYS]
"""
import os, sys
from pathlib import Path
os.environ.setdefault("ICBME_RUN_DIR", str(Path(__file__).resolve().parents[1] / "runs" / "strict_v1"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
import extract_baseline_predictors as E  # helper functions only (main() not called)

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
LOOKBACK = int(sys.argv[2]) if len(sys.argv) > 2 else 180
MEAS_DATES = ["EXAMDATE", "VISDATE", "SCANDATE", "APTESTDT", "EXAM_DATE", "Study Date"]
SENTINELS_ALWAYS = {-4.0, -999.0, -9999.0, -99.0}
cohort = pd.read_csv(E.PROCESSED_DIR / "strict_mci_to_ad_publication_cohort.csv")
cohort["PTID"] = E.normalize_id(cohort["PTID"]); cohort["baseline_date"] = pd.to_datetime(cohort["baseline_date"])
cohort["_rid"] = E.normalize_id(cohort["RID"]); cohort["_bl"] = cohort["baseline_visit"].astype(str).str.strip().str.lower()
N = len(cohort)
files = sorted(p for p in E.RAW_DIR.iterdir() if p.suffix.lower() == ".csv")
blocks = [cohort[["PTID", "conversion_label"]].copy()]; cov, prov, cat, leak, pol, notes = [], [], [], [], [], []

for i, path in enumerate(files, 1):
    raw = pd.read_csv(path, low_memory=False)
    sc = E.choose_subject_col(raw)
    cols = [c for c in raw.columns if E.is_candidate_feature(path.name, c, raw[c])]
    dcol = next((c for c in MEAS_DATES if c in raw.columns), None)
    if not sc or not cols:
        continue
    if dcol is None:
        notes.append(dict(file=path.name, n_candidates=len(cols), reason="no measurement date column (USERDATE-only or none): all features missing")); continue
    key = "_rid" if sc == "RID" else "PTID"
    r = raw.copy(); r["_k"] = E.normalize_id(r[sc]); r["_d"] = pd.to_datetime(r[dcol], errors="coerce")
    vc = r["VISCODE2"] if "VISCODE2" in r.columns else (r["VISCODE"] if "VISCODE" in r.columns else pd.Series(np.nan, index=r.index))
    r["_vn"] = vc.astype(str).str.strip().str.lower(); r["_vo"] = vc.apply(E.visit_order)
    m = cohort[[key, "PTID", "baseline_date", "_bl"]].merge(r, left_on=key, right_on="_k", how="inner", suffixes=("", "_raw"))
    m["_off"] = (m["baseline_date"] - m["_d"]).dt.days
    m = m[m["_off"].between(0, LOOKBACK)]
    m["_exact"] = (m["_vn"] == m["_bl"]).astype(int)
    m = m.sort_values(["PTID", "_off", "_exact", "_vo"], ascending=[True, True, False, False]).drop_duplicates("PTID", keep="first")
    assert (m["_d"] <= m["baseline_date"]).all(), f"post-origin row selected in {path.name}"
    sel = cohort[["PTID"]].merge(m, on="PTID", how="left")
    pref = E.source_prefix(path.name); blk = {"PTID": cohort["PTID"].values}
    prov.append(pd.DataFrame(dict(PTID=sel.PTID, source=path.name, date_col=dcol, row_date=sel["_d"], offset_days=sel["_off"], visit=sel["_vn"])))
    for c in cols:
        v = sel[c] if c in sel.columns else pd.Series(np.nan, index=sel.index)
        if c not in sel.columns: continue
        dt = E.infer_data_type(v)
        if dt == "numeric_like": v = pd.to_numeric(v, errors="coerce"); dt = "numeric"
        if dt == "text_or_high_cardinality": continue
        if pd.api.types.is_numeric_dtype(v):
            vv = v.dropna(); neg = vv[vv < 0]
            codes = sorted(set(neg.unique()) & SENTINELS_ALWAYS)
            if (vv == -1).any() and (neg == -1).all():  # -1 is the ONLY negative value in this column
                codes.append(-1.0)
            if codes:
                n_rep = int(v.isin(codes).sum()); v = v.mask(v.isin(codes))
                pol.append(dict(feature=f"{pref}_{E.sanitize_name(c)}", source=path.name, codes_replaced=str(codes), n_replaced=n_rep,
                                other_negative_values_kept=int((neg.isin(codes) == False).sum()), status="provisional: confirm against ADNI data dictionary"))
        name = f"{pref}_{E.sanitize_name(c)}"; blk[name] = v.values
        a = int(pd.Series(v).notna().sum()); cp = 100 * a / N
        category = E.infer_category(path.name, c); st, _ = E.leakage_status(path.name, c)
        cov.append(dict(feature_name=name, source_file=path.name, category=category, available_subjects=a, coverage_percent=cp,
                        missing_percent=100 - cp, leakage_status=st, recommendation_score=E.recommendation_score(category, cp, 100 - cp, st, name)))
    if len(blk) > 1: blocks.append(pd.DataFrame(blk))
    print(f"[{i}/{len(files)}] {path.name} ok", flush=True)

cov = pd.DataFrame(cov); cov.to_csv(OUT / "strict_coverage.csv", index=False)
rec = cov[(cov.leakage_status == "SAFE") & (cov.coverage_percent >= 20) & (cov.recommendation_score >= 55)]
final = blocks[0]
for b in blocks[1:]: final = final.merge(b, on="PTID", how="left")
keep = [c for c in rec.feature_name if c in final.columns]
final[["PTID", "conversion_label"] + keep].to_csv(OUT / "strict_predictors.csv", index=False)
P = pd.concat(prov); P.to_csv(OUT / "strict_provenance.csv", index=False)
pd.DataFrame(pol).to_csv(OUT / "missing_code_policy.csv", index=False); pd.DataFrame(notes).to_csv(OUT / "files_without_measurement_date.csv", index=False)
assert (P.row_date.dropna() <= P.merge(cohort[["PTID", "baseline_date"]], on="PTID").set_index(P.index).baseline_date[P.row_date.notna()]).all()
# before/after counts per modality vs historical
hist = pd.read_csv(E.PROCESSED_DIR / "feature_coverage_report.csv")
h = hist.groupby("category").available_subjects.mean().rename("historical_mean_available")
s = cov[cov.feature_name.isin(keep)].groupby("category").available_subjects.mean().rename("strict_mean_available")
cnt = pd.concat([h, s, hist.groupby("category").size().rename("historical_n_cand"), cov.groupby("category").size().rename("strict_n_cand")], axis=1)
cnt.to_csv(OUT / "strict_cohort_counts.csv")
nofeat = int((final[keep].notna().sum(axis=1) == 0).sum())
print("lookback", LOOKBACK, "| cohort", N, "| strict features kept", len(keep), "| participants with no strict input", nofeat)
print(cnt.round(1))
