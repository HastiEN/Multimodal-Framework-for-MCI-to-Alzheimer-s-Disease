"""A1: recover missing non-converter follow-up dates from raw DXSUM (read-only on inputs).
Writes only to the output directory given as argv[1]."""
import sys, pandas as pd, numpy as np
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
aud = pd.read_csv("audit/verified_tables/baseline_cohort_with_followup_audit.csv", parse_dates=["baseline_date", "last_followup_date", "conversion_date"])
raw = pd.read_csv("data/raw/Radipharmaceutical_Study-first_edite_DXSUM_19May2026.csv", dtype=str)
raw["d"] = pd.to_datetime(raw["EXAMDATE"], errors="coerce")
raw["dx"] = pd.to_numeric(raw["DIAGNOSIS"], errors="coerce")
rows = []
for _, r in aud[aud.last_followup_date.isna() & (aud.conversion_label == 0)].iterrows():
    g = raw[(raw.PTID == r.PTID) & raw.d.notna() & raw.dx.notna() & (raw.d > r.baseline_date)]
    if g.empty:
        rows.append(dict(PTID=r.PTID, old_date=None, new_date=None, source_table="DXSUM", source_viscode=None, new_followup_months=np.nan)); continue
    last = g.sort_values("d").iloc[-1]
    rows.append(dict(PTID=r.PTID, old_date=None, new_date=last.d.date(), source_table="DXSUM",
                     source_viscode=last.VISCODE2 or last.VISCODE, final_dx_code=int(last.dx),
                     new_followup_months=(last.d - r.baseline_date).days / 30.4375))
rec = pd.DataFrame(rows); rec.to_csv(out / "followup_recovery.csv", index=False)
aud = aud.merge(rec[["PTID", "new_followup_months"]], on="PTID", how="left")
aud["followup_months_recovered"] = aud.followup_months.fillna(aud.new_followup_months)
aud.to_csv(out / "cohort_with_recovered_followup.csv", index=False)
nc = aud[aud.conversion_label == 0]
old, new = nc.followup_months, nc.followup_months_recovered
summ = dict(n_recovered=int(rec.new_followup_months.notna().sum()), n_candidates=len(rec),
            noncon_median_old=old.median(), noncon_median_new=new.median(),
            noncon_iqr_new=[new.quantile(.25), new.quantile(.75)], noncon_lt24_old=int((old < 24).sum()), noncon_lt24_new=int((new < 24).sum()),
            survival_n_old=int(((aud.conversion_label==1)&aud.months_to_conversion.notna()).sum()+(nc.followup_months.notna()).sum()), survival_n_new=int(((aud.conversion_label==1)&aud.months_to_conversion.notna()).sum()+(nc.followup_months_recovered.notna()).sum()),
            unrecoverable_converter=aud[aud.missing_conversion_date == True].PTID.tolist())
pd.Series(summ).to_csv(out / "summary.csv"); print(summ)
# compare to earlier-audit candidates
c = pd.read_csv("audit/verified_tables/missing_followup_date_recovery_candidates.csv").merge(rec, on="PTID")
print("max abs diff vs earlier audit (months):", (c.recoverable_followup_months - c.new_followup_months).abs().max())
