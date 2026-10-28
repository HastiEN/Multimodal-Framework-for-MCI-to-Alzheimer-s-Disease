"""Task 2: baseline age, sex and education by outcome and ADNI phase (descriptive; PTDEMOG demographics, baseline date from the cohort)."""
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "revisor_analysis" / "2_age_sex_education"; OUT.mkdir(parents=True, exist_ok=True)
c = pd.read_csv(ROOT / "audit/verified_tables/baseline_cohort_with_followup_audit.csv", parse_dates=["baseline_date"])[["PTID", "baseline_date", "baseline_phase", "conversion_label"]]
d = pd.read_csv(ROOT / "data/raw/ALL/All_Subjects_PTDEMOG_08Jun2026.csv", low_memory=False, usecols=["PTID", "VISDATE", "PTGENDER", "PTDOB", "PTEDUCAT"])
d["VISDATE"] = pd.to_datetime(d.VISDATE, errors="coerce")
d = d.merge(c[["PTID", "baseline_date"]], on="PTID")
d = d[d.VISDATE.notna()].sort_values(["PTID", "VISDATE"])
# time-invariant fields: first non-missing value per participant (earliest dated row); education: row nearest to baseline on or before it, else earliest
d["PTGENDER"] = d.PTGENDER.where(d.PTGENDER.isin([1, 2])); d["PTEDUCAT"] = d.PTEDUCAT.where(d.PTEDUCAT >= 0)
g = d.groupby("PTID").agg(PTGENDER=("PTGENDER", "first"), PTDOB=("PTDOB", lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan), PTEDUCAT=("PTEDUCAT", "first")).reset_index()
t = c.merge(g, on="PTID", how="left")
dob = pd.to_datetime("15/" + t.PTDOB.astype(str), format="%d/%m/%Y", errors="coerce")   # month/year only: mid-month assumed
t["age"] = (t.baseline_date - dob).dt.days / 365.25
t["sex"] = t.PTGENDER.map({1.0: "Male", 2.0: "Female"})
t["outcome"] = t.conversion_label.map({0: "No observed conversion", 1: "Converted"})
t.to_csv(OUT / "participant_demographics_checked.csv", index=False)   # keep private: participant-level
# sanity: sex coding against the ADNI IDA export (Sex = M/F)
ida = pd.read_csv(ROOT / "data/raw/idaSearch_5_28_2026(1).csv", usecols=["Subject ID", "Sex", "Age"]).drop_duplicates("Subject ID")
chk = t.merge(ida, left_on="PTID", right_on="Subject ID", how="inner")
print("sex coding check (PTGENDER vs IDA Sex):"); print(pd.crosstab(chk.PTGENDER, chk.Sex))
print("age vs IDA Age at some scan (corr, only as plausibility):", round(chk[["age", "Age"]].corr().iloc[0, 1], 3))


def block(df):
    a = df.age.dropna(); e = df.PTEDUCAT.dropna(); s = df.sex.dropna()
    return {"N": len(df), "Age, mean (SD)": f"{a.mean():.1f} ({a.std():.1f})", "Age, median [IQR]": f"{a.median():.1f} [{a.quantile(.25):.1f}-{a.quantile(.75):.1f}]",
            "Female, n (%)": f"{(s == 'Female').sum()} ({100 * (s == 'Female').mean():.1f}%)", "Education (years), mean (SD)": f"{e.mean():.1f} ({e.std():.1f})",
            "Education (years), median [IQR]": f"{e.median():.0f} [{e.quantile(.25):.0f}-{e.quantile(.75):.0f}]",
            "Missing age/sex/education (n)": f"{df.age.isna().sum()}/{df.sex.isna().sum()}/{df.PTEDUCAT.isna().sum()}"}


cols = {"All": t}
for o, g_ in t.groupby("outcome"): cols[o] = g_
T = pd.DataFrame({k: block(v) for k, v in cols.items()}); T.to_csv(OUT / "table_age_sex_education_by_outcome.csv")
P = pd.DataFrame({f"{ph} | {o}": block(g_) for (ph, o), g_ in t.groupby(["baseline_phase", "outcome"])}); P.to_csv(OUT / "table_age_sex_education_by_phase_outcome.csv")
# standardised mean differences (converted vs not) for the three variables
cv, nc = t[t.conversion_label == 1], t[t.conversion_label == 0]
smd = lambda a, b: (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
S = pd.DataFrame([dict(variable="Age", smd=smd(cv.age.dropna(), nc.age.dropna())), dict(variable="Education", smd=smd(cv.PTEDUCAT.dropna(), nc.PTEDUCAT.dropna())),
                  dict(variable="Female (proportion)", smd=smd((cv.sex.dropna() == "Female").astype(float), (nc.sex.dropna() == "Female").astype(float)))])
S.to_csv(OUT / "standardised_mean_differences.csv", index=False)
print(T.to_string()); print(S.round(3).to_string())