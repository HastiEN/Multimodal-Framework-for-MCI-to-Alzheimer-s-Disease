"""A3: cohort/follow-up/missingness tables and flow figure (uses recovered follow-up; read-only inputs)."""
import sys, pandas as pd, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
c = pd.read_csv("Claude_Code_Outputs/A1_followup_recovery/cohort_with_recovered_followup.csv")
V = "audit/verified_tables/"
# survival time: months_to_conversion for events, recovered follow-up for censored
c["surv_time"] = np.where(c.conversion_label == 1, c.months_to_conversion, c.followup_months_recovered)
c["in_survival"] = c.surv_time.notna()
t = c.groupby(["baseline_phase", "conversion_label"]).size().unstack(fill_value=0)
t.columns = ["no_observed_conversion", "converters"]; t["total"] = t.sum(axis=1); t.loc["All"] = t.sum()
t.to_csv(out / "table_cohort.csv")
rows = []
for lab, g in [("No observed conversion", c[c.conversion_label == 0]), ("Converters (time to AD)", c[c.conversion_label == 1])]:
    s = g.surv_time.dropna()
    rows.append(dict(group=lab, n=len(g), n_with_time=len(s), median=s.median(), q1=s.quantile(.25), q3=s.quantile(.75),
                     n_lt24=int((s < 24).sum()), n_lt36=int((s < 36).sum())))
pd.DataFrame(rows).to_csv(out / "table_followup.csv", index=False)
# phase-specific follow-up
c.groupby(["baseline_phase", "conversion_label"]).surv_time.describe().to_csv(out / "table_followup_by_phase.csv")
m = pd.read_csv(V + "feature_missingness_by_phase_and_outcome.csv")
ph = m.groupby(["baseline_phase", "feature"]).agg(n=("n_subjects", "sum"), miss=("n_missing", "sum")).reset_index()
ph["missing_percent"] = 100 * ph.miss / ph.n
ph.pivot(index="feature", columns="baseline_phase", values="missing_percent").round(1).to_csv(out / "table_missingness_phase.csv")
amy = ph[ph.feature.str.contains("SUVR", case=False) & (ph.baseline_phase == "ADNI1")]
amy.to_csv(out / "amyloid_adni1_missingness.csv", index=False)
# horizon cohorts
h = {}
for H in (24, 36, 60):
    pos = (c.conversion_label == 1) & (c.months_to_conversion <= H)
    neg = (c.conversion_label == 0) & (c.followup_months_recovered >= H) | ((c.conversion_label == 1) & (c.months_to_conversion > H))
    h[H] = (int(pos.sum()), int(neg.sum()))
pd.DataFrame([dict(horizon=k, positives=v[0], negatives=v[1], n=sum(v)) for k, v in h.items()]).to_csv(out / "table_horizon_cohorts_recovered.csv", index=False)
print(h)
# flow figure
ev = int(((c.conversion_label == 1) & c.surv_time.notna()).sum()); ce = int(((c.conversion_label == 0) & c.surv_time.notna()).sum())
boxes = [("1,255 labelled MCI participants", 0.9), ("5 exclusions\n1,250 analysed (415 converters / 835 no observed conversion)", 0.68),
         (f"Survival cohort: {ev+ce:,}\n({ev} events / {ce} censored)\n[14 follow-up dates recovered]", 0.44),
         (f"Fixed horizons (historical rules): 24 mo 907 | 36 mo 807 | 60 mo 640\nrecomputed here: " + " | ".join(f"{k} mo {sum(v)}" for k, v in h.items()), 0.2)]
fig, ax = plt.subplots(figsize=(7, 5)); ax.axis("off")
for i, (txt, y) in enumerate(boxes):
    ax.text(.5, y, txt, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round", fc="#eef3fa", ec="#345"))
    if i: ax.annotate("", xy=(.5, y + .08), xytext=(.5, boxes[i-1][1] - .08), arrowprops=dict(arrowstyle="->"))
fig.savefig(out / "fig_cohort_flow.png", dpi=200, bbox_inches="tight")
print(t); print(pd.read_csv(out / "table_followup.csv"))
