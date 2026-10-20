"""Compare attribution ranks across full-fit interpretation configurations."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from project_paths import processed_dir


def main():
    folder = processed_dir() / "shap_interpretation/primary_model_shap_continuation"
    frames = []
    for path in sorted(folder.glob("*/feature_contribution_ranking.csv")):
        frame = pd.read_csv(path)
        frame["config"] = path.parent.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("Run extended interpretation before comparing its ranks.")
    rankings = pd.concat(frames, ignore_index=True)
    if rankings.duplicated(["config", "feature"]).any():
        raise ValueError("Each configuration must rank a feature only once.")
    result = folder / "rank_consistency_analysis"
    result.mkdir(exist_ok=True)
    rankings.to_csv(result / "combined_rankings_long.csv", index=False)
    count = rankings.config.nunique()
    summaries = []
    for feature, group in rankings.groupby("feature"):
        summaries.append({"feature": feature, "configs_present": group.config.nunique(), "presence_percent": group.config.nunique() / count * 100, "top5_count": int((group["rank"] <= 5).sum()), "top10_count": int((group["rank"] <= 10).sum()), "top20_count": int((group["rank"] <= 20).sum()), "median_rank_when_present": group["rank"].median(), "mean_rank_when_present": group["rank"].mean()})
    pd.DataFrame(summaries).sort_values(["top20_count", "median_rank_when_present", "feature"], ascending=[False, True, True]).to_csv(result / "feature_rank_consistency_summary.csv", index=False)
    pairs = []
    for first, second in combinations(sorted(rankings.config.unique()), 2):
        a = rankings[rankings.config == first].set_index("feature")["rank"]
        b = rankings[rankings.config == second].set_index("feature")["rank"]
        shared = a.index.intersection(b.index)
        row = {"config_1": first, "config_2": second, "n_common_ranked_features": len(shared), "spearman_rank_rho_common_features": float(spearmanr(a.loc[shared], b.loc[shared]).statistic) if len(shared) > 1 else np.nan}
        for k in [10, 20]:
            top_a = set(a[a <= k].index)
            top_b = set(b[b <= k].index)
            row[f"top{k}_overlap_count"] = len(top_a & top_b)
            row[f"top{k}_jaccard"] = len(top_a & top_b) / len(top_a | top_b) if top_a | top_b else np.nan
        pairs.append(row)
    pd.DataFrame(pairs).to_csv(result / "pairwise_rank_overlap_and_correlation.csv", index=False)
    print(f"Compared {count} interpretation configurations; results: {result}")


if __name__ == "__main__":
    main()
