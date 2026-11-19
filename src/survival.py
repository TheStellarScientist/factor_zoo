"""Does passing a stricter in-sample threshold predict post-publication survival?

Groups predictors by the strictest threshold they pass in-sample and compares
their returns after publication. Run after src/scan.py.

Output: results/survival_by_tier.csv (and a printed summary)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results" / ("vw" if "--vw" in sys.argv else "")
MIN_MONTHS, MIN_LEG = 36, 20


def main():
    df = pd.read_csv(RESULTS / "signal_stats.csv")
    pred = df[(df.group == "predictor") & (df.pp_n >= MIN_MONTHS) & (df.n_leg >= MIN_LEG)].copy()
    pred["tier"] = np.select(
        [pred.pass_bonf, pred.pass_hlz, pred.pass_naive],
        ["1 Bonferroni", "2 t > 3, not Bonferroni", "3 naive only"], "4 fails naive")

    tiers = pred.groupby("tier").agg(
        n=("signal", "size"),
        in_sample=("is_mean", "mean"),
        post_sample=("ps_mean", "mean"),
        post_pub=("pp_mean", "mean"),
        still_significant=("pp_p", lambda p: (p < 0.05).mean()),
    )
    tiers["retention"] = tiers.post_pub / tiers.in_sample
    tiers.loc["all"] = [len(pred), pred.is_mean.mean(), pred.ps_mean.mean(), pred.pp_mean.mean(),
                        (pred.pp_p < 0.05).mean(), pred.pp_mean.mean() / pred.is_mean.mean()]
    tiers.to_csv(RESULTS / "survival_by_tier.csv")

    slope = np.polyfit(pred.is_mean, pred.pp_mean, 1)[0]
    ctrl = df[df.group == "negative control"]   # empty for value-weighted
    print(tiers.round(3).to_string())
    print(f"\nSlope of post-pub on in-sample mean: {slope:.2f}")
    print(f"Negative controls significant in-sample (p<0.05): {(ctrl.is_p < 0.05).sum()} of {len(ctrl)}; "
          f"t > 3: {(ctrl.is_t.abs() > 3).sum()}")


if __name__ == "__main__":
    main()
