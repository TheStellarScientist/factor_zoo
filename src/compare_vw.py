"""Equal-weighted vs value-weighted: do the headline results hold for big stocks?

Run after the full pipeline in both modes:
  python src/scan.py && python src/survival.py && python src/robustness.py && python src/combine.py && python src/costs.py
  python src/scan.py --vw && python src/survival.py --vw && python src/robustness.py --vw \
      && python src/combine.py --vw && python src/costs.py --vw

Output: results/compare_ew_vw.csv (and a printed table)
"""
from pathlib import Path

import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def summary(folder: Path) -> dict:
    sig = pd.read_csv(folder / "signal_stats.csv")
    pred = sig[sig.group == "predictor"]
    tiers = pd.read_csv(folder / "survival_by_tier.csv", index_col=0)
    reg = pd.read_csv(folder / "decay_regressions.csv")
    combo = pd.read_csv(folder / "combo_stats.csv")
    costs = pd.read_csv(folder / "costs_breakeven.csv")
    full = combo[combo.period == "1995-2025"].set_index("strategy")
    be = costs[costs.turnover_scenario == "medium (q = 0.5)"].set_index("strategy")
    fe = reg[(reg.model == "signal + month FE") & (reg.term == "post_pub")].iloc[0]
    both = pred.dropna(subset=["is_t", "pp_t"])
    bonf_row = tiers.loc[tiers.index.str.contains("Bonferroni") & ~tiers.index.str.contains("not")].iloc[0]
    return {
        "Predictors tested": len(pred),
        "Pass p < 0.05 in-sample": int(pred.pass_naive.sum()),
        "Pass Benjamini-Hochberg 5%": int(pred.pass_bh.sum()),
        "Pass t > 3 in-sample": int(pred.pass_hlz.sum()),
        "Pass Bonferroni in-sample": int(pred.pass_bonf.sum()),
        "Mean in-sample return (%/mo)": round(tiers.loc["all", "in_sample"], 2),
        "Mean post-publication return (%/mo)": round(tiers.loc["all", "post_pub"], 2),
        "Retention after publication": f"{tiers.loc['all', 'retention']:.0%}",
        "Still significant after publication": f"{tiers.loc['all', 'still_significant']:.0%}",
        "Bonferroni tier still significant": f"{bonf_row['still_significant']:.0%}",
        "Bonferroni survivors after publication": len(pd.read_csv(folder / "post_pub_survivors.csv")),
        "Post-pub decline with month FE (%/mo)": f"{fe.coef_pct_per_month:.2f} (t = {fe.t:.1f})",
        "Rank corr., in-sample vs post-pub t": round(stats.spearmanr(both.is_t, both.pp_t)[0], 2),
        "Sharpe, equal weight all (1995–2025)": round(full.loc["ew_all", "sharpe"], 2),
        "Sharpe, Bonferroni survivors": round(full.loc["ew_bonf", "sharpe"], 2),
        "Sharpe, ridge with CV": round(full.loc["ridge_cv", "sharpe"], 2),
        "Breakeven cost, equal weight all (bps, q = 0.5)": round(be.loc["ew_all", "breakeven_bps"]),
        "Breakeven cost, ridge with CV (bps, q = 0.5)": round(be.loc["ridge_cv", "breakeven_bps"]),
    }


def main():
    table = pd.DataFrame({"Equal-weighted": summary(ROOT / "results"),
                          "Value-weighted": summary(ROOT / "results" / "vw")})
    table.to_csv(ROOT / "results" / "compare_ew_vw.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()
