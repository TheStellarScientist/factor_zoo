"""Robustness checks for the post-publication decay. Run after src/scan.py.

1. Calendar-time control (McLean & Pontiff 2016 style panel regression)
   ret[i,t] = a_i (+ m_t) + b1 * post_sample[i,t] + b2 * post_pub[i,t] + e[i,t]
   a_i = signal fixed effects; m_t = month fixed effects. With month fixed
   effects, published and not-yet-published signals are compared in the SAME
   months, so a market-wide trend toward weaker anomalies can't masquerade as
   a publication effect. Standard errors are clustered by month.

2. Who survives? Post-publication Bonferroni survivors vs. in-sample leaders,
   and annualized Sharpe ratios, which (unlike t-stats) don't grow with the
   length of the window. Old papers have long post-publication windows, so
   their t-stats get a mechanical boost.

Usage: python src/robustness.py [--vw]
Outputs (results/, or results/vw/ with --vw):
  decay_regressions.csv    coefficients, clustered SEs, t-stats
  post_pub_survivors.csv   predictors passing Bonferroni after publication
  in_sample_leaders.csv    top 12 predictors in-sample and what happened next
  sharpe_leaders.csv       top 10 by post-publication Sharpe ratio
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VW = "--vw" in sys.argv
RESULTS = ROOT / "results" / ("vw" if VW else "")
RET_FILE = "ls_returns_vw.csv" if VW else "ls_returns.csv"
MIN_LEG = 20      # same tradability screen as the survival analysis
N_ITER = 50       # alternating projections for two-way fixed effects


# ---------- 1. Calendar-time control ----------

def panel():
    """Monthly long-short returns from each signal's sample start, with period dummies."""
    stats_ = pd.read_csv(RESULTS / "signal_stats.csv")
    keep = stats_[(stats_.group == "predictor") & (stats_.n_leg >= MIN_LEG)
                  & stats_.is_mean.notna()]
    doc = pd.read_csv(DATA / "signal_doc.csv")[["Acronym", "SampleStartYear", "SampleEndYear", "Year"]]
    r = pd.read_csv(DATA / RET_FILE, parse_dates=["date"])
    r = r[r.signalname.isin(keep.signal)].merge(doc, left_on="signalname", right_on="Acronym")
    y = r.date.dt.year
    r = r[y >= r.SampleStartYear].copy()
    y = r.date.dt.year
    r["post_sample"] = ((y > r.SampleEndYear) & (y <= r.Year)).astype(float)
    r["post_pub"] = (y > r.Year).astype(float)
    return r, keep.is_mean.mean()


def demean(df, cols, by):
    """Remove fixed effects by (alternating) group demeaning."""
    out = df[cols].astype(float).copy()
    for _ in range(N_ITER if len(by) > 1 else 1):
        for g in by:
            out = out - out.groupby(df[g]).transform("mean")
    return out


def fe_regression(r, by):
    cols = ["ret", "post_sample", "post_pub"]
    d = demean(r, cols, by)
    X, y = d[["post_sample", "post_pub"]].to_numpy(), d["ret"].to_numpy()
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    bread = np.linalg.inv(X.T @ X)
    scores = pd.DataFrame(X * e[:, None]).groupby(r["date"].to_numpy()).sum().to_numpy()
    se = np.sqrt(np.diag(bread @ scores.T @ scores @ bread))   # clustered by month
    return b, se


# ---------- 2. Who survives? ----------

def survivors():
    df = pd.read_csv(RESULTS / "signal_stats.csv")
    th = pd.read_csv(RESULTS / "thresholds.csv").set_index("threshold")
    bonf = th.loc["Bonferroni (0.05 / m)", "p_cutoff"]
    doc = pd.read_csv(DATA / "signal_doc.csv")[["Acronym", "Portfolio Period", "Authors"]]
    df = df[df.group == "predictor"].merge(doc[["Acronym", "Portfolio Period"]],
                                           left_on="signal", right_on="Acronym")
    df = df.rename(columns={"Portfolio Period": "holding_months"})
    # Annualized Sharpe ratio = monthly mean / monthly sd * sqrt(12) = t / sqrt(n) * sqrt(12)
    for k in ["is", "pp"]:
        df[f"{k}_sharpe"] = df[f"{k}_t"] / np.sqrt(df[f"{k}_n"]) * np.sqrt(12)
    ranked = df.dropna(subset=["is_t", "pp_t"]).copy()
    ranked["is_rank"] = ranked.is_t.rank(ascending=False).astype(int)
    ranked["pp_rank"] = ranked.pp_t.rank(ascending=False).astype(int)
    rho, rho_p = stats.spearmanr(ranked.is_t, ranked.pp_t)

    cols = ["signal", "description", "authors", "year", "category", "holding_months", "n_leg",
            "is_mean", "is_t", "is_rank", "is_sharpe", "pp_mean", "pp_t", "pp_rank", "pp_sharpe", "pp_n"]
    surv = ranked[ranked.pp_p <= bonf].sort_values("pp_t", ascending=False)[cols]
    leaders = ranked.sort_values("is_t", ascending=False).head(12)[cols]
    sharpe = ranked[ranked.n_leg >= MIN_LEG].sort_values("pp_sharpe", ascending=False).head(10)[cols]
    return surv, leaders, sharpe, rho, rho_p


def main():
    r, avg_is = panel()
    rows = []
    for label, by in [("signal FE", ["signalname"]), ("signal + month FE", ["signalname", "date"])]:
        b, se = fe_regression(r, by)
        for name, bi, si in zip(["post_sample", "post_pub"], b, se):
            rows.append(dict(model=label, term=name, coef_pct_per_month=bi, se=si, t=bi / si,
                             share_of_in_sample_mean=bi / avg_is))
    reg = pd.DataFrame(rows)
    reg.to_csv(RESULTS / "decay_regressions.csv", index=False)
    print(f"Average in-sample mean: {avg_is:.3f}%/month; {r.signalname.nunique()} signals, {len(r)} signal-months")
    print(reg.round(3).to_string(index=False))

    surv, leaders, sharpe, rho, rho_p = survivors()
    surv.to_csv(RESULTS / "post_pub_survivors.csv", index=False)
    leaders.to_csv(RESULTS / "in_sample_leaders.csv", index=False)
    sharpe.to_csv(RESULTS / "sharpe_leaders.csv", index=False)
    show = ["signal", "year", "holding_months", "is_t", "is_rank", "pp_t", "pp_rank", "pp_sharpe", "pp_n"]
    print(f"\nSpearman rank correlation, in-sample vs post-pub t: {rho:.2f} (p = {rho_p:.1e})")
    print("\nPost-publication Bonferroni survivors:\n", surv[show].round(2).to_string(index=False))
    print("\nIn-sample top 12:\n", leaders[show].round(2).to_string(index=False))
    print("\nTop 10 by post-publication Sharpe:\n", sharpe[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
