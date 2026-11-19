"""Breakeven trading-cost analysis. Run after src/combine.py (and src/robustness.py).

The data give portfolio returns, not trades, so true turnover is unknown.
Instead of claiming a cost, this asks: how large would one-way trading costs
have to be to wipe out each strategy, under explicit turnover assumptions?

Cost model (per month, in % of gross long-short exposure):

  cost_t = c × [ 2q · Σ_i |w_i,t| / H_i   +   2 · Σ_i |w_i,t − w_i,t−1| ]
                 └ trading inside each signal's portfolio ┘ └ reweighting the combination ┘

  c    one-way cost per dollar traded
  q    share of each leg replaced at a rebalance (unknown: scenarios 0.2, 0.5, 1.0)
  H_i  signal i's rebalancing period in months (from the signal documentation)
  2    each long-short unit has a long leg and a short leg

Breakeven cost c* sets the average net return to zero. Net Sharpe ratios do
not depend on leverage as long as costs scale with position size.

Usage: python src/costs.py [--vw]
Outputs: costs_breakeven.csv, costs_signals.csv, costs_net_sharpe.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VW = "--vw" in sys.argv
RESULTS = ROOT / "results" / ("vw" if VW else "")
LABEL = "value-weighted" if VW else "equal-weighted"

Q = {"low (q = 0.2)": 0.2, "medium (q = 0.5)": 0.5, "high (q = 1.0)": 1.0}
COSTS_BPS = np.arange(0, 101, 1)
SHOW = {"ew_all": "Equal weight, all published", "ew_bonf": "Equal weight, Bonferroni survivors",
        "ridge_cv": "Ridge, γ chosen by CV"}
COLORS = {"ew_all": "#a79ef0", "ew_bonf": "#6250d6", "ridge_cv": "#1b1b1a"}
INK, MUTED, GRID, SURFACE = "#1b1b1a", "#6b6b67", "#e6e5e1", "#fcfcfb"


def net_returns(ret, expo, turn, c_bps, q):
    c = c_bps / 1e4 * 100          # bps -> % (returns are in %)
    return ret - c * (2 * q * expo + 2 * turn)


def combos():
    ret = pd.read_csv(RESULTS / "combo_returns.csv", index_col=0)
    expo = pd.read_csv(RESULTS / "combo_exposure.csv", index_col=0)
    turn = pd.read_csv(RESULTS / "combo_turnover.csv", index_col=0)
    turn.iloc[0] = 0               # first month is the initial purchase, not ongoing turnover
    rows, curves = [], {}
    for k in ret:
        r, e, t = ret[k].dropna(), expo[k].loc[ret[k].dropna().index], turn[k].loc[ret[k].dropna().index]
        for qname, q in Q.items():
            per_bp = (2 * q * e + 2 * t).mean() / 100   # % of return lost per 1 bp of one-way cost
            breakeven = r.mean() / per_bp
            net = {c: net_returns(r, e, t, c, q) for c in COSTS_BPS}
            sharpe = {c: x.mean() / x.std() * np.sqrt(12) for c, x in net.items()}
            curves[(k, q)] = pd.Series(sharpe)
            rows.append(dict(strategy=k, turnover_scenario=qname, gross_mean_pct_month=r.mean(),
                             gross_sharpe=sharpe[0], net_sharpe_10bps=sharpe[10],
                             net_sharpe_25bps=sharpe[25], net_sharpe_50bps=sharpe[50],
                             breakeven_bps=breakeven,
                             monthly_traded_per_dollar=(2 * q * e + 2 * t).mean()))
    return pd.DataFrame(rows), curves


def signals():
    """Breakeven for each post-publication Bonferroni survivor (equal-weighted results)."""
    surv = pd.read_csv(ROOT / "results" / "post_pub_survivors.csv")
    H = surv["holding_months"].fillna(1).clip(lower=1)
    out = surv[["signal", "description", "holding_months", "pp_mean", "pp_sharpe"]].copy()
    for qname, q in Q.items():
        # pp_mean is % per month; breakeven (bps) = mean / monthly traded per $ × 100
        out[f"breakeven_bps_{qname.split()[0]}"] = out.pp_mean * 100 / (2 * q / H)
    return out


def plot(curves):
    fig, ax = plt.subplots(figsize=(8.5, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for k, name in SHOW.items():
        lo, mid, hi = curves[(k, 1.0)], curves[(k, 0.5)], curves[(k, 0.2)]
        ax.fill_between(mid.index, lo, hi, color=COLORS[k], alpha=0.15, linewidth=0)
        ax.plot(mid.index, mid, color=COLORS[k], linewidth=2, label=name)
    ax.set_xlabel("One-way trading cost (basis points)", color=MUTED)
    ax.set_ylabel("Net Sharpe ratio (annualized)", color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, 3.2)   # keep the zero-crossings readable; bands are clipped below -3
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper right")
    ax.set_title(f"How much trading cost can each combination absorb? ({LABEL} signals)",
                 loc="left", fontsize=11, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Lines: half of each leg replaced per rebalance (q = 0.5). "
             "Bands: q from 0.2 to 1.0. Turnover is assumed, not measured.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(RESULTS / "costs_net_sharpe.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    table, curves = combos()
    table.to_csv(RESULTS / "costs_breakeven.csv", index=False)
    plot(curves)
    print(f"[{LABEL}] breakeven one-way cost (bps) and net Sharpe at 25 bps")
    print(table.pivot(index="strategy", columns="turnover_scenario", values="breakeven_bps").round(1))
    print(table.pivot(index="strategy", columns="turnover_scenario", values="net_sharpe_25bps").round(2))
    if not VW:
        s = signals()
        s.to_csv(RESULTS / "costs_signals.csv", index=False)
        print("\nPost-publication survivors, breakeven one-way cost (bps):")
        print(s.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
