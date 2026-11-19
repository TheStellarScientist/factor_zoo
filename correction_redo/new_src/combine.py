"""Phase 2: combine published predictors out-of-sample (walk-forward).

Question: if an investor, at each point in time, combined only the signals
that had been published by then, how would different combination rules do?

No lookahead:
  - A signal becomes available in January of the year after its publication year.
  - "Bonferroni survivor" is decided with information available at that date:
    the signal's in-sample p-value (known at publication) vs 0.05 / m_t, where
    m_t is the number of predictors published by then.
  - Ridge weights are estimated each January from the previous 10 years only.

Strategies (weights over long-short portfolios, scaled so sum |w| = 1,
rebalanced monthly to target weights):
  ew_all        equal weight, all published predictors
  ew_bonf       equal weight, only signals that passed Bonferroni in-sample
  ridge_g{γ}    ridge mean-variance weights  w ∝ (Σ + λI)^-1 μ  on trailing
                10-year returns, λ = γ · trace(Σ)/N, for fixed γ = 0.1, 1, 10
  ridge_cv      γ chosen each January by time-series cross-validation inside the
                trailing window only (train on the first k years, validate on
                year k+1, for k = 5..9; pick the γ with the best pooled
                validation Sharpe), then refit on the full window

Out-of-sample period: January 1995 to December 2025.
Returns are gross (before trading costs); see src/costs.py for costs.

Usage:
  python src/combine.py          equal-weighted signal portfolios -> results/
  python src/combine.py --vw     value-weighted signal portfolios -> results/vw/

Outputs: combo_returns.csv, combo_stats.csv, combo_weights_log.csv,
         combo_turnover.csv (monthly sum |Δw| per strategy), combo_exposure.csv
         (monthly sum |w_i| / holding period, used by costs.py),
         combo_cumulative.png / .html
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VW = "--vw" in sys.argv
RESULTS = ROOT / "results" / ("vw" if VW else "")
RESULTS.mkdir(parents=True, exist_ok=True)
RET_FILE = "ls_returns_vw.csv" if VW else "ls_returns.csv"
LABEL = "value-weighted" if VW else "equal-weighted"

OOS_START = "1995-01-01"
WINDOW = 120          # months of history for ridge estimation
MIN_HIST = 60         # a signal needs this many months in the window to get a ridge weight
MIN_LEG = 20          # same tradability screen as before
GAMMAS = [0.1, 1.0, 10.0]
CV_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
CV_FOLDS = range(5, 10)   # train on first k years of the window, validate on year k+1
TARGET_VOL = 10.0     # % per year, for the plot only (Sharpe ratios are scale-free)

NAMES = {"ew_all": "Equal weight, all published", "ew_bonf": "Equal weight, Bonferroni survivors",
         "ridge_g0.1": "Ridge (γ = 0.1)", "ridge_g1.0": "Ridge (γ = 1)",
         "ridge_g10.0": "Ridge (γ = 10)", "ridge_cv": "Ridge, γ chosen by time-series CV"}
COLORS = {"ew_all": "#a79ef0", "ew_bonf": "#6250d6", "ridge_cv": "#1b1b1a"}
INK, MUTED, GRID, SURFACE = "#1b1b1a", "#6b6b67", "#e6e5e1", "#fcfcfb"


def load():
    stats = pd.read_csv(RESULTS / "signal_stats.csv")
    sig = stats[stats.group == "predictor"].set_index("signal")
    doc = pd.read_csv(DATA / "signal_doc.csv").set_index("Acronym")
    sig["holding_months"] = doc["Portfolio Period"].reindex(sig.index).fillna(1).clip(lower=1)
    r = pd.read_csv(DATA / RET_FILE, parse_dates=["date"])
    R = r[r.signalname.isin(sig.index)].pivot(index="date", columns="signalname", values="ret")
    R.index = R.index.to_period("M")
    counts = r[r.signalname.isin(sig.index)].copy()
    counts["date"] = counts.date.dt.to_period("M")
    counts["n_leg"] = counts[["Nlong", "Nshort"]].min(axis=1)
    N = counts.pivot(index="date", columns="signalname", values="n_leg").reindex(index=R.index, columns=R.columns)
    return R, sig, N


def ridge_weights(hist: pd.DataFrame, gamma: float) -> pd.Series:
    """w ∝ (Σ + λI)^-1 μ on the signals with enough history; scaled to sum |w| = 1."""
    h = hist.loc[:, hist.notna().sum() >= MIN_HIST]
    if h.empty:
        return pd.Series(dtype=float)
    h = h.fillna(h.mean())   # a few gaps: fill with the signal's own window mean
    mu, S = h.mean().to_numpy(), h.cov().to_numpy()
    lam = gamma * np.trace(S) / len(mu)
    w = np.linalg.solve(S + lam * np.eye(len(mu)), mu)
    return pd.Series(w / np.abs(w).sum(), index=h.columns)


def choose_gamma(hist: pd.DataFrame) -> float:
    """Pick γ using only the trailing window: expanding train, next-year validation."""
    years = 12
    scores = {}
    if len(hist) < 120 or hist.shape[1] == 0:
        return 1.0
    for g in CV_GRID:
        val = []
        for k in CV_FOLDS:
            train = hist.iloc[: k * years]
            test = hist.iloc[k * years: (k + 1) * years]
            w = ridge_weights(train, g)
            cols = [c for c in w.index if c in test.columns]
            if not cols:
                continue
            t = test[cols].fillna(0)
            val.append(t.to_numpy() @ w[cols].to_numpy())
        v = np.concatenate(val) if val else np.array([])
        scores[g] = v.mean() / v.std() if len(v) and v.std() > 0 else -np.inf
    return max(scores, key=scores.get)


def live(w: pd.Series, avail: list) -> pd.Series:
    """Restrict target weights to signals with a return this month; rescale to sum |w| = 1."""
    w = w[[s for s in w.index if s in avail]]
    return w / np.abs(w).sum() if len(w) else w


def walk_forward(R, sig, N):
    months = R.loc[OOS_START:].index
    rets = {k: [] for k in NAMES}
    turnover = {k: [] for k in NAMES}
    exposure = {k: [] for k in NAMES}      # sum |w_i| / holding_i: drives within-signal trading costs
    prev_w = {k: pd.Series(dtype=float) for k in NAMES}
    log, targets = [], {}
    for t in months:
        published = sig.index[sig.year < t.year]            # available from Jan of year after publication
        # Eligibility is decided from information at t-1, never the return/count at t.
        prior = N.loc[:t - 1].tail(12).median()
        avail = [s for s in published if s in R.columns and prior.get(s, np.nan) >= MIN_LEG]
        m_t = len(published)
        bonf = [s for s in avail if sig.at[s, "is_p"] <= 0.05 / m_t]

        if t.month == 1 or not targets:                     # re-estimate ridge each January
            hist = R.loc[t - WINDOW:t - 1, avail]
            targets = {f"ridge_g{g}": ridge_weights(hist, g) for g in GAMMAS}
            g_cv = choose_gamma(hist)
            targets["ridge_cv"] = ridge_weights(hist, g_cv)
            log.append(dict(date=str(t), n_available=len(avail), n_bonf=len(bonf),
                            n_ridge=len(targets["ridge_g1.0"]), gamma_cv=g_cv,
                            **{f"short_share_{k}": float((w < 0).mean()) for k, w in targets.items()}))

        weights = {"ew_all": pd.Series(1.0 / len(avail), index=avail) if avail else pd.Series(dtype=float),
                   "ew_bonf": pd.Series(1.0 / len(bonf), index=bonf) if bonf else pd.Series(dtype=float)}
        weights.update({k: live(w, avail) for k, w in targets.items()})

        # A missing realized return does not change precommitted weights.
        # Zero imputation is an explicit conservative convention, not an observed return.
        r_t = R.loc[t].fillna(0)
        for k, w in weights.items():
            rets[k].append(float((w * r_t[w.index]).sum()) if len(w) else np.nan)
            both = w.index.union(prev_w[k].index)
            turnover[k].append(float((w.reindex(both, fill_value=0)
                                      - prev_w[k].reindex(both, fill_value=0)).abs().sum()))
            exposure[k].append(float((w.abs() / sig.loc[w.index, "holding_months"]).sum()))
            prev_w[k] = w
    idx = months
    return (pd.DataFrame(rets, index=idx), pd.DataFrame(turnover, index=idx),
            pd.DataFrame(exposure, index=idx), pd.DataFrame(log))


def summarize(ret: pd.DataFrame) -> pd.DataFrame:
    rows = []
    periods = {"1995-2025": (None, None), "1995-2009": (None, "2009-12"), "2010-2025": ("2010-01", None)}
    for col in ret:
        for label, (a, b) in periods.items():
            x = ret[col].loc[a:b].dropna()
            sharpe = x.mean() / x.std() * np.sqrt(12)
            scaled = x * (TARGET_VOL / np.sqrt(12)) / x.std()      # same vol for drawdown comparison
            wealth = (1 + scaled / 100).cumprod()
            mdd = (wealth / wealth.cummax() - 1).min()
            rows.append(dict(strategy=col, name=NAMES[col], period=label, months=len(x),
                             mean_pct_month=x.mean(), vol_pct_year=x.std() * np.sqrt(12),
                             sharpe=sharpe, t_stat=x.mean() / x.std() * np.sqrt(len(x)),
                             max_drawdown_at_10pct_vol=mdd))
    return pd.DataFrame(rows)


def plot(ret: pd.DataFrame):
    show = ["ew_all", "ew_bonf", "ridge_cv"]
    scaled = ret[show].apply(lambda x: x * (TARGET_VOL / np.sqrt(12)) / x.std())
    wealth = (1 + scaled / 100).cumprod()
    x = wealth.index.to_timestamp()
    title = f"Combining published predictors out-of-sample, 1995–2025 ({LABEL} signals)"

    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    for k in show:
        ax.plot(x, wealth[k], color=COLORS[k], linewidth=2, label=NAMES[k])
        ax.text(x[-1], wealth[k].iloc[-1], f"  {NAMES[k].split(',')[0]}", va="center", fontsize=8, color=MUTED)
    ax.legend(frameon=False, fontsize=9, labelcolor=MUTED, loc="upper left")
    ax.set_ylabel("Growth of $1 (log scale, each scaled to 10% annual volatility)", color=MUTED, fontsize=9)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Walk-forward: only signals published by each date; ridge weights from the previous "
             "10 years. Gross of trading costs.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 0.88, 1))
    fig.savefig(RESULTS / "combo_cumulative.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)

    f = go.Figure()
    for k in show:
        f.add_trace(go.Scatter(x=x, y=wealth[k], mode="lines", name=NAMES[k],
                               line=dict(color=COLORS[k], width=2),
                               hovertemplate="%{x|%b %Y}: $%{y:.2f}<extra>" + NAMES[k] + "</extra>"))
    f.update_layout(title=dict(text=title, x=0.01),
                    yaxis=dict(type="log", title="Growth of $1 (each scaled to 10% annual vol)", gridcolor=GRID),
                    template="simple_white", plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
                    hovermode="x unified", height=520, legend=dict(x=0.01, y=0.99))
    f.write_html(RESULTS / "combo_cumulative.html", include_plotlyjs="cdn")


def main():
    R, sig, N = load()
    ret, turn, expo, wlog = walk_forward(R, sig, N)
    ret.to_csv(RESULTS / "combo_returns.csv")
    turn.to_csv(RESULTS / "combo_turnover.csv")
    expo.to_csv(RESULTS / "combo_exposure.csv")
    wlog.to_csv(RESULTS / "combo_weights_log.csv", index=False)
    stats = summarize(ret)
    stats.to_csv(RESULTS / "combo_stats.csv", index=False)
    plot(ret)
    print(f"[{LABEL}]")
    print(stats.pivot(index="name", columns="period", values="sharpe").round(2))
    print()
    print(stats[stats.period == "1995-2025"][["name", "mean_pct_month", "vol_pct_year", "sharpe",
                                               "max_drawdown_at_10pct_vol"]].round(3).to_string(index=False))
    print("\nγ chosen by CV, by year:", dict(zip(wlog.date.str[:4], wlog.gamma_cv)))
    print("Average monthly combination turnover (sum |Δw|):", turn.iloc[1:].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
