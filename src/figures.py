"""Figures for the factor zoo scan.

  results/manhattan.png         static Manhattan plot (in-sample vs post-publication), for the README
  results/survival.png          in-sample vs post-publication mean return, for the README
  results/manhattan.html        interactive versions (hover for signal, paper, year),
  results/survival.html         for the website

Run after src/scan.py.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

MIN_LEG = 20                  # survival analysis: drop strategies with <20 stocks in a leg (untradeable noise)
SURV_LIM = (-0.8, 3.2)        # survival plot range; points outside are marked at the edge
Y_CAP = 15                       # cap -log10(p) so a few huge signals don't flatten the rest
CHROM = ["#6250d6", "#a79ef0"]   # alternating category colors (violet, matches website Projects page)
CONTROL = "#8a8a86"              # negative controls: gray, hollow diamonds
INK, MUTED, GRID = "#1b1b1a", "#6b6b67", "#e6e5e1"
SURFACE = "#fcfcfb"
TIERS = [  # ordinal violet ramp, validated light->dark
    ("Fails naive (p ≥ 0.05)", "#a79ef0"),
    ("Naive only", "#8a7de6"),
    ("t > 3, not Bonferroni", "#6250d6"),
    ("Bonferroni", "#3d2f9e"),
]
LINE_STYLE = {"Naive (p < 0.05)": ":", "Benjamini-Hochberg FDR 5%": "-.",
              "Harvey-Liu-Zhu (t > 3)": "--", "Bonferroni (0.05 / m)": "-"}


def load():
    df = pd.read_csv(RESULTS / "signal_stats.csv")
    th = pd.read_csv(RESULTS / "thresholds.csv")
    # Order like a genome: categories ("chromosomes") by size, signals by publication year
    order = df[df.group == "predictor"]["category"].value_counts().index.tolist()
    df["cat_rank"] = df["category"].map({c: i for i, c in enumerate(order)})
    df = df.sort_values(["cat_rank", "year", "signal"]).reset_index(drop=True)
    df["x"] = np.arange(len(df)) + df["cat_rank"] * 6     # small gap between categories
    for k in ["is", "pp"]:
        df[f"{k}_y"] = np.minimum(-np.log10(df[f"{k}_p"]), Y_CAP)
    df["tier"] = np.select(
        [df.pass_bonf, df.pass_hlz, df.pass_naive],
        [TIERS[3][0], TIERS[2][0], TIERS[1][0]], TIERS[0][0])
    return df, th, order


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def manhattan_png(df, th, order):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, facecolor=SURFACE)
    panels = [("is", "In-sample (original paper's sample period)"),
              ("pp", "After publication (out-of-sample)")]
    pred, ctrl = df[df.group == "predictor"], df[df.group == "negative control"]
    for ax, (k, title) in zip(axes, panels):
        style(ax)
        for i, c in enumerate(order):
            sub = pred[pred.category == c]
            ax.scatter(sub.x, sub[f"{k}_y"], s=22, color=CHROM[i % 2],
                       edgecolor=SURFACE, linewidth=0.6, zorder=3)
        ax.scatter(ctrl.x, ctrl[f"{k}_y"], s=34, marker="D", facecolor="none",
                   edgecolor=CONTROL, linewidth=1.2, zorder=4)
        capped = df[df[f"{k}_y"] >= Y_CAP]
        ax.scatter(capped.x, capped[f"{k}_y"], marker="^", s=30, color=INK, zorder=5)
        for _, t in th.iterrows():
            ax.axhline(t.neg_log10_p, color=MUTED, linewidth=1, linestyle=LINE_STYLE[t.threshold], zorder=2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=6)
        ax.set_ylabel("−log₁₀(p)", color=MUTED, fontsize=10)
        ax.set_ylim(0, Y_CAP + 0.8)
    # One legend row under the title (threshold line styles + controls), instead of labels crowding the right edge
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=MUTED, linewidth=1, linestyle=LINE_STYLE[t])
               for t in LINE_STYLE]
    names = list(LINE_STYLE)
    handles.append(Line2D([], [], marker="D", linestyle="", markerfacecolor="none",
                          markeredgecolor=CONTROL, markersize=6))
    names.append("Negative controls")
    fig.legend(handles, names, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=5,
               frameon=False, fontsize=9, labelcolor=MUTED, handlelength=2.6, columnspacing=1.6)
    centers = df[df.group == "predictor"].groupby("category").x.median().reindex(order)
    axes[1].set_xticks(centers.values)
    axes[1].set_xticklabels(order, fontsize=9, color=MUTED)
    axes[1].tick_params(axis="x", length=0)
    fig.suptitle("A genome-wide scan of 212 published stock-return predictors",
                 x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"Each dot is one long-short strategy. Values above {Y_CAP} are capped (▲). "
             "Data: Chen & Zimmermann, Open Source Asset Pricing (release 2026.10).",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    fig.savefig(RESULTS / "manhattan.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def survival_png(df):
    pred = df[(df.group == "predictor") & (df.pp_n >= 36) & (df.n_leg >= MIN_LEG)]
    fig, ax = plt.subplots(figsize=(7.5, 6), facecolor=SURFACE)
    style(ax)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    lim = SURV_LIM
    ax.plot(lim, lim, color=MUTED, linewidth=1, linestyle="--", zorder=2)
    ax.axhline(0, color=MUTED, linewidth=0.8, zorder=2)
    ax.text(lim[1] - 0.6, lim[1] - 0.45, "no decay", ha="right", fontsize=8, color=MUTED)
    for name, color in TIERS:
        sub = pred[pred.tier == name]
        ax.scatter(sub.is_mean, sub.pp_mean, s=30, color=color, edgecolor=SURFACE,
                   linewidth=0.8, zorder=3, label=f"{name} (n={len(sub)})")
    off = pred[(pred.pp_mean > lim[1]) | (pred.is_mean > lim[1])]
    for _, r in off.iterrows():
        ax.scatter(min(r.is_mean, lim[1] - 0.05), lim[1] - 0.05, marker="^", s=40, color=INK, zorder=5)
        ax.annotate(f"{r.signal}: {r.pp_mean:.1f}%/mo post-pub, off chart\n"
                    f"(~{r.n_leg:.0f} stocks per leg)", (min(r.is_mean, lim[1]) , lim[1] - 0.05),
                    xytext=(-8, -6), textcoords="offset points", ha="right", va="top",
                    fontsize=8, color=MUTED)
    slope, intercept = np.polyfit(pred.is_mean, pred.pp_mean, 1)
    xs = np.array([0, pred.is_mean.max()])
    ax.plot(xs, intercept + slope * xs, color=INK, linewidth=2, zorder=4)
    ax.text(xs[1], intercept + slope * xs[1] + 0.08, f"fit: slope {slope:.2f}",
            ha="right", fontsize=9, color=INK)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("In-sample mean return (% per month)", color=MUTED)
    ax.set_ylabel("Post-publication mean return (% per month)", color=MUTED)
    ax.legend(title="In-sample significance", frameon=False, fontsize=8,
              title_fontsize=9, labelcolor=MUTED, loc="upper left")
    ax.set_title("Published predictors keep about 40% of their return", loc="left",
                 fontsize=12, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"{len(pred)} predictors with ≥36 post-publication months and ≥{MIN_LEG} stocks per leg.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(RESULTS / "survival.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def hover(df, k):
    return [
        f"<b>{r.signal}</b> · {r.description}<br>{r.authors} ({int(r.year)}), {r.journal}"
        f"<br>{r.category} · {r.group}<br>t = {r[f'{k}_t']:.2f}, mean = {r[f'{k}_mean']:.2f}%/mo, "
        f"{int(r[f'{k}_n'])} months"
        for _, r in df.iterrows()
    ]


def manhattan_html(df, th, order):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=("In-sample (original paper's sample period)",
                                        "After publication (out-of-sample)"))
    pred, ctrl = df[df.group == "predictor"], df[df.group == "negative control"]
    for row, k in [(1, "is"), (2, "pp")]:
        for i, c in enumerate(order):
            sub = pred[pred.category == c]
            fig.add_trace(go.Scatter(
                x=sub.x, y=sub[f"{k}_y"], mode="markers", name=c, legendgroup=c,
                showlegend=False, text=hover(sub, k), hoverinfo="text",
                marker=dict(size=8, color=CHROM[i % 2], line=dict(width=1, color=SURFACE))),
                row=row, col=1)
        fig.add_trace(go.Scatter(
            x=ctrl.x, y=ctrl[f"{k}_y"], mode="markers", name="Negative controls",
            legendgroup="ctrl", showlegend=(row == 1), text=hover(ctrl, k), hoverinfo="text",
            marker=dict(size=9, symbol="diamond-open", color=CONTROL, line=dict(width=1.5))),
            row=row, col=1)
        for _, t in th.iterrows():
            fig.add_hline(y=t.neg_log10_p, line=dict(color=MUTED, width=1,
                          dash={"-": "solid", "--": "dash", "-.": "dashdot", ":": "dot"}[LINE_STYLE[t.threshold]]),
                          annotation_text=t.threshold if row == 1 else None,
                          annotation_font=dict(size=10, color=MUTED), row=row, col=1)
        fig.update_yaxes(title_text="−log₁₀(p)", range=[0, Y_CAP + 0.8], gridcolor=GRID,
                         zeroline=False, row=row, col=1)
    centers = pred.groupby("category").x.median().reindex(order)
    fig.update_xaxes(tickvals=centers.values, ticktext=order, showgrid=False, row=2, col=1)
    fig.update_xaxes(showgrid=False, row=1, col=1)
    fig.update_layout(
        title=dict(text="A genome-wide scan of 212 published stock-return predictors", x=0.01),
        template="simple_white", plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK), height=720, hoverlabel=dict(bgcolor="white"),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"))
    fig.write_html(RESULTS / "manhattan.html", include_plotlyjs="cdn")


def survival_html(df):
    pred = df[(df.group == "predictor") & (df.pp_n >= 36) & (df.n_leg >= MIN_LEG)]
    fig = go.Figure()
    lim = list(SURV_LIM)  # hover still works on the off-chart point if the user zooms out
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="No decay",
                             line=dict(color=MUTED, dash="dash", width=1), hoverinfo="skip"))
    for name, color in TIERS:
        sub = pred[pred.tier == name]
        text = [f"<b>{r.signal}</b> · {r.description}<br>{r.authors} ({int(r.year)})"
                f"<br>in-sample {r.is_mean:.2f}%/mo (t={r.is_t:.1f})"
                f"<br>post-pub {r.pp_mean:.2f}%/mo (t={r.pp_t:.1f})" for _, r in sub.iterrows()]
        fig.add_trace(go.Scatter(x=sub.is_mean, y=sub.pp_mean, mode="markers",
                                 name=f"{name} (n={len(sub)})", text=text, hoverinfo="text",
                                 marker=dict(size=9, color=color, line=dict(width=1, color=SURFACE))))
    slope, intercept = np.polyfit(pred.is_mean, pred.pp_mean, 1)
    xs = [0, pred.is_mean.max()]
    fig.add_trace(go.Scatter(x=xs, y=[intercept + slope * x for x in xs], mode="lines",
                             name=f"Fit (slope {slope:.2f})", line=dict(color=INK, width=2),
                             hoverinfo="skip"))
    fig.update_layout(
        title=dict(text="Published predictors keep about 40% of their return", x=0.01),
        xaxis=dict(title="In-sample mean return (% per month)", range=lim, gridcolor=GRID),
        yaxis=dict(title="Post-publication mean return (% per month)", range=lim, gridcolor=GRID),
        template="simple_white", plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        height=600, legend=dict(title="In-sample significance"))
    fig.write_html(RESULTS / "survival.html", include_plotlyjs="cdn")


def main():
    df, th, order = load()
    manhattan_png(df, th, order)
    survival_png(df)
    manhattan_html(df, th, order)
    survival_html(df)
    print("Saved figures to", RESULTS)


if __name__ == "__main__":
    main()
