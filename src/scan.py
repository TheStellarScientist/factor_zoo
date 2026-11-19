"""Genome-wide scan of the factor zoo.

For every signal, test whether its monthly long-short return has a nonzero
mean in three periods (following McLean & Pontiff 2016):

  in-sample      the original paper's sample period (SampleStartYear..SampleEndYear)
  post-sample    after the sample ended, before publication
  post-pub       after the publication year (true out-of-sample)

Then apply multiple-testing thresholds to the in-sample p-values, the way a
genome-wide association study (GWAS) corrects for testing many variants.

Groups:
  predictors         212 signals the literature reports as predictive
  negative controls  14 placebos the original papers found NOT predictive
                     ("4_not" in Chen & Zimmermann). The other 100 placebos are
                     mostly variants of real predictors (e.g. quarterly versions),
                     so they are not clean negative controls and are excluded.

Usage: python src/scan.py [--vw]   (--vw: value-weighted portfolios, results/vw/)
Output: results/signal_stats.csv and results/thresholds.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
VW = "--vw" in sys.argv        # value-weighted predictors (no value-weighted placebos available)
RESULTS = ROOT / "results" / ("vw" if VW else "")
RESULTS.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05          # family-wise / FDR level
HLZ_T = 3.0           # Harvey, Liu & Zhu (2016) hurdle
MIN_MONTHS = 36       # minimum months to report a period's statistics


def load():
    doc = pd.read_csv(DATA / "signal_doc.csv")
    if VW:
        rets = pd.read_csv(DATA / "ls_returns_vw.csv")
    else:
        rets = pd.concat([
            pd.read_csv(DATA / "ls_returns.csv"),
            pd.read_csv(DATA / "ls_placebo.csv"),
        ])
    rets["date"] = pd.to_datetime(rets["date"])
    rets["year"] = rets["date"].dt.year

    # Keep predictors and the clean negative controls only
    doc = doc[(doc["Cat.Signal"] == "Predictor")
              | (doc["Predictability in OP"] == "4_not")].copy()
    doc["group"] = np.where(doc["Cat.Signal"] == "Predictor", "predictor", "negative control")
    rets = rets[rets["signalname"].isin(doc["Acronym"])]
    return doc, rets


def t_test(r: pd.Series) -> dict:
    """Mean monthly return (%), t-stat, two-sided p-value, number of months."""
    r = r.dropna()
    n = len(r)
    if n < MIN_MONTHS:
        return dict(mean=np.nan, t=np.nan, p=np.nan, n=n)
    se = r.std(ddof=1) / np.sqrt(n)
    t = r.mean() / se
    p = 2 * stats.t.sf(abs(t), df=n - 1)
    return dict(mean=r.mean(), t=t, p=p, n=n)


def period_stats(doc: pd.DataFrame, rets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    by_signal = dict(tuple(rets.groupby("signalname")))
    for _, s in doc.iterrows():
        r = by_signal.get(s["Acronym"])
        if r is None:
            continue
        y = r["year"]
        start, end, pub = s["SampleStartYear"], s["SampleEndYear"], s["Year"]
        periods = {
            "is": r.loc[(y >= start) & (y <= end), "ret"],
            "ps": r.loc[(y > end) & (y <= pub), "ret"],
            "pp": r.loc[y > pub, "ret"],   # publication year itself counted as pre-publication
        }
        row = dict(signal=s["Acronym"], group=s["group"], category=s["Cat.Data"],
                   year=pub, authors=s["Authors"], journal=s["Journal"],
                   description=s["LongDescription"], evidence=s["Predictability in OP"],
                   paper_t=s["T-Stat"],
                   # typical number of stocks in the smaller leg (tiny legs = noisy, hard to trade)
                   n_leg=(np.median(np.minimum(r["Nlong"], r["Nshort"]))
                          if {"Nlong", "Nshort"} <= set(r.columns) else np.nan))
        for k, series in periods.items():
            for stat, v in t_test(series).items():
                row[f"{k}_{stat}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


def bh_cutoff(p: np.ndarray, alpha: float = ALPHA) -> float:
    """Largest p-value rejected by Benjamini-Hochberg (the BH line on the plot)."""
    p = np.sort(p[~np.isnan(p)])
    m = len(p)
    passed = p <= alpha * np.arange(1, m + 1) / m
    return p[passed].max() if passed.any() else alpha / m


def main():
    doc, rets = load()
    df = period_stats(doc, rets)
    if VW:
        df = df[df.group == "predictor"]
        if df.n_leg.isna().all():   # stock counts missing in this file: borrow from equal-weighted run
            ew = pd.read_csv(ROOT / "results" / "signal_stats.csv").set_index("signal")["n_leg"]
            df["n_leg"] = df.signal.map(ew)

    # Thresholds are computed on the predictor family only (the "genome")
    pred = df[df["group"] == "predictor"]
    m = pred["is_p"].notna().sum()
    df_t = pred["is_n"].median() - 1
    thresholds = pd.DataFrame([
        ("Naive (p < 0.05)", ALPHA),
        ("Harvey-Liu-Zhu (t > 3)", 2 * stats.t.sf(HLZ_T, df=df_t)),
        ("Benjamini-Hochberg FDR 5%", bh_cutoff(pred["is_p"].to_numpy())),
        ("Bonferroni (0.05 / m)", ALPHA / m),
    ], columns=["threshold", "p_cutoff"])
    thresholds["neg_log10_p"] = -np.log10(thresholds["p_cutoff"])
    thresholds["m_tests"] = m

    for name, cut in zip(["naive", "hlz", "bh", "bonf"], thresholds["p_cutoff"]):
        df[f"pass_{name}"] = df["is_p"] <= cut

    df.to_csv(RESULTS / "signal_stats.csv", index=False)
    thresholds.to_csv(RESULTS / "thresholds.csv", index=False)

    # Quick summary
    print(thresholds.to_string(index=False))
    for g, sub in df.groupby("group"):
        print(f"\n{g}: {len(sub)} signals")
        for name in ["naive", "hlz", "bh", "bonf"]:
            print(f"  pass {name:5s}: {sub[f'pass_{name}'].sum()}")


if __name__ == "__main__":
    main()
