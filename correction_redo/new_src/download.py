"""Download the data for the factor zoo scan.

Source: Open Source Asset Pricing (Chen & Zimmermann), data release 2026.10,
hosted on Hugging Face. Same files the `openassetpricing` package reads,
fetched directly with polars to avoid its heavy dependencies.

Saves four small CSVs to data/:
  signal_doc.csv      one row per signal: paper, year, sample period, category
  ls_returns.csv      monthly long-short returns, published predictors
  ls_placebo.csv      monthly long-short returns, placebos (negative controls)
  ls_returns_vw.csv   monthly long-short returns, value-weighted deciles

Run with --vw-only to fetch just the value-weighted file.

Usage (from the repo root):
    pip install --no-cache-dir polars
    python src/download.py
"""
import sys
from pathlib import Path
from urllib.parse import quote

import polars as pl

RELEASE = "Data-Release-2026.10"
BASE = f"https://huggingface.co/datasets/OpenSourceAP/CrossSection/resolve/{quote(RELEASE)}"
FILES = {
    "signal_doc": "SignalDoc.csv",
    "predictors": "Portfolios/Full Sets OP/PredictorPortsFull.parquet",
    "placebos": "Portfolios/Full Sets OP/PlaceboPortsFull.parquet",
    # Value-weighted decile portfolios (predictors only)
    "predictors_vw": "Portfolios/Full Sets Alt/PredictorAltPorts_DecilesVW.parquet",
}

DATA = Path(__file__).resolve().parents[1] / "data"
DATA.mkdir(exist_ok=True)


def url(name: str) -> str:
    return f"{BASE}/{quote(FILES[name])}"


def long_short(name: str) -> pl.DataFrame:
    """Read only the long-short rows of a portfolio file."""
    lf = pl.scan_parquet(url(name))
    cols = lf.collect_schema().names()
    print(f"{name} columns: {cols}")
    keep = [c for c in ["signalname", "date", "ret", "Nlong", "Nshort"] if c in cols]
    labels = lf.select(pl.col("port").cast(pl.Utf8).unique()).collect()["port"].to_list()
    print(f"{name} portfolio labels: {sorted(map(str, labels))}")
    return (
        lf.filter(pl.col("port").cast(pl.Utf8) == "LS")
        .select(keep)
        .collect()
        .sort(["signalname", "date"])
    )


def main():
    if "--vw-only" in sys.argv:   # just fetch the value-weighted file
        vw = long_short("predictors_vw")
        vw.write_csv(DATA / "ls_returns_vw.csv")
        print(f"ls_returns_vw.csv: {vw['signalname'].n_unique()} signals, {vw.height} rows")
        return

    doc = pl.read_csv(url("signal_doc"), infer_schema_length=1000, null_values="NA")
    doc.write_csv(DATA / "signal_doc.csv")
    print(f"signal_doc.csv: {doc.height} signals\ncolumns: {doc.columns}\n")

    ls = long_short("predictors")
    ls.write_csv(DATA / "ls_returns.csv")
    print(f"ls_returns.csv: {ls['signalname'].n_unique()} signals, {ls.height} rows\n")

    try:
        lsp = long_short("placebos")
        lsp.write_csv(DATA / "ls_placebo.csv")
        print(f"ls_placebo.csv: {lsp['signalname'].n_unique()} signals, {lsp.height} rows")
    except Exception as e:  # placebos are optional
        print("Placebo download skipped:", e)

    vw = long_short("predictors_vw")
    vw.write_csv(DATA / "ls_returns_vw.csv")
    print(f"ls_returns_vw.csv: {vw['signalname'].n_unique()} signals, {vw.height} rows")


if __name__ == "__main__":
    main()
