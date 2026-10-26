"""
Load and clean the index files.

Every later step gets its data from load_market() / load_all(), so the
cleaning rules live in one place.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "Dataset"

FILES = {
    "Indonesia":   "Jakarta Composite (Yahoo).csv",           # download_data.py
    "Malaysia":    "FTSE Bursa Malaysia KLCI (Yahoo).csv",    # download_data.py
    "Philippines": "PSEi Composite Historical Data.csv",      # Investing.com
    "Singapore":   "Straits Times Index (Yahoo).csv",         # download_data.py
    "Thailand":    "SET Index Historical Data.csv",           # Investing.com
    "India":       "Nifty 50 (Yahoo).csv",                    # download_data.py; not ASEAN, a benchmark
}
MARKETS = list(FILES)

START = "2012-01-01"          # the same start for every market
HOLDOUT_START = "2023-03-17"  # 2023-03-17 onward stays untouched until the final check

WEEKEND_SESSIONS_OK = {"India"}  # real special sessions, e.g. Budget day, Diwali Muhurat trading
MAX_GAP_DAYS = 14                # a return spanning a longer gap is missing data, not a holiday


def read_raw(path):
    """Read one file as-is: Date index, Open/High/Low/Close columns."""
    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
    if "Price" in df.columns:  # Investing.com export: "Price" is the close, dates are month-first
        df = df.rename(columns={"Price": "Close"})
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    else:                      # download_data.py output
        df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close"]]


def read_market(market, include_holdout=False):
    """Raw rows for one market from START, stopping before the holdout unless asked."""
    raw = read_raw(DATA_DIR / FILES[market])
    raw = raw[raw.index >= START]
    if not include_holdout:
        raw = raw[raw.index < HOLDOUT_START]
    return raw


def clean(raw, market):
    """Apply the cleaning rules, add the daily log return `ret`, and check the result."""
    df = raw[raw["Close"].diff() != 0].copy()  # unchanged close = holiday/weekend filler

    ohlc = df[["Open", "High", "Low", "Close"]]
    df["High"] = ohlc.max(axis=1)  # High can't be below Open or Close
    df["Low"] = ohlc.min(axis=1)   # Low can't be above Open or Close

    gap_days = df.index.to_series().diff().dt.days
    df["ret"] = np.log(df["Close"]).diff().where(gap_days <= MAX_GAP_DAYS)  # NaN across long gaps

    check(df, market)
    return df


def check(df, market):
    """Raise an error if the cleaned data breaks any rule."""
    prices = df[["Open", "High", "Low", "Close"]]
    gap_days = df.index.to_series().diff().dt.days
    problems = {
        "dates not sorted": not df.index.is_monotonic_increasing,
        "duplicate dates": df.index.duplicated().any(),
        "missing or non-positive prices": prices.isna().any().any() or (prices <= 0).any().any(),
        "High below Open/Low/Close": (df["High"] < prices.max(axis=1)).any(),
        "Low above Open/High/Close": (df["Low"] > prices.min(axis=1)).any(),
        "unchanged closes left": (df["Close"].diff() == 0).any(),
        "weekend rows": market not in WEEKEND_SESSIONS_OK and (df.index.dayofweek >= 5).any(),
        "return across a long gap": df["ret"][gap_days > MAX_GAP_DAYS].notna().any(),
    }
    failed = [problem for problem, found in problems.items() if found]
    if failed:
        raise ValueError(f"{market}: {', '.join(failed)}")


def load_market(market, include_holdout=False):
    """Cleaned, checked data for one market. Excludes the holdout unless asked."""
    return clean(read_market(market, include_holdout), market)


def load_all(include_holdout=False):
    """Cleaned, checked data for every market, as {market: DataFrame}."""
    return {market: load_market(market, include_holdout) for market in MARKETS}
