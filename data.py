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
    "Indonesia":   "Jakarta Stock Exchange Composite Index Historical Data.csv",
    "Malaysia":    "FTSE Malaysia KLCI Historical Data.csv",
    "Philippines": "FTSE Philippines Historical Data.csv",
    "Singapore":   "FTSE Straits Times Singapore Historical Data.csv",
    "Thailand":    "SET Thailand Index Historical Data.csv",
    "India":       "Nifty 50 Historical Data.csv",  # not ASEAN, kept as a benchmark
}
MARKETS = list(FILES)

WEEKEND_SESSIONS_OK = {"India"}  # real special sessions, e.g. Budget day, Diwali Muhurat trading
CLOSE_ONLY = {"Philippines"}     # Open/High/Low are copies of Close: don't build features from them
MAX_GAP_DAYS = 14                # a return spanning a longer gap is missing data, not a holiday


def read_raw(path):
    """Read one Investing.com export as-is: Date index, Open/High/Low/Close columns."""
    df = pd.read_csv(path, encoding="utf-8-sig", thousands=",")
    df = df.rename(columns={"Price": "Close"})
    # Dates come in two styles ("01-03-2012" and "03/15/2023"); both are month-first.
    df["Date"] = pd.to_datetime(df["Date"].str.replace("-", "/"), format="%m/%d/%Y")
    return df.set_index("Date").sort_index()[["Open", "High", "Low", "Close"]]


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


def load_market(market):
    """Cleaned, checked data for one market."""
    return clean(read_raw(DATA_DIR / FILES[market]), market)


def load_all():
    """Cleaned, checked data for every market, as {market: DataFrame}."""
    return {market: load_market(market) for market in MARKETS}
