"""
Download the four Yahoo Finance markets into Dataset/.

Run:  python download_data.py

The end date is fixed so every run gets the same data. The Philippines (PSEi) and
Thailand (SET) aren't on Yahoo; download those from Investing.com (see README).
"""
import warnings
from pathlib import Path

import yfinance as yf

from data import DATA_DIR, FILES, START

warnings.filterwarnings("ignore", module="yfinance")  # pandas deprecation notices from inside yfinance

END = "2026-10-24"  # Yahoo's end date is exclusive: the last day downloaded is 2026-10-23

TICKERS = {
    "Indonesia": "^JKSE",  # IDX Composite
    "Malaysia":  "^KLSE",  # FTSE Bursa Malaysia KLCI
    "Singapore": "^STI",   # Straits Times Index
    "India":     "^NSEI",  # Nifty 50
}


def download(market, ticker):
    df = yf.download(ticker, start=START, end=END, auto_adjust=False,
                     progress=False, multi_level_index=False)
    # Yahoo stores prices as float32, so round back to the published 2 decimals.
    df = df[["Open", "High", "Low", "Close"]].round(2)
    path = Path(DATA_DIR) / FILES[market]
    df.to_csv(path, date_format="%Y-%m-%d")
    print(f"{market:10} {ticker:6} {len(df)} days, {df.index.min().date()} to {df.index.max().date()} -> {path.name}")


if __name__ == "__main__":
    for market, ticker in TICKERS.items():
        download(market, ticker)
