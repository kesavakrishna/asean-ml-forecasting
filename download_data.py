"""
Download the Yahoo Finance data into Dataset/.

Run:  python download_data.py          # the index files
      python download_data.py others   # VIX, currencies and Brent oil, for step 9

The end dates are fixed so every run gets the same data. The Philippines (PSEi) and
Thailand (SET) aren't on Yahoo; download those from Investing.com (see README).
"""
import warnings

import yfinance as yf

import sys

import pandas as pd

from data import DATA_DIR, FILES, OTHERS_FILE, START, US_FILE, US_HOLDOUT_FILE

warnings.filterwarnings("ignore", module="yfinance")  # pandas deprecation notices from inside yfinance

END = "2026-10-24"  # Yahoo's end date is exclusive: the last day downloaded is 2026-10-23

TICKERS = {
    "Indonesia": "^JKSE",  # IDX Composite
    "Malaysia":  "^KLSE",  # FTSE Bursa Malaysia KLCI
    "Singapore": "^STI",   # Straits Times Index
    "India":     "^NSEI",  # Nifty 50
}

US_TICKER = "^GSPC"                          # S&P 500
US_START, US_END = "2010-01-01", "2023-01-01"  # 2010-2022 only
US_HOLDOUT_START = US_END                      # kept in its own file, so the file above never changes

# Step 9: other markets, closes only. VIX, each market's currency per US dollar, and Brent oil.
OTHER_TICKERS = ["^VIX", "IDR=X", "MYR=X", "PHP=X", "SGD=X", "THB=X", "INR=X", "BZ=F"]


def download(ticker, file_name, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False, multi_level_index=False)
    # Yahoo stores prices as float32, so round back to the published 2 decimals.
    df = df[["Open", "High", "Low", "Close"]].round(2)
    df.to_csv(DATA_DIR / file_name, date_format="%Y-%m-%d")
    print(f"{ticker:6} {len(df)} days, {df.index.min().date()} to {df.index.max().date()} -> {file_name}")


def download_others():
    closes = {}
    for ticker in OTHER_TICKERS:
        df = yf.download(ticker, start=US_START, end=END, auto_adjust=False, progress=False, multi_level_index=False)
        closes[ticker] = df["Close"].astype(float).round(6)
        print(f"{ticker:6} {len(df)} days, {df.index.min().date()} to {df.index.max().date()}")
    pd.DataFrame(closes).rename_axis("Date").to_csv(DATA_DIR / OTHERS_FILE, date_format="%Y-%m-%d")
    print("->", OTHERS_FILE)


if __name__ == "__main__" and "others" in sys.argv:
    download_others()
elif __name__ == "__main__":
    for market, ticker in TICKERS.items():
        download(ticker, FILES[market], START, END)
    download(US_TICKER, US_FILE, US_START, US_END)
    download(US_TICKER, US_HOLDOUT_FILE, US_HOLDOUT_START, END)
