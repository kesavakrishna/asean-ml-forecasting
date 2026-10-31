"""
Download the Yahoo Finance data into Dataset/.

Run:  python download_data.py

The end dates are fixed so every run gets the same data. The Philippines (PSEi) and
Thailand (SET) aren't on Yahoo; download those from Investing.com (see README).
"""
import warnings

import yfinance as yf

from data import DATA_DIR, FILES, START, US_FILE

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


def download(ticker, file_name, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=False,
                     progress=False, multi_level_index=False)
    # Yahoo stores prices as float32, so round back to the published 2 decimals.
    df = df[["Open", "High", "Low", "Close"]].round(2)
    df.to_csv(DATA_DIR / file_name, date_format="%Y-%m-%d")
    print(f"{ticker:6} {len(df)} days, {df.index.min().date()} to {df.index.max().date()} -> {file_name}")


if __name__ == "__main__":
    for market, ticker in TICKERS.items():
        download(ticker, FILES[market], START, END)
    download(US_TICKER, US_FILE, US_START, US_END)
