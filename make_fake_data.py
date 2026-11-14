"""
Write made-up price files into fake_data/, laid out like Dataset/.

Run:  python make_fake_data.py

The real data can't be shared, so the automated test on GitHub runs on these files.
They are random walks: fine for testing the code, meaningless as results.
To load them, set the DATA_DIR environment variable to fake_data.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from data import FILES, OTHERS_FILE, US_FILE
from features import CURRENCIES

FOLDER = Path(__file__).resolve().parent / "fake_data"
DAYS = pd.bdate_range("2010-01-01", "2023-03-16", name="Date")  # weekdays up to the end of the development period


def fake_prices(seed):
    """A random walk shaped like real days: Open near yesterday's Close, High on top, Low at the bottom.

    Calm and wild spells alternate, as in real markets. Without them GARCH has nothing to learn.
    """
    rng = np.random.default_rng(seed)
    spell = np.repeat(rng.choice([0.5, 1, 2], size=len(DAYS) // 20 + 1), 20)[:len(DAYS)]  # changes every 20 days
    close = 1000 * np.exp((rng.normal(0, 0.01, len(DAYS)) * spell).cumsum())
    open_ = np.roll(close, 1) * np.exp(rng.normal(0, 0.004, len(DAYS)) * spell)
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.01, len(DAYS)) * spell)
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.01, len(DAYS)) * spell)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=DAYS).round(2)


if __name__ == "__main__":
    FOLDER.mkdir(exist_ok=True)
    for seed, file_name in enumerate([*FILES.values(), US_FILE]):
        fake_prices(seed).to_csv(FOLDER / file_name, date_format="%Y-%m-%d")
    tickers = ["^VIX", *CURRENCIES.values(), "BZ=F"]  # step 9's other markets: closes only
    others = pd.DataFrame({t: fake_prices(100 + n)["Close"] for n, t in enumerate(tickers)})
    others.to_csv(FOLDER / OTHERS_FILE, date_format="%Y-%m-%d")
    print(f"wrote {len(FILES) + 2} made-up files to {FOLDER}")
