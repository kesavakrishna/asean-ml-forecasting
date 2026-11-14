"""Made-up markets for checking the harness. Nothing about them is real.

- random: each day's move is independent of the past, so no rule can work.
- planted: each day tends to reverse the day before, so "reverse the 1-day move" is right about 60% of the time.
"""
import numpy as np
import pandas as pd

from data import clean

DAYS = pd.bdate_range("2010-01-01", "2023-03-16", name="Date")
NAMES = [f"fake {n}" for n in range(1, 7)]  # six made-up markets, like the six real ones
PULL_BACK = 0.3  # planted: today's move is -0.3 x yesterday's, plus noise


def make(kind, seed):
    """One made-up market of the given kind, cleaned like the real ones."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.01, len(DAYS))
    moves = noise.copy()
    if kind == "planted":
        for t in range(1, len(DAYS)):
            moves[t] = -PULL_BACK * moves[t - 1] + noise[t]
    close = 1000 * np.exp(moves.cumsum())
    open_ = np.roll(close, 1) * np.exp(rng.normal(0, 0.002, len(DAYS)))
    open_[0] = close[0]
    high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.008, len(DAYS)))
    low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.008, len(DAYS)))
    raw = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close}, index=DAYS).round(2)
    return clean(raw, "fake")


def markets(kind):
    """Six made-up markets of one kind, as {name: DataFrame}."""
    first_seed = {"random": 100, "planted": 200}[kind]
    return {name: make(kind, first_seed + number) for number, name in enumerate(NAMES)}
