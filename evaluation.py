"""
Walk-forward splits, scoring and no-model baselines for up/down predictions.

Convention for every model: the row for day t has
    label    = did day t close up?  (ret > 0)
    features = built only from data up to day t-1
so splitting rows by calendar year never lets a label cross into another split.
"""
import math
import numpy as np
import pandas as pd

TEST_YEARS = range(2017, 2023)


def walk_forward_splits(dates, test_years=TEST_YEARS):
    """Yield (test_year, train, tune, test) boolean masks over `dates`.

    For test year Y: train on every year up to Y-2, tune on Y-1, test on Y.
    """
    years = dates.year
    for y in test_years:
        yield y, years <= y - 2, years == y - 1, years == y


def score(actual, predicted):
    """Accuracy, balanced accuracy and MCC for up/down guesses (True = up)."""
    tp = (actual & predicted).sum()    # up days called up
    tn = (~actual & ~predicted).sum()  # down days called down
    fp = (~actual & predicted).sum()   # down days called up
    fn = (actual & ~predicted).sum()   # up days called down
    mcc_denominator = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {
        "days": len(actual),
        "accuracy %": 100 * (tp + tn) / len(actual),
        "balanced acc %": 100 * (tp / (tp + fn) + tn / (tn + fp)) / 2,
        "MCC": (tp * tn - fp * fn) / mcc_denominator if mcc_denominator else 0.0,
    }


def baseline_guesses(df):
    """No-model guesses for each day's direction (True = up). Each uses only earlier days."""
    up_yesterday = df["ret"].shift(1) > 0
    return pd.DataFrame(
        {"always up": True, "always down": False, "same as yesterday": up_yesterday},
        index=df.index,
    )

def majority_direction(up):
    """The fair baseline: the direction that was more common in training (True = up).

    Fixed before the test year starts, so a model could have used it in real time.
    """
    return bool(up.mean() >= 0.5)

def money_test(day_return, calls_up, cost):
    """Buy at the open and sell at the close. Returns the growth of 1 unit (1.10 = +10%).

    model: trades only on the days it calls up. every_day: trades every day.
    day_return = Close / Open - 1 for each day. cost = round-trip cost (0.002 = 0.2%).
    """
    model = (1 + day_return[calls_up] - cost).prod()
    every_day = (1 + day_return - cost).prod()
    return model, every_day

def diebold_mariano(errors_a, errors_b, lags=5):
    """Paired test of two forecasts' squared errors on the same days. Returns a two-sided p-value.

    Nearby days are correlated, so the variance also counts neighbours up to `lags` days apart.
    """
    diff = np.asarray(errors_a) ** 2 - np.asarray(errors_b) ** 2
    centred = diff - diff.mean()
    variance = (centred ** 2).mean()
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        variance += 2 * weight * (centred[lag:] * centred[:-lag]).mean()
    t = diff.mean() / math.sqrt(variance / len(diff))
    return math.erfc(abs(t) / math.sqrt(2))

def mcnemar(actual, calls_a, calls_b):
    """Paired test of two sets of up/down calls scored on the same days.

    Only days where exactly one of them is right carry information.
    Returns (days only A got right, days only B got right, two-sided p-value).
    """
    a_right = calls_a == actual
    b_right = calls_b == actual
    only_a = int((a_right & ~b_right).sum())
    only_b = int((~a_right & b_right).sum())
    n = only_a + only_b
    # Exact binomial test: if A and B were equally good, each disagreement is a coin flip.
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return only_a, only_b, min(1.0, 2 * tail)