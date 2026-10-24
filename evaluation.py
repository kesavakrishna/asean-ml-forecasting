"""
Walk-forward splits, scoring and no-model baselines for up/down predictions.

Convention for every model: the row for day t has
    label    = did day t close up?  (ret > 0)
    features = built only from data up to day t-1
so splitting rows by calendar year never lets a label cross into another split.
"""
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
