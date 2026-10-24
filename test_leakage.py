"""
Leakage test: changing data after day t must not change anything computed for day t.

Run:  python test_leakage.py
"""
import numpy as np
import pandas as pd

from data import DATA_DIR, FILES, MARKETS, clean, read_raw
from evaluation import baseline_guesses


def pipeline(raw, market):
    """Everything computed per day so far. Add each new feature function here."""
    df = clean(raw, market)
    return df.join(baseline_guesses(df))


def scramble_after(raw, t, rng):
    """Copy of `raw` with every row after day t multiplied by a random factor.

    Filler rows (same close as the day before) keep the previous row's factor, so they
    stay filler rows and the scrambled data still passes the cleaning checks.
    """
    factor = pd.Series(rng.uniform(0.5, 1.5, len(raw)), index=raw.index)
    factor[raw.index <= t] = 1.0
    factor[raw["Close"].diff() == 0] = np.nan
    return raw.mul(factor.ffill(), axis=0)


def assert_no_lookahead(compute, raw, market, t, rng):
    """Results up to day t must be identical with the future removed or scrambled."""
    expected = compute(raw, market).loc[:t]
    for change, changed_raw in {
        "future removed": raw.loc[:t],
        "future scrambled": scramble_after(raw, t, rng),
    }.items():
        result = compute(changed_raw, market).loc[:t]
        pd.testing.assert_frame_equal(expected, result, obj=f"{market} up to {t:%Y-%m-%d}, {change}")


def test_no_lookahead():
    rng = np.random.default_rng(0)
    for market in MARKETS:
        raw = read_raw(DATA_DIR / FILES[market])
        for t in rng.choice(raw.index[50:-50], size=10, replace=False):
            assert_no_lookahead(pipeline, raw, market, pd.Timestamp(t), rng)


def test_catches_a_leak():
    """Sanity check: the test must fail on a feature scaled with full-sample statistics."""
    def leaky(raw, market):
        df = pipeline(raw, market)
        df["ret_z"] = (df["ret"] - df["ret"].mean()) / df["ret"].std()  # uses the future
        return df

    rng = np.random.default_rng(0)
    raw = read_raw(DATA_DIR / FILES["Indonesia"])
    try:
        assert_no_lookahead(leaky, raw, "Indonesia", pd.Timestamp("2018-06-29"), rng)
    except AssertionError:
        return
    raise AssertionError("leakage test missed a feature that uses future data")


if __name__ == "__main__":
    test_no_lookahead()
    test_catches_a_leak()
    print("Leakage test passed: nothing up to day t depends on data after day t.")
