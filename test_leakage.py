"""
Leakage test: changing data after day t must not change anything computed for day t.

Run:  python test_leakage.py
"""
import numpy as np
import pandas as pd

from data import MARKETS, clean, read_market, read_us
from evaluation import baseline_guesses
from features import make_features, make_indicators, make_us_features

def pipeline(raw, market):
    """Everything computed per day so far. Add each new feature function here."""
    df = clean(raw, market)
    return df.join(baseline_guesses(df)).join(make_features(df)).join(make_indicators(df))


def model_inputs(raw, market):
    """Only the model features."""
    df = clean(raw, market)
    return make_features(df).join(make_indicators(df))


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


def assert_features_ignore_day_t(compute, raw, market, t, rng):
    """Features up to day t must not change when day t's own prices are scrambled too."""
    day_before = raw.index[raw.index.get_loc(t) - 1]
    expected = compute(raw, market).loc[:t]
    result = compute(scramble_after(raw, day_before, rng), market).loc[:t]
    pd.testing.assert_frame_equal(expected, result, obj=f"{market} features up to {t:%Y-%m-%d}, day t scrambled")


def test_no_lookahead():
    rng = np.random.default_rng(0)
    for market in MARKETS:
        raw = read_market(market)  # development period only, never the holdout
        for t in rng.choice(raw.index[50:-50], size=10, replace=False):
            assert_no_lookahead(pipeline, raw, market, pd.Timestamp(t), rng)


def test_features_ignore_day_t():
    rng = np.random.default_rng(1)
    for market in MARKETS:
        raw = read_market(market)
        for t in rng.choice(raw.index[50:-50], size=10, replace=False):
            assert_features_ignore_day_t(model_inputs, raw, market, pd.Timestamp(t), rng)


def test_catches_a_leak():
    """Sanity check: the test must fail on a feature scaled with full-sample statistics."""
    def leaky(raw, market):
        df = pipeline(raw, market)
        df["ret_z"] = (df["ret"] - df["ret"].mean()) / df["ret"].std()  # uses the future
        return df

    rng = np.random.default_rng(0)
    raw = read_market("Indonesia")
    try:
        assert_no_lookahead(leaky, raw, "Indonesia", pd.Timestamp("2018-06-29"), rng)
    except AssertionError:
        return
    raise AssertionError("leakage test missed a feature that uses future data")


def test_catches_a_same_day_leak():
    """Sanity check: the stricter test must fail on a feature that uses day t's own close."""
    def leaky(raw, market):
        X = model_inputs(raw, market)
        X["close_today"] = clean(raw, market)["Close"]  # uses day t itself
        return X

    rng = np.random.default_rng(0)
    raw = read_market("Indonesia")
    try:
        assert_features_ignore_day_t(leaky, raw, "Indonesia", pd.Timestamp("2018-06-29"), rng)
    except AssertionError:
        return
    raise AssertionError("leakage test missed a feature that uses day t's own price")


# ---------- the US feature (step 5) ----------

def assert_us_feature_ignores_us_day_t(make, asia, us_raw, t, rng):
    """The US feature for Asian days up to t must not change when US data from day t onward
    is removed or scrambled."""
    day_before = t - pd.Timedelta(days=1)
    expected = make(asia, clean(us_raw, "US")).loc[:t]
    for changed in [us_raw.loc[:day_before], scramble_after(us_raw, day_before, rng)]:
        result = make(asia, clean(changed, "US")).loc[:t]
        pd.testing.assert_frame_equal(expected, result, obj=f"US feature up to {t:%Y-%m-%d}")


def test_us_feature_ignores_us_day_t():
    rng = np.random.default_rng(2)
    us_raw = read_us()
    for market in MARKETS:
        asia = clean(read_market(market), market)[:"2022"]  # the US file ends in 2022
        for t in rng.choice(asia.index[50:-50], size=10, replace=False):
            assert_us_feature_ignores_us_day_t(make_us_features, asia, us_raw, pd.Timestamp(t), rng)


def test_catches_a_same_date_join():
    """Sanity check: joining the US session with the same date must fail the test."""
    def same_date(asia, us):
        return pd.merge_asof(asia[[]], us["ret"].rename("us_ret"), left_index=True, right_index=True)

    rng = np.random.default_rng(0)
    asia = clean(read_market("Indonesia"), "Indonesia")[:"2022"]
    try:
        assert_us_feature_ignores_us_day_t(same_date, asia, read_us(), pd.Timestamp("2018-06-29"), rng)
    except AssertionError:
        return
    raise AssertionError("leakage test missed a same-date join with US data")


if __name__ == "__main__":
    test_no_lookahead()
    test_features_ignore_day_t()
    test_catches_a_leak()
    test_catches_a_same_day_leak()
    test_us_feature_ignores_us_day_t()
    test_catches_a_same_date_join()
    print("Leakage test passed.")
