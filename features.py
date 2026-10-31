"""Features for the models. Row t only uses data up to day t-1."""
import numpy as np
import pandas as pd

N_LAGS = 5

def make_features(df):
    """The last 5 daily returns, plus yesterday's direction (+1 up, -1 down)."""
    X = pd.DataFrame(index=df.index)
    for lag in range(1, N_LAGS + 1):
        X[f"ret_lag{lag}"] = df["ret"].shift(lag)
    X["dir_yesterday"] = np.sign(df["ret"].shift(1))
    return X

N_DAYS = 10  # look-back days; the papers don't say, Patel (2015) uses 10

def mean_deviation(s, n):
    """Average distance from the mean, over each n-day window (used by CCI)."""
    return s.rolling(n).apply(lambda window: np.abs(window - window.mean()).mean(), raw=True)

def make_indicators(df):
    """The 12 technical indicators from Kim (2003). Row t uses prices up to day t-1."""
    close, high, low = df["Close"], df["High"], df["Low"]
    highest = high.rolling(N_DAYS).max()
    lowest = low.rolling(N_DAYS).min()
    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    typical = (high + low + close) / 3
    gain = close.diff().clip(lower=0)
    loss = (-close.diff()).clip(lower=0)

    X = pd.DataFrame(index=df.index)
    X["stoch_k"] = 100 * (close - lowest) / (highest - lowest)
    X["stoch_d"] = X["stoch_k"].rolling(3).mean()
    X["slow_d"] = X["stoch_d"].rolling(3).mean()
    X["momentum"] = close - close.shift(4)
    X["roc"] = 100 * close / close.shift(N_DAYS)
    X["williams_r"] = 100 * (highest - close) / (highest - lowest)
    X["ad_osc"] = (high - close.shift(1)) / (high - low).where(high > low)  # blank if High = Low
    X["disparity5"] = 100 * close / ma5
    X["disparity10"] = 100 * close / ma10
    X["oscp"] = (ma5 - ma10) / ma5
    X["cci"] = (typical - typical.rolling(N_DAYS).mean()) / (0.015 * mean_deviation(typical, N_DAYS))
    X["rsi"] = 100 - 100 / (1 + gain.rolling(N_DAYS).mean() / loss.rolling(N_DAYS).mean())
    return X.shift(1)  # move down one row: day t gets day t-1's values

def make_us_features(df, us):
    """The S&P 500's return in the last US session before day t.

    The US session dated t ends after Asia has closed on day t, so day t can't use it.
    """
    us_ret = us["ret"].rename("us_ret")
    return pd.merge_asof(df[[]], us_ret, left_index=True, right_index=True,
                         allow_exact_matches=False)  # strictly earlier date

def up_close_to_close(df):
    """Label: did day t close above day t-1's close?"""
    return df["ret"] > 0

def up_open_to_close(df):
    """Label: did day t close above its own open? (Buy at the open, sell at the close.)

    Blank when the open is just a copy of yesterday's close, i.e. not a real open.
    """
    real_open = df["Open"] != df["Close"].shift(1)
    return (df["Close"] > df["Open"]).where(real_open)

def make_dataset(df, make_x=make_features, make_y=up_close_to_close):
    """Features X and label y, keeping only complete rows."""
    X, y = make_x(df), make_y(df)
    complete = X.notna().all(axis=1) & y.notna() & df["ret"].notna()
    return X[complete], y[complete].astype(bool)
