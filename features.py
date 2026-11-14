"""Features for the models. Row t only uses data up to day t-1."""
import numpy as np
import pandas as pd
from arch import arch_model

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
    return indicator_values(df).shift(1)  # move down one row: day t gets day t-1's values

def indicator_values(df):
    """Kim's 12 indicators on each day, using that day's own prices (so not yet shifted)."""
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
    return X

def make_us_features(df, us):
    """The S&P 500's return in the last US session before day t.

    The US session dated t ends after Asia has closed on day t, so day t can't use it.
    """
    us_ret = us["ret"].rename("us_ret")
    return pd.merge_asof(df[[]], us_ret, left_index=True, right_index=True,
                         allow_exact_matches=False)  # strictly earlier date

SEQ_DAYS = 20  # about one trading month

def make_sequence(df):
    """The last 20 daily returns, oldest first. The newest is day t-1."""
    X = pd.DataFrame(index=df.index)
    for lag in range(SEQ_DAYS, 0, -1):
        X[f"seq_ret{lag}"] = df["ret"].shift(lag)
    return X

def make_sequence_with_us(df, us):
    """Each of the 20 steps is one Asian day's return, then the US session that followed it.

    The newest step is day t-1's return and the last US session before day t.
    """
    us_ret = make_us_features(df, us)["us_ret"]
    X = pd.DataFrame(index=df.index)
    for lag in range(SEQ_DAYS, 0, -1):
        X[f"seq_ret{lag}"] = df["ret"].shift(lag)
        X[f"seq_us{lag}"] = us_ret.shift(lag - 1)
    return X

def log_range(df):
    """How big day t's move was: the log of log(High / Low). Blank if High = Low."""
    return np.log(np.log(df["High"] / df["Low"]).where(df["High"] > df["Low"]))

def past_ranges(df):
    """log_range for row t-1 and earlier. A blank day repeats the day before it."""
    return log_range(df).ffill().shift(1)

def make_har(df):
    """The average range over the last 1, 5 and 22 days. Row t ends at day t-1."""
    past = past_ranges(df)
    return pd.DataFrame({
        "range_1d": past,
        "range_5d": past.rolling(5).mean(),
        "range_22d": past.rolling(22).mean(),
    })

def make_har_with_us(df, us):
    """HAR plus the log size of the last US session's move. Moves under 0.1% count as 0.1%."""
    us_size = make_us_features(df, us)["us_ret"].abs().clip(lower=0.001)
    return make_har(df).assign(us_size=np.log(us_size))

def make_garch(df, test_year):
    """Log of GARCH(1,1)'s volatility forecast for day t, made at the end of day t-1.

    The GARCH settings are fitted only on days before test_year.
    """
    returns = 100 * df["ret"].fillna(0)  # arch works best with returns in %
    first_test_day = df.index[df.index.year == test_year][0]
    fitted = arch_model(returns, vol="GARCH", p=1, q=1).fit(last_obs=first_test_day, disp="off")
    variance = fitted.forecast(horizon=1, start=0, align="target", reindex=True).variance["h.1"]
    return pd.DataFrame({"garch": np.log(variance) / 2}, index=df.index)

def make_range_sequence(df):
    """The last 20 daily ranges, oldest first. The newest is day t-1."""
    past = past_ranges(df)
    X = pd.DataFrame(index=df.index)
    for lag in range(SEQ_DAYS, 0, -1):
        X[f"seq_range{lag}"] = past.shift(lag - 1)
    return X

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


# ---------- step 9: more inputs ----------

def rsi(close, n):
    """Relative strength index over n days, 0 to 100."""
    gain = close.diff().clip(lower=0)
    loss = (-close.diff()).clip(lower=0)
    return 100 - 100 / (1 + gain.rolling(n).mean() / loss.rolling(n).mean())

def wilder(s, n=14):
    """Wilder's smoothing, used by ATR and ADX: an average where each day counts 1/n."""
    return s.ewm(alpha=1 / n, adjust=False).mean()

def make_indicators_extended(df):
    """Set B: Kim's 12, plus standard extras. Row t uses prices up to day t-1."""
    close, high, low = df["Close"], df["High"], df["Low"]
    typical = (high + low + close) / 3
    true_range = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)

    X = pd.DataFrame(index=df.index)
    for n in (5, 20):  # Kim's 12 use 10 days
        highest, lowest = high.rolling(n).max(), low.rolling(n).min()
        X[f"rsi{n}"] = rsi(close, n)
        X[f"stoch_k{n}"] = 100 * (close - lowest) / (highest - lowest)
        X[f"williams_r{n}"] = 100 * (highest - close) / (highest - lowest)
        X[f"roc{n}"] = 100 * close / close.shift(n)
        X[f"cci{n}"] = (typical - typical.rolling(n).mean()) / (0.015 * mean_deviation(typical, n))
    X["disparity20"] = 100 * close / close.rolling(20).mean()  # Kim's 12 have 5 and 10

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    X["macd"], X["macd_signal"], X["macd_hist"] = macd / close, signal / close, (macd - signal) / close

    ma20, sd20 = close.rolling(20).mean(), close.rolling(20).std()
    X["bollinger_b"] = (close - (ma20 - 2 * sd20)) / (4 * sd20)  # 0 = lower band, 1 = upper band
    X["bollinger_width"] = 4 * sd20 / ma20

    atr = wilder(true_range)
    X["atr"] = atr / close
    up_move, down_move = high.diff(), -low.diff()
    plus_di = 100 * wilder(up_move.where((up_move > down_move) & (up_move > 0), 0.0)) / atr
    minus_di = 100 * wilder(down_move.where((down_move > up_move) & (down_move > 0), 0.0)) / atr
    X["plus_di"], X["minus_di"] = plus_di, minus_di
    X["adx"] = wilder(100 * (plus_di - minus_di).abs() / (plus_di + minus_di))

    return make_indicators(df).join(X.shift(1))  # move down one row: day t gets day t-1's values

def make_tsfresh(df, days=None):
    """Set C: tsfresh's features of the last 20 daily returns before day t (all, or only `days`).

    Gaps are filled with the median of earlier days only, or 0 if there is none. Slow: a few minutes per market.
    """
    from tsfresh import extract_features  # slow to import, so only when needed
    from tsfresh.feature_extraction import EfficientFCParameters

    windows = make_sequence(df).dropna()  # oldest first; the newest is day t-1
    if days is not None:
        windows = windows.loc[windows.index.intersection(days)]
    long = pd.DataFrame({
        "id": np.repeat(np.arange(len(windows)), SEQ_DAYS),
        "time": np.tile(np.arange(SEQ_DAYS), len(windows)),
        "ret": windows.to_numpy().ravel(),
    })
    X = extract_features(long, column_id="id", column_sort="time", default_fc_parameters=EfficientFCParameters(),
                         disable_progressbar=True, n_jobs=4 if days is None else 0).sort_index()
    X.index = windows.index
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(X.expanding().median().shift(1)).fillna(0).reindex(df.index)

CURRENCIES = {"Indonesia": "IDR=X", "Malaysia": "MYR=X", "Philippines": "PHP=X",
              "Singapore": "SGD=X", "Thailand": "THB=X", "India": "INR=X"}  # local currency per US dollar

def last_move_before(df, moves):
    """For each day t: the series' last daily move dated strictly before t (at most a week old).

    Blank if that move is blank (a long gap, or a bad print).
    """
    moves = moves.rename("move")
    joined = pd.merge_asof(df[[]], moves, left_index=True, right_index=True,
                           allow_exact_matches=False, tolerance=pd.Timedelta(days=7))
    return joined["move"]

def make_other_markets(df, us, others, market):
    """Set D: Kim's 12, plus the last moves before day t of the S&P 500, VIX, the currency and Brent oil."""
    X = make_indicators(df)
    X["us_ret"] = last_move_before(df, us["ret"])
    X["vix_move"] = last_move_before(df, others["^VIX"])
    X["fx_move"] = last_move_before(df, others[CURRENCIES[market]])
    X["oil_move"] = last_move_before(df, others["BZ=F"])
    return X
