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

def make_dataset(df):
    """Features X and label y (did day t close up?), keeping only complete rows."""
    X = make_features(df)
    complete = X.notna().all(axis=1) & df["ret"].notna()
    return X[complete], (df["ret"] > 0)[complete]
