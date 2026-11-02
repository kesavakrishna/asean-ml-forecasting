"""Does the GRU rediscover HAR? Look inside its memory, on the development years only."""
from ast import literal_eval

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from evaluation import walk_forward_splits
from features import make_range_sequence
from volatility import make_vol_dataset, train_gru

WINDOWS = [1, 5, 20]  # the GRU only reads 20 days, so 20 stands in for HAR's 22
CONTROL_DAY = 15      # a single old day: something HAR has no special use for


def recent_averages(X):
    """The average range over the last 1, 5 and 20 days, from the GRU's own inputs."""
    return pd.DataFrame({f"avg_{n}": X[[f"seq_range{lag}" for lag in range(1, n + 1)]].mean(axis=1) for n in WINDOWS})


def probe(memory_fit, target_fit, memory_test):
    """A straight line from memory to a target, fitted on earlier rows. Returns its forecasts for the test rows."""
    return LinearRegression().fit(memory_fit, target_fit).predict(memory_test)


def erase(memory, target, memory_fit, target_fit):
    """Take out of the memory whatever moves in step with the target. The slopes come from earlier rows."""
    slopes = LinearRegression().fit(target_fit, memory_fit)
    return memory - slopes.predict(target) + memory_fit.mean(axis=0)


def probe_one_fold(X, y, year, before_test, test, setting, seed):
    """Everything for one test year: the probes, the imitation line, and the forecasts after erasing."""
    hidden, epochs = setting
    trained, _ = train_gru(X[before_test], y[before_test], hidden, epochs, seed=seed)
    untrained, _ = train_gru(X[before_test], y[before_test], hidden, 0, seed=seed)  # 0 epochs: random weights

    memory_fit, memory_test = trained.memory(X[before_test]), trained.memory(X[test])
    random_fit, random_test = untrained.memory(X[before_test]), untrained.memory(X[test])

    averages = recent_averages(X)
    to_erase = {column: [column] for column in averages} | {"all": list(averages), "control": [f"day_{CONTROL_DAY}"]}
    averages[f"day_{CONTROL_DAY}"] = X[f"seq_range{CONTROL_DAY}"]

    fold = pd.DataFrame({
        "year": year,
        "actual": y[test],
        "gru": trained.predict(X[test]),
        "average_so_far": y[before_test].mean(),
    }, index=X.index[test]).join(averages[test])

    for column in [f"avg_{n}" for n in WINDOWS]:
        fold[f"trained_{column}"] = probe(memory_fit, averages.loc[before_test, column], memory_test)
        fold[f"random_{column}"] = probe(random_fit, averages.loc[before_test, column], random_test)

    for name, columns in to_erase.items():
        target_fit, target_test = averages.loc[before_test, columns], averages.loc[test, columns]
        erased_fit = erase(memory_fit, target_fit, memory_fit, target_fit)
        erased_test = erase(memory_test, target_test, memory_fit, target_fit)
        fold[f"forecast_without_{name}"] = trained.from_memory(erased_test)
        # Check that erasing worked: a new probe on the erased memory should find almost nothing.
        fold[f"readable_after_{name}"] = probe(erased_fit, target_fit[columns[0]], erased_test)

    # The closest a straight line on the three averages can get to the GRU's own forecasts.
    window_columns = [f"avg_{n}" for n in WINDOWS]
    imitation = LinearRegression().fit(averages.loc[before_test, window_columns], trained.predict(X[before_test]))
    fold["imitation"] = imitation.predict(averages.loc[test, window_columns])

    # An untrained GRU with only its last line fitted to the real target.
    fold["random_gru_forecast"] = probe(random_fit, y[before_test], random_test)
    return fold


def probe_market(df, saved_gru, seed):
    """Refit the GRU on each development fold with its saved setting, and check it reproduces the saved forecasts."""
    X, y = make_vol_dataset(df, make_range_sequence)
    folds = []
    for year, train, tune, test in walk_forward_splits(X.index):
        saved = saved_gru[saved_gru["year"] == year]
        setting = literal_eval(saved["setting"].iloc[0])
        fold = probe_one_fold(X, y, year, train | tune, test, setting, seed)
        assert np.allclose(fold["gru"], saved["predicted"], atol=1e-4), f"{year}: refit doesn't match the saved GRU"
        folds.append(fold)
    return pd.concat(folds)


def r2(guess, actual):
    """Plain R2: 1 is perfect, 0 is no better than the overall mean of `actual`."""
    return 1 - ((actual - guess) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


def r2_vs_average(guess, preds):
    """R2 against the average so far, the same score as step 7."""
    return 1 - ((preds["actual"] - guess) ** 2).sum() / ((preds["actual"] - preds["average_so_far"]) ** 2).sum()
