"""Step 7: predict how big each day's range will be (a number, not up/down)."""
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from classifiers import BATCH_SIZE, MAX_EPOCHS, PATIENCE
from evaluation import diebold_mariano, walk_forward_splits
from features import log_range, past_ranges
from gru import GRUNet


# ---------- the six-fold loop ----------

def make_vol_dataset(df, make_x):
    """Features X and label y (day t's log range), keeping only complete rows."""
    X, y = make_x(df), log_range(df)
    complete = X.notna().all(axis=1) & y.notna()
    return X[complete], y[complete]


def pick_by_mse(fit, settings, X_train, y_train, X_tune, y_tune):
    """Try every setting; keep the one with the lowest squared error on the tune year."""
    def tune_error(setting):
        predict = fit(X_train, y_train, setting)
        return mean_squared_error(y_tune, predict(X_tune))
    return min(settings, key=tune_error)


def walk_forward_vol(df, fit, settings, make_x, pick=pick_by_mse):
    """For each test year: pick a setting on the tune year, refit on train + tune, predict the test year."""
    X, y = make_vol_dataset(df, make_x)
    yesterday = past_ranges(df)
    folds = []
    for year, train, tune, test in walk_forward_splits(X.index):
        best = pick(fit, settings, X[train], y[train], X[tune], y[tune])
        before_test = train | tune
        predict = fit(X[before_test], y[before_test], best)
        folds.append(pd.DataFrame({
            "year": year,
            "setting": " ".join(str(best).split()),
            "predicted": predict(X[test]),
            "actual": y[test],
            "same_as_yesterday": yesterday[X.index[test]],
            "average_so_far": y[before_test].mean(),  # fixed before the test year
        }, index=X.index[test]))
    return pd.concat(folds)


# ---------- sklearn models ----------

def fit_regressor(X, y, model):
    """Scaler + a fresh copy of `model`, trained on (X, y). Returns a function that predicts."""
    pipeline = make_pipeline(StandardScaler(), clone(model))
    pipeline.fit(X, y)
    return pipeline.predict


# ---------- GRU ----------

def train_gru(X, y, hidden, epochs, X_tune=None, y_tune=None, seed=0):
    """Train a GRU on squared error. Returns predict() and the tune-year error after each epoch."""
    x_mean, x_std = X.mean(), X.std(ddof=0)  # scaling learned from the training rows only
    y_mean, y_std = y.mean(), y.std(ddof=0)

    def to_tensor(frame):
        return torch.tensor(((frame - x_mean) / x_std).to_numpy(), dtype=torch.float32)

    def predict(X_new):
        with torch.no_grad():
            return net(to_tensor(X_new)).squeeze(1).numpy() * y_std + y_mean

    x = to_tensor(X)
    target = torch.tensor(((y - y_mean) / y_std).to_numpy(), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(seed)
    net = GRUNet(X.shape[1], hidden)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    tune_errors = []
    for epoch in range(epochs):
        for batch in torch.randperm(len(x)).split(BATCH_SIZE):
            optimizer.zero_grad()
            loss_fn(net(x[batch]), target[batch]).backward()
            optimizer.step()

        if X_tune is not None:
            tune_errors.append(mean_squared_error(y_tune, predict(X_tune)))
            epochs_since_best = len(tune_errors) - 1 - np.argmin(tune_errors)
            if epochs_since_best == PATIENCE:
                break

    return predict, tune_errors


def fit_gru_vol(X, y, setting, seed=0):
    """Train with a fixed setting = (hidden size, epochs). Returns predict()."""
    hidden, epochs = setting
    predict, _ = train_gru(X, y, hidden, epochs, seed=seed)
    return predict


def pick_gru_vol_setting(fit, hidden_sizes, X_train, y_train, X_tune, y_tune, seed=0):
    """Early stopping on the tune year for each hidden size. Returns the best (hidden size, epochs)."""
    best_error = {}
    for hidden in hidden_sizes:
        _, errors = train_gru(X_train, y_train, hidden, MAX_EPOCHS, X_tune, y_tune, seed=seed)
        best_epochs = int(np.argmin(errors)) + 1
        best_error[(hidden, best_epochs)] = min(errors)
    return min(best_error, key=best_error.get)


# ---------- scoring ----------

def vol_scores(preds, baseline):
    """R2 against the average so far (1 = perfect, 0 = no better than it), and a paired test against `baseline`."""
    error = preds["predicted"] - preds["actual"]
    error_of_average = preds["average_so_far"] - preds["actual"]
    return {
        "R2": 1 - (error ** 2).sum() / (error_of_average ** 2).sum(),
        "RMSE": np.sqrt((error ** 2).mean()),
        "p vs baseline": diebold_mariano(error, baseline - preds["actual"]),
    }
