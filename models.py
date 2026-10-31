
"""Logistic regression in sklearn and PyTorch, run through the walk-forward folds."""
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, matthews_corrcoef, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from evaluation import majority_direction, mcnemar, walk_forward_splits
from features import make_dataset, make_features, up_close_to_close


# ---------- the two models ----------
# Each fit function trains on (X, y) and returns a function that gives P(up) for new rows.

def fit_sklearn(X, y, C):
    """Scaler + logistic regression. The scaler is fitted on these training rows only."""
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C))
    model.fit(X, y)
    return lambda X_new: model.predict_proba(X_new)[:, 1]


def fit_torch(X, y, weight_decay, epochs=500, lr=0.05):
    """The same model in PyTorch: one linear layer, trained with our own loop."""
    mean, std = X.mean(), X.std(ddof=0)  # scaling fitted on these training rows only

    def to_tensor(frame):
        return torch.tensor(((frame - mean) / std).to_numpy(), dtype=torch.float32)

    x = to_tensor(X)
    target = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(0)
    model = nn.Linear(X.shape[1], 1)  # 6 features -> 1 logit
    optimizer = torch.optim.Adam([
        {"params": model.weight, "weight_decay": weight_decay},
        {"params": model.bias, "weight_decay": 0.0},  # like sklearn: the intercept isn't penalised
    ], lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for _ in range(epochs):  # full batch: the whole training set every step
        optimizer.zero_grad()
        loss = loss_fn(model(x), target)
        loss.backward()
        optimizer.step()

    def predict(X_new):
        with torch.no_grad():
            return torch.sigmoid(model(to_tensor(X_new))).squeeze(1).numpy()

    return predict


# ---------- the six-fold loop ----------

def pick_by_log_loss(fit, grid, X_train, y_train, X_tune, y_tune):
    """Try every setting in the grid; keep the one with the lowest log loss on the tune year."""
    def tune_loss(setting):
        predict = fit(X_train, y_train, setting)
        return log_loss(y_tune, predict(X_tune), labels=[False, True])
    return min(grid, key=tune_loss)


def walk_forward(df, fit, grid, make_x=make_features, make_y=up_close_to_close, pick=pick_by_log_loss):
    """For each test year: pick a setting on the tune year, refit on train + tune with it,
    and predict the test year."""
    X, y = make_dataset(df, make_x, make_y)
    up_yesterday = make_y(df).shift(1).eq(True)  # yesterday's label; unknown counts as down
    folds = []
    for year, train, tune, test in walk_forward_splits(X.index):
        best = pick(fit, grid, X[train], y[train], X[tune], y[tune])

        before_test = train | tune
        predict = fit(X[before_test], y[before_test], best)
        folds.append(pd.DataFrame({
            "year": year,
            "setting": " ".join(str(best).split()),  # as one line of text
            "p_up": predict(X[test]),
            "up": y[test],
            "fair_call": majority_direction(y[before_test]),  # fixed before the test year
            "same_as_yesterday": up_yesterday[X.index[test]],
        }, index=X.index[test]))
    return pd.concat(folds)


# ---------- scoring ----------

def report(preds):
    """Per test year: the model's scores, both bars, and paired tests against each bar."""
    rows = []
    for year, f in preds.groupby("year"):
        up, calls = f["up"], f["p_up"] > 0.5
        hindsight = {  # the stretch bar: best guess, chosen by looking at the test year
            "always up": pd.Series(True, index=f.index),
            "always down": pd.Series(False, index=f.index),
            "same as yesterday": f["same_as_yesterday"],
        }
        stretch = max(hindsight, key=lambda g: (hindsight[g] == up).mean())
        rows.append({
            "year": year,
            "setting": f["setting"].iloc[0],
            "accuracy %": 100 * (calls == up).mean(),
            "MCC": matthews_corrcoef(up, calls),
            "AUC": roc_auc_score(up, f["p_up"]),
            "log loss": log_loss(up, f["p_up"], labels=[False, True]),
            "fair bar %": 100 * (f["fair_call"] == up).mean(),
            "p vs fair": mcnemar(up, calls, f["fair_call"])[2],
            "stretch bar %": 100 * (hindsight[stretch] == up).mean(),
            "stretch guess": stretch,
            "p vs stretch": mcnemar(up, calls, hindsight[stretch])[2],
        })
    return pd.DataFrame(rows).set_index("year")


def pooled(preds):
    """All six test years of one market together: model vs fair bar vs same as yesterday."""
    up, calls = preds["up"], preds["p_up"] > 0.5
    return {
        "accuracy %": 100 * (calls == up).mean(),
        "MCC": matthews_corrcoef(up, calls),
        "AUC": roc_auc_score(up, preds["p_up"]),
        "fair bar %": 100 * (preds["fair_call"] == up).mean(),
        "p vs fair": mcnemar(up, calls, preds["fair_call"])[2],
        "same as yesterday %": 100 * (preds["same_as_yesterday"] == up).mean(),
        "p vs same as yesterday": mcnemar(up, calls, preds["same_as_yesterday"])[2],
    }
