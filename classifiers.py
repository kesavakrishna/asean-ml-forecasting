"""Step 4: the paper's classifiers (sklearn) and a small neural network (PyTorch)."""
import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch import nn

from models import pooled, report


# ---------- sklearn models ----------
# Each model has a short list of versions to try. The tune year picks one.

CLASSIFIERS = {
    "LDA": [LinearDiscriminantAnalysis()],
    "Logistic regression": [LogisticRegression(C=C) for C in [0.0001, 0.001, 0.01, 0.1, 1, 10]],
    "KNN": [KNeighborsClassifier(n_neighbors=k) for k in [5, 25, 101, 251, 501]],
    # RBF kernel, as in the paper. The calibrator turns the SVM's scores into P(up).
    "SVM": [CalibratedClassifierCV(SVC(C=C, gamma=g), ensemble=False) for C in [0.1, 1, 10] for g in [0.01, 0.1]],
    # 3 features per split, as in the paper
    "Random forest": [RandomForestClassifier(n_estimators=300, max_features=3, min_samples_leaf=leaf, random_state=0, n_jobs=-1)
                      for leaf in [1, 10, 50, 200]],
    "AdaBoost": [AdaBoostClassifier(n_estimators=n, learning_rate=lr, random_state=0) for n in [50, 200] for lr in [0.1, 1.0]],
}


def fit_classifier(X, y, model):
    """Scaler + a fresh copy of `model`, trained on (X, y). Returns a function that gives P(up)."""
    pipeline = make_pipeline(StandardScaler(), clone(model))
    pipeline.fit(X, y)
    return lambda X_new: pipeline.predict_proba(X_new)[:, 1]


# ---------- neural network ----------

HIDDEN_GRID = [2, 4, 6, 8, 10, 12]  # the paper's hidden-node counts
MAX_EPOCHS = 200
PATIENCE = 20                       # early stopping: stop after 20 epochs with no improvement
BATCH_SIZE = 64
torch.set_num_threads(1)            # faster for a network this small


def train_net(X, y, hidden, epochs, X_tune=None, y_tune=None):
    """Train a one-hidden-layer network in mini-batches.

    Returns predict() and the tune-year loss after each epoch.
    With tune data, it stops early once that loss stops improving.
    """
    mean, std = X.mean(), X.std(ddof=0)  # scaling learned from the training rows only

    def to_tensor(frame):
        return torch.tensor(((frame - mean) / std).to_numpy(), dtype=torch.float32)

    def predict(X_new):
        with torch.no_grad():
            return torch.sigmoid(net(to_tensor(X_new))).squeeze(1).numpy()

    x = to_tensor(X)
    target = torch.tensor(y.to_numpy(), dtype=torch.float32).unsqueeze(1)

    torch.manual_seed(0)
    net = nn.Sequential(
        nn.Linear(X.shape[1], hidden),  # 12 indicators -> hidden nodes
        nn.Tanh(),
        nn.Linear(hidden, 1),           # hidden nodes -> 1 logit
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss()

    tune_losses = []
    for epoch in range(epochs):
        for batch in torch.randperm(len(x)).split(BATCH_SIZE):  # shuffled mini-batches
            optimizer.zero_grad()
            loss_fn(net(x[batch]), target[batch]).backward()
            optimizer.step()

        if X_tune is not None:
            tune_losses.append(log_loss(y_tune, predict(X_tune), labels=[False, True]))
            epochs_since_best = len(tune_losses) - 1 - np.argmin(tune_losses)
            if epochs_since_best == PATIENCE:
                break

    return predict, tune_losses


def fit_net(X, y, setting):
    """Train with a fixed setting = (hidden nodes, epochs). Returns predict()."""
    hidden, epochs = setting
    predict, _ = train_net(X, y, hidden, epochs)
    return predict


def pick_net_setting(fit, hidden_grid, X_train, y_train, X_tune, y_tune):
    """Early stopping on the tune year for each hidden size. Returns the best (hidden nodes, epochs)."""
    best_loss = {}
    for hidden in hidden_grid:
        _, losses = train_net(X_train, y_train, hidden, MAX_EPOCHS, X_tune, y_tune)
        best_epochs = int(np.argmin(losses)) + 1
        best_loss[(hidden, best_epochs)] = min(losses)
    return min(best_loss, key=best_loss.get)


# ---------- scoring ----------

def all_up_folds(preds):
    """Number of test years in which the model said up on every day."""
    return int(preds.groupby("year")["p_up"].min().gt(0.5).sum())


def model_table(preds_by_market):
    """Step 3's pooled table for one model, plus how often it said up."""
    rows = {}
    for market, preds in preds_by_market.items():
        rows[market] = pooled(preds)
        rows[market]["called up %"] = 100 * (preds["p_up"] > 0.5).mean()
        rows[market]["all-up folds"] = all_up_folds(preds)  # out of 6
    return pd.DataFrame(rows).T


def overview(results):
    """One row per model, counted over all 36 folds."""
    rows = {}
    for name, preds_by_market in results.items():
        years = pd.concat([report(preds) for preds in preds_by_market.values()])
        above = years["accuracy %"] > years["fair bar %"]
        below = years["accuracy %"] < years["fair bar %"]
        significant = years["p vs fair"] < 0.05
        rows[name] = {
            "mean AUC": years["AUC"].mean(),
            "above fair bar": above.sum(),
            "below fair bar": below.sum(),
            "significantly above": (above & significant).sum(),
            "significantly below": (below & significant).sum(),
            "above stretch bar": (years["accuracy %"] > years["stretch bar %"]).sum(),
            "all-up folds": sum(all_up_folds(preds) for preds in preds_by_market.values()),
        }
    return pd.DataFrame(rows).T
