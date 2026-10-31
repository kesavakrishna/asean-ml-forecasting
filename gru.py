"""Step 6: a small GRU in PyTorch. It reads the last 20 days in order, oldest first."""
import numpy as np
from torch import nn

from classifiers import MAX_EPOCHS, train_net
from features import SEQ_DAYS

HIDDEN_SIZES = [8, 16, 32]


class GRUNet(nn.Module):
    def __init__(self, n_inputs, hidden):
        super().__init__()
        self.inputs_per_day = n_inputs // SEQ_DAYS
        self.gru = nn.GRU(self.inputs_per_day, hidden, batch_first=True)
        self.out = nn.Linear(hidden, 1)  # GRU's final memory -> 1 logit

    def forward(self, x):
        days = x.reshape(len(x), SEQ_DAYS, self.inputs_per_day)  # flat columns -> rows, days, inputs per day
        _, memory = self.gru(days)
        return self.out(memory[-1])


def fit_gru(X, y, setting, seed=0):
    """Train with a fixed setting = (hidden size, epochs). Returns predict()."""
    hidden, epochs = setting
    predict, _ = train_net(X, y, hidden, epochs, build_net=GRUNet, seed=seed)
    return predict


def pick_gru_setting(fit, hidden_sizes, X_train, y_train, X_tune, y_tune, seed=0):
    """Early stopping on the tune year for each hidden size. Returns the best (hidden size, epochs)."""
    best_loss = {}
    for hidden in hidden_sizes:
        _, losses = train_net(X_train, y_train, hidden, MAX_EPOCHS, X_tune, y_tune, build_net=GRUNet, seed=seed)
        best_epochs = int(np.argmin(losses)) + 1
        best_loss[(hidden, best_epochs)] = min(losses)
    return min(best_loss, key=best_loss.get)
