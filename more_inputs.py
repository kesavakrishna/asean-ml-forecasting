"""
Step 9: do more inputs help the step 4 models? See PLAN_more_inputs.md.

Four sets of inputs (A: Kim's 12, B: standard extras, C: tsfresh, D: other markets), scored on the same days,
with test years 2017-2026. Every run is saved, so a stopped run carries on where it left off.

Run:  python more_inputs.py pilot   # one market, one test year: checks the code and the time
      python more_inputs.py         # everything
"""
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

from classifiers import CLASSIFIERS, HIDDEN_GRID, fit_classifier, fit_net, pick_net_setting
from data import load_all, load_other_moves, load_us
from features import (make_dataset, make_indicators, make_indicators_extended, make_other_markets, make_tsfresh,
                      up_close_to_close, up_open_to_close)
from models import pick_by_log_loss, walk_forward

TEST_YEARS = range(2017, 2027)
SAVED = Path("results") / "more_inputs"  # not in git
FALLBACK = 10  # if tsfresh keeps no features, keep the 10 with the smallest p-values

MODELS = {name: (fit_classifier, grid, pick_by_log_loss) for name, grid in CLASSIFIERS.items()}
MODELS["Neural network"] = (fit_net, HIDDEN_GRID, pick_net_setting)
TARGETS = {"close to close": up_close_to_close, "open to close": up_open_to_close}
RUNS = {"close to close": ["A", "B", "C", "D"], "open to close": ["A", "D"]}  # as in the plan


# ---------- the inputs ----------

def tsfresh_features(market, df):
    """Set C for one market. Computed once and saved, because it takes minutes."""
    path = SAVED / f"tsfresh_{market}.pkl"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        make_tsfresh(df).to_pickle(path)
    return pd.read_pickle(path)


def all_inputs(markets):
    """{set: {market: X}}. Each X keeps only the days where all four sets have values."""
    us, others = load_us(include_holdout=True), load_other_moves()
    inputs = {name: {} for name in "ABCD"}
    for market, df in markets.items():
        X = {
            "A": make_indicators(df),
            "B": make_indicators_extended(df),
            "C": tsfresh_features(market, df),
            "D": make_other_markets(df, us, others, market),
        }
        common = pd.concat([x.notna().all(axis=1) for x in X.values()], axis=1).all(axis=1)
        for name, x in X.items():
            inputs[name][market] = x[common]
    return inputs


# ---------- tsfresh's feature selection, on training rows only ----------

def tsfresh_selection(X, y, last_year):
    """The tsfresh columns kept using rows up to the end of last_year only. Returns (columns, fell back?)."""
    from tsfresh.feature_selection.relevance import calculate_relevance_table

    rows = X.index.year <= last_year
    table = calculate_relevance_table(X[rows], y[rows], fdr_level=0.05)
    keep = list(table.index[table["relevant"]])
    if keep:
        return keep, False
    return list(table.dropna(subset=["p_value"]).nsmallest(FALLBACK, "p_value").index), True


def selections_for(market, df, X, target):
    """Set C's selections for every training cutoff the folds use, computed once and saved."""
    path = SAVED / f"tsfresh_selection_{target}_{market}.pkl".replace(" ", "_")
    if path.exists():
        return pd.read_pickle(path)
    X_all, y_all = make_dataset(df, lambda _: X.reindex(df.index), TARGETS[target])
    cutoffs = range(min(TEST_YEARS) - 2, max(TEST_YEARS))  # train ends at Y-2; train + tune ends at Y-1
    chosen = {year: tsfresh_selection(X_all, y_all, year) for year in cutoffs}
    pd.to_pickle(chosen, path)
    return chosen


def with_selection(fit, pick, selections):
    """fit and pick for set C: each sees only the columns chosen on its own training rows."""
    def columns(X):
        return selections[X.index[-1].year][0]  # the training rows end in this year

    def fit_c(X, y, setting):
        keep = columns(X)
        predict = fit(X[keep], y, setting)
        return lambda X_new: predict(X_new[keep])

    def pick_c(_, grid, X_train, y_train, X_tune, y_tune):
        keep = columns(X_train)
        return pick(fit, grid, X_train[keep], y_train, X_tune[keep], y_tune)

    return fit_c, pick_c


# ---------- the runs ----------

def run_one(path, set_name, target, model, df, X, selections, test_years):
    """One model on one market with one set of inputs. Loads the saved run if there is one."""
    if path.exists():
        return pd.read_pickle(path)
    fit, grid, pick = MODELS[model]
    if set_name == "C":
        fit, pick = with_selection(fit, pick, selections)
    preds = walk_forward(df, fit, grid, make_x=lambda _: X.reindex(df.index), make_y=TARGETS[target],
                         pick=pick, test_years=test_years)
    path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_pickle(path)
    return preds


def run_all(markets, test_years=TEST_YEARS, folder=SAVED, jobs=8):
    """Every run in RUNS, as {(target, set, model, market): predictions}."""
    inputs = all_inputs(markets)
    selections = {(target, market): selections_for(market, df, inputs["C"][market], target)
                  for target, sets in RUNS.items() if "C" in sets for market, df in markets.items()}
    keys = [(target, s, model, market) for target, sets in RUNS.items() for s in sets
            for model in MODELS for market in markets]
    paths = [folder / f"{s}_{target}_{model}_{market}.pkl".replace(" ", "_") for target, s, model, market in keys]
    preds = Parallel(n_jobs=jobs)(
        delayed(run_one)(path, s, target, model, markets[market], inputs[s][market],
                         selections.get((target, market)), test_years)
        for path, (target, s, model, market) in zip(paths, keys))
    return dict(zip(keys, preds))


if __name__ == "__main__":
    markets = load_all(include_holdout=True)
    start = time.time()
    if "pilot" in sys.argv:  # timing and a check that everything runs; not scored
        run_all({"Singapore": markets["Singapore"]}, test_years=[2017], folder=SAVED / "pilot")
    else:
        run_all(markets)
    print(f"done in {(time.time() - start) / 60:.0f} minutes")
