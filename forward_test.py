"""
The forward test: step 9's models, frozen now, scored on days that didn't exist when they were trained.
See PLAN_clean_checks.md.

Run:  python forward_test.py freeze   # now: train on every day to 2026-10-23, save the models and their hashes
      python forward_test.py score    # late 2027, with DATA_DIR=Dataset/forward (see the README)
"""
import hashlib
import sys
from pathlib import Path

import cloudpickle  # saves the models' predict functions, which plain pickle can't
import pandas as pd
from joblib import Parallel, delayed

from data import DATA_DIR, MARKETS, NEW_MARKETS, load_all
from evaluation import majority_direction
from features import make_dataset
from more_inputs import MODELS, RUNS, TARGETS, all_inputs, selections_for, tsfresh_selection, with_selection

LAST_TRAINING_DAY = pd.Timestamp("2026-10-23")
LAST_NEW_DAY = pd.Timestamp("2027-11-26")  # fixed in advance; download_data.py forward stops here
FOLDER = Path("results") / "forward"       # the saved models, and later their predictions; not in git
HASHES = Path("forward_test_models.csv")   # in git: each model's setting, fair bar and hash


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------- now: freeze ----------

def freeze_one(path, set_name, target, model, market, df, X, selections):
    """Train one model on every day to LAST_TRAINING_DAY and save it.

    As in step 9's folds: the setting is picked on the last year (2026) after training on the years before,
    then the model is refitted on everything.
    """
    fit, grid, pick = MODELS[model]
    if set_name == "C":
        fit, pick = with_selection(fit, pick, selections)
    X, y = make_dataset(df, lambda _: X.reindex(df.index), TARGETS[target])
    last_year = LAST_TRAINING_DAY.year
    train, tune = X.index.year < last_year, X.index.year == last_year
    best = pick(fit, grid, X[train], y[train], X[tune], y[tune])
    saved = {"predict": fit(X, y, best), "fair_call": majority_direction(y)}
    path.write_bytes(cloudpickle.dumps(saved))
    return {"target": target, "set": set_name, "model": model, "market": market,
            "setting": " ".join(str(best).split()), "fair_call": "up" if saved["fair_call"] else "down",
            "training_days": len(y), "file": path.name, "sha256": file_hash(path)}


def freeze(markets, jobs=4):
    """Freeze every run in more_inputs.RUNS, in every market, and write HASHES."""
    if HASHES.exists():
        raise SystemExit(f"{HASHES} exists: the models are already frozen")
    markets = {m: df[df.index <= LAST_TRAINING_DAY] for m, df in markets.items()}
    inputs = all_inputs(markets)
    selections = {}  # set C: the step 9 cutoffs (2015-2025), plus one on every training row
    for m, df in markets.items():
        X, y = make_dataset(df, lambda _: inputs["C"][m].reindex(df.index), TARGETS["close to close"])
        selections[m] = {**selections_for(m, df, inputs["C"][m], "close to close"),
                         LAST_TRAINING_DAY.year: tsfresh_selection(X, y, LAST_TRAINING_DAY.year)}
    FOLDER.mkdir(parents=True, exist_ok=True)
    keys = [(target, s, model, m) for target, sets in RUNS.items() for s in sets for model in MODELS for m in markets]
    rows = Parallel(n_jobs=jobs)(
        delayed(freeze_one)(FOLDER / f"{s}_{target}_{model}_{m}.pkl".replace(" ", "_"), s, target, model, m,
                            markets[m], inputs[s][m], selections[m])
        for target, s, model, m in keys)
    pd.DataFrame(rows).to_csv(HASHES, index=False)
    print(f"froze {len(rows)} models -> {FOLDER}, hashes -> {HASHES}")


# ---------- late 2027: score ----------

def score(markets):
    """Check every frozen model's hash, then predict each new day. Saves {(target, set, model, market): predictions}."""
    if DATA_DIR.name != "forward":
        raise SystemExit("set DATA_DIR to Dataset/forward first: the files in Dataset/ stop at 2026-10-23")
    markets = {m: df[df.index <= LAST_NEW_DAY] for m, df in markets.items()}
    ends_early = [m for m, df in markets.items() if df.index[-1] < LAST_NEW_DAY - pd.Timedelta(days=7)]
    if ends_early:
        raise SystemExit(f"the data stops early in {ends_early}: download it to {LAST_NEW_DAY.date()}")
    frozen = pd.read_csv(HASHES)
    changed = [row.file for row in frozen.itertuples() if file_hash(FOLDER / row.file) != row.sha256]
    if changed:
        raise SystemExit(f"{len(changed)} models changed since they were frozen, e.g. {changed[0]}")

    inputs = all_inputs(markets, folder=FOLDER)  # tsfresh's features are recomputed, on the new data
    preds = {}
    for row in frozen.itertuples():
        saved = cloudpickle.loads((FOLDER / row.file).read_bytes())
        df, make_y = markets[row.market], TARGETS[row.target]
        X, y = make_dataset(df, lambda _: inputs[row.set][row.market].reindex(df.index), make_y)
        new = X.index[X.index > LAST_TRAINING_DAY]
        preds[row.target, row.set, row.model, row.market] = pd.DataFrame({
            "year": new.year,
            "setting": row.setting,
            "p_up": saved["predict"](X.loc[new]),
            "up": y[new],
            "fair_call": saved["fair_call"],  # frozen with the model
            "same_as_yesterday": make_y(df).shift(1).eq(True)[new],
        }, index=new)
    pd.to_pickle(preds, FOLDER / "predictions.pkl")
    print(f"scored {len(preds)} models -> {FOLDER / 'predictions.pkl'}")


if __name__ == "__main__":
    all_markets = load_all(include_holdout=True, markets=MARKETS + NEW_MARKETS)
    if "freeze" in sys.argv:
        freeze(all_markets)
    elif "score" in sys.argv:
        score(all_markets)
    else:
        print(__doc__)
