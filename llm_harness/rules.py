"""The menu of rules the model may use. Each one can be checked on data, so a stated reason can be tested."""
import numpy as np
import pandas as pd

WINDOWS = {"1-day move": 1, "5-day move": 5, "20-day move": 19}  # 20 days shown = 19 moves from first to last close
RULES = [f"{direction} the {window}" for window in WINDOWS for direction in ("follow", "reverse")] + ["none of these"]
# e.g. "follow the 5-day move": if the last 5 days went up overall, say up. "reverse": say the opposite.


def parse(rule):
    """'reverse the 5-day move' -> ('reverse', 5). None for 'none of these'."""
    if rule == "none of these":
        return None
    direction, _, window = rule.partition(" the ")
    return direction, WINDOWS[window]


def implied_answer(rows, rule):
    """The answer this rule gives for these rows: "up" or "down". None if the rule is "none of these"."""
    if parse(rule) is None:
        return None
    direction, n = parse(rule)
    went_up = rows["Close"].iloc[-1] > rows["Close"].iloc[-1 - n]
    return "up" if went_up == (direction == "follow") else "down"


def mirror(rows, n):
    """The same rows with the last n daily moves turned upside down (a 1% rise becomes a 1% fall).

    Earlier days are unchanged. Each mirrored day's open, high and low move with its close, and high/low swap
    so the high stays on top. Used to test whether an answer really depends on the move the model named.
    """
    close = rows["Close"].to_numpy()
    moves = np.diff(np.log(close))
    moves[-n:] *= -1
    new_close = close[0] * np.exp(np.concatenate([[0.0], np.cumsum(moves)]))
    factor = new_close / close
    out = rows.mul(factor, axis=0)
    flipped = rows.index[-n:]
    # Mirroring a day turns its high into its low: reflect open/high/low around the day's new close as well.
    for day in flipped:
        c = out.at[day, "Close"]
        o, h, l = (c * c / out.at[day, col] for col in ("Open", "Low", "High"))
        out.loc[day, ["Open", "High", "Low"]] = [o, h, l]
    return out


def hit_rate(df, rule, before):
    """How often the rule was right on every day before `before`, using that market's full history. None for no rule."""
    if parse(rule) is None:
        return None
    direction, n = parse(rule)
    close = df["Close"]
    past_move_up = (close.shift(1) > close.shift(1 + n))
    went_up = df["ret"] > 0
    calls_up = past_move_up if direction == "follow" else ~past_move_up
    usable = df.index[(df.index < pd.Timestamp(before)) & close.shift(1 + n).notna() & df["ret"].notna()]
    return float((calls_up[usable] == went_up[usable]).mean())
