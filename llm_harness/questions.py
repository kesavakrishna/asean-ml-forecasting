"""The three ways of asking the model about day t. Nothing from day t or later goes into a question."""

DAYS_SHOWN = 20
SYSTEM = "You are taking part in a forecasting experiment. Reply with exactly one word: up or down."
ENDING = "Answer with one word: up or down."
INDEX_NAMES = {
    "Indonesia": "Jakarta Composite Index (Indonesia)",
    "Malaysia": "FTSE Bursa Malaysia KLCI (Malaysia)",
    "Philippines": "PSEi Composite Index (Philippines)",
    "Singapore": "Straits Times Index (Singapore)",
    "Thailand": "SET Index (Thailand)",
    "India": "Nifty 50 (India)",
}


def days_before(df, day):
    """The 20 trading days before `day`. This is the wall: day t itself is never shown."""
    rows = df.loc[df.index < day, ["Open", "High", "Low", "Close"]].tail(DAYS_SHOWN)
    assert len(rows) == DAYS_SHOWN and rows.index.max() < day
    return rows


def as_text(rows, labels):
    """One line per day: its label, then open, high, low, close."""
    return "\n".join(f"{label}, " + ", ".join(f"{price:.2f}" for price in row) for label, row in zip(labels, rows.to_numpy()))


def named(market, df, day):
    """Real index name, real dates, real prices."""
    rows = days_before(df, day)
    return (f"Market: {INDEX_NAMES[market]}\n"
            f"Its last {DAYS_SHOWN} trading days (date, open, high, low, close):\n"
            f"{as_text(rows, rows.index.strftime('%Y-%m-%d'))}\n"
            f"Will it close higher or lower on {day:%Y-%m-%d} than on {rows.index[-1]:%Y-%m-%d}?\n{ENDING}")


def hidden(market, df, day):
    """The same prices with the name and the dates removed, rescaled so the first close is 100."""
    rows = days_before(df, day)
    rows = rows * 100 / rows["Close"].iloc[0]
    return ("Market: a stock index (name and dates hidden)\n"
            f"Its last {DAYS_SHOWN} trading days (day, open, high, low, close), rescaled so the first close is 100:\n"
            f"{as_text(rows, [f'day {n}' for n in range(-DAYS_SHOWN, 0)])}\n"
            f"Will it close higher or lower on the next trading day than on day -1?\n{ENDING}")


def date_only(market, df, day):
    """No prices at all, only the index name and the date. Anything it gets right here is memory."""
    previous = df.index[df.index < day][-1]
    return (f"Market: {INDEX_NAMES[market]}\n"
            f"Did it close higher or lower on {day:%Y-%m-%d} than on {previous:%Y-%m-%d}?\n{ENDING}")
