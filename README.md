# asean-ml-forecasting

Forecasting the daily direction (up/down) of the ASEAN-5 stock indices with machine learning: Indonesia, Malaysia, the Philippines, Singapore and Thailand, with India's Nifty 50 as a benchmark.

**Status:** exploratory data analysis is done. See [eda_report.md](eda_report.md) for the findings.

## Files

| File | What it is |
|---|---|
| `eda.ipynb` | EDA notebook: data checks, cleaning, returns, up-day shares, baseline scores |
| `eda_report.md` | Write-up of the EDA findings |
| `eda_output/` | Figures used in the report (saved by the notebook) |

## Data

The data isn't included in this repo. It's daily index history exported from [Investing.com](https://www.investing.com), Jan 2012 – Mar 2023 (the Nifty 50 starts in Jan 2010). To run the notebook, put these files in a `Dataset/` folder:

- `Jakarta Stock Exchange Composite Index Historical Data.csv`
- `FTSE Malaysia KLCI Historical Data.csv`
- `FTSE Philippines Historical Data.csv`
- `FTSE Straits Times Singapore Historical Data.csv`
- `SET Thailand Index Historical Data.csv`
- `Nifty 50 Historical Data.csv`

Each file needs `Date`, `Open`, `High`, `Low` and `Price` (or `Close`) columns.

## Running

Tested with Python 3.13 and pandas 3.0.

```
pip install pandas numpy matplotlib jupyter
jupyter notebook eda.ipynb
```
