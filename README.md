# asean-ml-forecasting

Forecasting the daily direction (up/down) of the ASEAN-5 stock indices with machine learning: Indonesia, Malaysia, the Philippines, Singapore and Thailand, with India's Nifty 50 as a benchmark.

**Status:** EDA is done (see [eda_report.md](eda_report.md)). The cleaning, walk-forward evaluation and leakage test are in place.

## Files

| File | What it is |
|---|---|
| `data.py` | `load_market()` / `load_all()`: load, clean and check the data. Every step after the EDA uses this |
| `evaluation.py` | Walk-forward splits (test years 2017–2022), scoring (accuracy, balanced accuracy, MCC), no-model baselines |
| `test_leakage.py` | Checks that nothing computed for day t changes when data after t is removed or scrambled |
| `walk_forward_baselines.ipynb` | Baselines scored on each test year |
| `eda.ipynb` | EDA notebook: data checks, cleaning, returns, up-day shares, baseline scores |
| `eda_report.md` | Write-up of the EDA findings |
| `eda_output/` | Figures used in the report (saved by the EDA notebook) |

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
python test_leakage.py
jupyter notebook walk_forward_baselines.ipynb
```
