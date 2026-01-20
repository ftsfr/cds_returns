"""
Pull Treasury yield data from FRED.
"""

import sys
from pathlib import Path

sys.path.insert(0, "./src")

import pandas as pd
import pandas_datareader.data as web

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

START_DATE = pd.Timestamp("2001-01-01")
END_DATE = pd.Timestamp("2025-01-01")

SERIES_TO_PULL = {
    "DGS1MO": "1-Month Treasury Yield",
    "DGS3MO": "3-Month Treasury Yield",
    "DGS6MO": "6-Month Treasury Yield",
    "DGS1": "1-Year Treasury Yield",
    "DGS2": "2-Year Treasury Yield",
    "DGS3": "3-Year Treasury Yield",
}


def pull_fred(start_date=START_DATE, end_date=END_DATE):
    """
    Pull Treasury yield series from FRED.
    """
    print(">> Pulling FRED Treasury yields...")
    df = web.DataReader(list(SERIES_TO_PULL.keys()), "fred", start_date, end_date)
    print(f">> Records: {len(df):,}")
    return df


def load_fred(data_dir=DATA_DIR):
    file_path = data_dir / "fred.parquet"
    return pd.read_parquet(file_path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    df = pull_fred(START_DATE, today)
    df.to_parquet(DATA_DIR / "fred.parquet")
    print(f">> Saved fred.parquet")


if __name__ == "__main__":
    main()
