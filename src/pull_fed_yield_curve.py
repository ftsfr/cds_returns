"""
Pull zero coupon yield curve from the Federal Reserve.
"""

import sys
from pathlib import Path
from io import BytesIO

sys.path.insert(0, "./src")

import pandas as pd
import requests

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"


def pull_fed_yield_curve():
    """
    Download the latest yield curve from the Federal Reserve.
    Uses Gurkaynak, Sack, and Wright (2007) model.
    """
    print(">> Pulling Fed yield curve...")
    url = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
    response = requests.get(url)
    pdf_stream = BytesIO(response.content)
    df = pd.read_csv(pdf_stream, skiprows=9, index_col=0, parse_dates=True)
    cols = ["SVENY" + str(i).zfill(2) for i in range(1, 31)]
    print(f">> Records: {len(df):,}")
    return df[cols]


def load_fed_yield_curve(data_dir=DATA_DIR):
    path = data_dir / "fed_yield_curve.parquet"
    return pd.read_parquet(path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pull_fed_yield_curve()
    path = DATA_DIR / "fed_yield_curve.parquet"
    df.to_parquet(path)
    print(f">> Saved {path}")


if __name__ == "__main__":
    main()
