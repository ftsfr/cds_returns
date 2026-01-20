"""
Create FTSFR standardized datasets for CDS returns.

Outputs:
- ftsfr_cds_portfolio_returns.parquet: Monthly CDS portfolio returns (20 portfolios: 4 tenors x 5 credit quintiles)
- ftsfr_cds_contract_returns.parquet: Monthly CDS returns at the individual contract level
"""

import sys
from pathlib import Path

sys.path.insert(0, "./src")

import pandas as pd

import chartbook
import calc_cds_returns

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # Portfolio-level returns
    # =========================================================================
    print(">> Creating ftsfr_cds_portfolio_returns...")
    df_portfolios = calc_cds_returns.load_portfolio(data_dir=DATA_DIR)

    # Convert from wide to long format for ftsfr standard
    df_portfolio_long = df_portfolios.melt(
        id_vars=["Month"], var_name="unique_id", value_name="y"
    )

    # Rename columns to match ftsfr standard
    df_portfolio_long = df_portfolio_long.rename(columns={"Month": "ds"})

    # Sort by portfolio and date
    df_portfolio_long = df_portfolio_long.sort_values(
        by=["unique_id", "ds"]
    ).reset_index(drop=True)
    df_portfolio_long = df_portfolio_long[["unique_id", "ds", "y"]]

    # Check for duplicates
    duplicates = df_portfolio_long.duplicated(subset=["unique_id", "ds"])
    num_duplicates = duplicates.sum()
    if num_duplicates > 0:
        print(
            f"   Warning: Found {num_duplicates} duplicate (unique_id, ds) pairs in portfolio returns."
        )
        df_portfolio_long = df_portfolio_long.drop_duplicates(
            subset=["unique_id", "ds"]
        )
    else:
        print("   No duplicate (unique_id, ds) pairs found.")

    # Save as ftsfr dataset
    df_portfolio_long = df_portfolio_long.dropna()
    output_path = DATA_DIR / "ftsfr_cds_portfolio_returns.parquet"
    df_portfolio_long.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path.name}")
    print(f"   Records: {len(df_portfolio_long):,}")
    print(f"   Portfolios: {df_portfolio_long['unique_id'].nunique()}")

    # =========================================================================
    # Contract-level returns
    # =========================================================================
    print("\n>> Creating ftsfr_cds_contract_returns...")
    df_contracts = calc_cds_returns.load_contract_returns(data_dir=DATA_DIR)

    # Create unique_id by combining ticker and tenor
    df_contracts["unique_id"] = df_contracts["ticker"] + "_" + df_contracts["tenor"]

    # Select and rename columns to match ftsfr standard
    df_contract_long = df_contracts[["unique_id", "Month", "monthly_return"]].rename(
        columns={"Month": "ds", "monthly_return": "y"}
    )

    # Sort by contract and date
    df_contract_long = df_contract_long.sort_values(
        by=["unique_id", "ds"]
    ).reset_index(drop=True)

    # Check for duplicates
    duplicates = df_contract_long.duplicated(subset=["unique_id", "ds"])
    num_duplicates = duplicates.sum()
    if num_duplicates > 0:
        print(
            f"   Warning: Found {num_duplicates} duplicate (unique_id, ds) pairs in contract returns."
        )
        df_contract_long = df_contract_long.drop_duplicates(subset=["unique_id", "ds"])
    else:
        print("   No duplicate (unique_id, ds) pairs found.")

    # Save as ftsfr dataset
    df_contract_long = df_contract_long.dropna()
    output_path = DATA_DIR / "ftsfr_cds_contract_returns.parquet"
    df_contract_long.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path.name}")
    print(f"   Records: {len(df_contract_long):,}")
    print(f"   Unique contracts: {df_contract_long['unique_id'].nunique()}")


if __name__ == "__main__":
    main()
