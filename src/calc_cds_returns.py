"""
Calculate CDS returns following He, Kelly, and Manela (2017) methodology.

The CDS return calculation uses the He-Kelly formula:
    CDS_Return_t = CDS_{t-1}/250 + ΔCDS_t * RD_{t-1}

Where:
- CDS_{t-1}/250: Carry return, daily accrual from previous day's spread
- ΔCDS_t: Daily change in spread
- RD_{t-1}: Risky duration, proxy for PV of future spread payments
"""

import sys
from pathlib import Path
import datetime
import time

sys.path.insert(0, "./src")

import numpy as np
import pandas as pd
import polars as pl
from scipy.interpolate import CubicSpline

import chartbook
import pull_fed_yield_curve
import pull_fred
import pull_markit_cds

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

START_DATE = pull_markit_cds.START_DATE
END_DATE = pull_markit_cds.END_DATE


def _format_elapsed_time(seconds):
    """Format elapsed time in a human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f} sec"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} min"
    else:
        return f"{seconds / 3600:.1f} hours"


def process_rates(raw_rates=None, start_date=START_DATE, end_date=END_DATE, data_dir=DATA_DIR):
    """
    Processes raw interest rate data by filtering within a specified date range
    and converting column names to numerical maturity values.
    """
    raw_rates = raw_rates.copy().dropna()
    swap_rates = pull_fred.load_fred(data_dir=data_dir)
    short_tenor_rates = swap_rates[["DGS3MO", "DGS6MO"]]
    short_tenor_rates_renamed = short_tenor_rates.rename(
        columns={"DGS3MO": 0.25, "DGS6MO": 0.5}
    )
    raw_rates.columns = raw_rates.columns.str.extract(r"(\d+)$")[0].astype(int)
    rates = raw_rates[
        (raw_rates.index >= pd.to_datetime(start_date))
        & (raw_rates.index <= pd.to_datetime(end_date))
    ]

    merged_rates = pd.merge(
        rates, short_tenor_rates_renamed, left_index=True, right_index=True, how="inner"
    ).sort_index()
    cols = merged_rates.columns.tolist()
    ordered_cols = [0.25, 0.5] + [col for col in cols if col not in [0.25, 0.5]]
    merged_rates = merged_rates[ordered_cols]
    return merged_rates


def extrapolate_rates(rates=None):
    """
    Interpolates interest rates to quarterly intervals using cubic splines.
    """
    years = np.array(rates.columns)
    quarterly_maturities = np.arange(0.25, 30.25, 0.25)

    interpolated_data = []
    for _, row in rates.iterrows():
        values = row.values
        cs = CubicSpline(years, values, extrapolate=True)
        interpolated_values = cs(quarterly_maturities)
        interpolated_data.append(interpolated_values)

    df_quarterly = pd.DataFrame(interpolated_data, columns=quarterly_maturities)
    df_quarterly.index = rates.index
    return df_quarterly


def calc_discount(raw_rates=None, start_date=START_DATE, end_date=END_DATE, data_dir=DATA_DIR):
    """
    Calculates discount factors from interest rates for risky duration computation.
    """
    rates_data = process_rates(raw_rates, start_date, end_date, data_dir=data_dir)
    if rates_data is None:
        print("No data available for the given date range.")
        return None

    quarterly_rates = extrapolate_rates(rates_data)

    quarterly_discount = pd.DataFrame(
        columns=quarterly_rates.columns, index=quarterly_rates.index
    )
    for col in quarterly_rates.columns:
        quarterly_discount[col] = quarterly_rates[col].apply(
            lambda x: np.exp(-(col * x) / 4)
        )

    return quarterly_discount


def _get_filtered_cds_data_with_quantiles(start_date=START_DATE, end_date=END_DATE, cds_spreads=None):
    """
    Common function to filter CDS data and assign credit quantiles.
    """
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d")
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    if isinstance(cds_spreads, pd.DataFrame):
        cds_spreads = pl.from_pandas(cds_spreads)

    if isinstance(cds_spreads, pl.DataFrame):
        cds_spreads = cds_spreads.lazy()

    cds_spread_clean = (
        cds_spreads.filter(
            (pl.col("date") >= start_date)
            & (pl.col("date") <= end_date)
            & (pl.col("country") == "United States")
            & (pl.col("parspread").is_not_null())
            & (pl.col("parspread") <= 0.5)
        )
        .drop(["convspreard", "year", "redcode"])
        .unique()
        .with_columns(pl.col("date").dt.strftime("%Y-%m").alias("year_month"))
    )

    spread_5y = cds_spread_clean.filter(pl.col("tenor") == "5Y")

    first_spread_5y = (
        spread_5y.sort("date")
        .group_by(["ticker", "year_month"])
        .first()
        .select(["ticker", "year_month", "parspread"])
        .collect()
    )

    credit_quantiles = first_spread_5y.group_by("year_month").agg(
        [
            pl.col("parspread").quantile(0.2).alias("q1"),
            pl.col("parspread").quantile(0.4).alias("q2"),
            pl.col("parspread").quantile(0.6).alias("q3"),
            pl.col("parspread").quantile(0.8).alias("q4"),
        ]
    )

    first_spread_5y = first_spread_5y.join(credit_quantiles, on="year_month")

    first_spread_5y = first_spread_5y.with_columns(
        pl.when(pl.col("parspread") <= pl.col("q1"))
        .then(1)
        .when(pl.col("parspread") <= pl.col("q2"))
        .then(2)
        .when(pl.col("parspread") <= pl.col("q3"))
        .then(3)
        .when(pl.col("parspread") <= pl.col("q4"))
        .then(4)
        .otherwise(5)
        .alias("credit_quantile")
    ).select(["ticker", "year_month", "credit_quantile"])

    cds_spreads_final = cds_spread_clean.join(
        first_spread_5y.lazy(), on=["ticker", "year_month"], how="left"
    ).sort("date")

    return cds_spreads_final


def get_contract_data(start_date=START_DATE, end_date=END_DATE, cds_spreads=None):
    """
    Gets individual CDS contract data with credit quantile assignments.
    """
    cds_spreads_final = _get_filtered_cds_data_with_quantiles(
        start_date, end_date, cds_spreads
    )

    relevant_tenors = ["3Y", "5Y", "7Y", "10Y"]

    contract_data = cds_spreads_final.filter(
        (pl.col("tenor").is_in(relevant_tenors))
        & (pl.col("credit_quantile").is_not_null())
    ).collect()

    return contract_data


def get_portfolio_dict(start_date=START_DATE, end_date=END_DATE, cds_spreads=None):
    """
    Creates a dictionary of credit portfolios based on the CDS spread data.
    """
    cds_spreads_final = _get_filtered_cds_data_with_quantiles(
        start_date, end_date, cds_spreads
    )

    relevant_tenors = ["3Y", "5Y", "7Y", "10Y"]
    relevant_quantiles = [1, 2, 3, 4, 5]

    rep_parspread_df = (
        cds_spreads_final.filter(
            (pl.col("tenor").is_in(relevant_tenors))
            & (pl.col("credit_quantile").is_in(relevant_quantiles))
        )
        .group_by(["date", "tenor", "credit_quantile"])
        .agg(pl.col("parspread").mean().alias("rep_parspread"))
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
    )

    portfolio_dict = {}

    for tenor in relevant_tenors:
        for quantile in relevant_quantiles:
            key = f"{tenor}_Q{quantile}"
            portfolio_df = (
                rep_parspread_df.filter(
                    (pl.col("tenor") == tenor) & (pl.col("credit_quantile") == quantile)
                )
                .sort("date")
                .collect()
            )
            portfolio_dict[key] = portfolio_df

    return portfolio_dict


def _get_quarterly_discount_polars(raw_rates=None, start_date=START_DATE, end_date=END_DATE, data_dir=DATA_DIR):
    """
    Helper function to compute quarterly discount rates and convert to Polars format.
    """
    quarterly_discount_pd = calc_discount(
        raw_rates, start_date, end_date, data_dir=data_dir
    )
    quarterly_discount_pd = quarterly_discount_pd.iloc[:-1]
    return pl.from_pandas(quarterly_discount_pd.reset_index())


def calc_cds_return_for_contracts(
    contract_data=None,
    raw_rates=None,
    start_date=START_DATE,
    end_date=END_DATE,
    data_dir=DATA_DIR,
):
    """
    Calculates CDS returns for individual contracts using the He-Kelly formula.
    """
    quarterly_discount = _get_quarterly_discount_polars(raw_rates, start_date, end_date, data_dir=data_dir)

    fiveY_data = contract_data.filter(pl.col("tenor") == "5Y")

    loss_given_default = 0.6
    fiveY_lambdas = fiveY_data.with_columns(
        (4 * np.log(1 + (pl.col("parspread") / (4 * loss_given_default)))).alias("lambda")
    ).select(["ticker", "date", "credit_quantile", "lambda"])

    contract_data_with_lambda = contract_data.join(
        fiveY_lambdas, on=["ticker", "date", "credit_quantile"], how="left"
    )

    avg_lambda_by_quantile = fiveY_lambdas.group_by(["date", "credit_quantile"]).agg(
        pl.col("lambda").mean().alias("avg_lambda")
    )

    contract_data_with_lambda = contract_data_with_lambda.join(
        avg_lambda_by_quantile, on=["date", "credit_quantile"], how="left"
    ).with_columns(
        pl.coalesce([pl.col("lambda"), pl.col("avg_lambda")]).alias("lambda_final")
    )

    contract_data_sorted = contract_data_with_lambda.sort(["ticker", "tenor", "date"])

    quarters = np.arange(0.25, 20.25, 0.25)
    all_contract_returns = []

    unique_contracts = contract_data_sorted.select(["ticker", "tenor"]).unique()
    total_contracts = len(unique_contracts)

    print(f"\nProcessing {total_contracts} individual CDS contracts...")
    start_time = time.time()

    for idx, row in enumerate(unique_contracts.iter_rows()):
        ticker, tenor = row

        if idx % 500 == 0:
            elapsed = time.time() - start_time
            progress_pct = (idx / total_contracts) * 100
            if idx > 0:
                rate = idx / elapsed
                remaining = (total_contracts - idx) / rate
                print(
                    f"  Progress: {idx}/{total_contracts} ({progress_pct:.1f}%), "
                    f"Elapsed: {_format_elapsed_time(elapsed)}, "
                    f"Remaining: {_format_elapsed_time(remaining)}"
                )

        contract = contract_data_sorted.filter(
            (pl.col("ticker") == ticker) & (pl.col("tenor") == tenor)
        ).sort("date")

        if len(contract) < 2:
            continue

        dates = contract["date"].to_numpy()
        lambda_vals = contract["lambda_final"].to_numpy()
        parspreads = contract["parspread"].to_numpy()

        survival_probs_list = []
        for lambda_val in lambda_vals:
            survival_probs = np.exp(-quarters * lambda_val)
            survival_probs_list.append(survival_probs)

        discount_dates = quarterly_discount["index"].to_numpy()

        risky_durations = []
        for i, date in enumerate(dates):
            if date in discount_dates:
                date_idx = np.where(discount_dates == date)[0][0]
                discount_row = quarterly_discount.row(date_idx)[1:]

                rd = 0.25 * sum(
                    survival_probs_list[i][j] * discount_row[j]
                    for j in range(min(len(quarters), len(discount_row)))
                )
                risky_durations.append(rd)
            else:
                risky_durations.append(np.nan)

        daily_returns = []
        for i in range(1, len(parspreads)):
            if not np.isnan(risky_durations[i - 1]):
                carry = parspreads[i - 1] / 250
                spread_change = parspreads[i] - parspreads[i - 1]
                daily_return = carry + (spread_change * risky_durations[i - 1])
                daily_returns.append(daily_return)
            else:
                daily_returns.append(np.nan)

        if daily_returns:
            result_df = pl.DataFrame(
                {
                    "ticker": [ticker] * len(daily_returns),
                    "tenor": [tenor] * len(daily_returns),
                    "date": dates[1:],
                    "credit_quantile": contract["credit_quantile"][1:],
                    "daily_return": daily_returns,
                }
            )
            all_contract_returns.append(result_df)

    if all_contract_returns:
        final_returns = pl.concat(all_contract_returns).sort(["ticker", "tenor", "date"])
        total_elapsed = time.time() - start_time
        print(f"  Completed: {total_contracts} contracts, Time: {_format_elapsed_time(total_elapsed)}")
        return final_returns
    else:
        return pl.DataFrame(
            {
                "ticker": pl.Series([], dtype=pl.Utf8),
                "tenor": pl.Series([], dtype=pl.Utf8),
                "date": pl.Series([], dtype=pl.Date),
                "credit_quantile": pl.Series([], dtype=pl.Int64),
                "daily_return": pl.Series([], dtype=pl.Float64),
            }
        )


def calc_cds_return_for_portfolios(
    portfolio_dict=None,
    raw_rates=None,
    start_date=START_DATE,
    end_date=END_DATE,
    data_dir=DATA_DIR,
):
    """
    Calculates CDS returns for each portfolio in the portfolio_dict.
    """
    quarterly_discount = _get_quarterly_discount_polars(
        raw_rates, start_date, end_date, data_dir=data_dir
    )

    cds_return_dict = {}
    fiveY_lambda_dict = {}

    for key, portfolio_df in portfolio_dict.items():
        if key.startswith("5Y_Q"):
            pivot_table = portfolio_df.pivot(
                index="date", on="tenor", values="rep_parspread"
            )

            if "5Y" in pivot_table.columns:
                spread_5Y_df = pivot_table.select(pl.col("5Y"))
                loss_given_default = 0.6
                lambda_df = 4 * np.log(1 + (spread_5Y_df / (4 * loss_given_default)))
                fiveY_lambda_dict[key] = lambda_df

    total_portfolios = len(portfolio_dict)
    print(f"\nProcessing {total_portfolios} CDS portfolios...")
    portfolio_start_time = time.time()

    for portfolio_idx, (key, portfolio_df) in enumerate(portfolio_dict.items()):
        if portfolio_idx % 5 == 0 or portfolio_idx == total_portfolios - 1:
            elapsed = time.time() - portfolio_start_time
            progress_pct = ((portfolio_idx + 1) / total_portfolios) * 100
            print(
                f"  Portfolio {portfolio_idx + 1}/{total_portfolios}: {key} ({progress_pct:.0f}%)"
            )

        pivot_table = portfolio_df.pivot(
            index="date", on="tenor", values="rep_parspread"
        )

        pivot_table = pivot_table.rename(
            {col: f"{key}" for col in pivot_table.columns if col != "date"}
        )

        loss_given_default = 0.6
        quintile_number = key.split("_Q")[-1]
        vol_target_key = f"5Y_Q{quintile_number}"
        lambda_constant = fiveY_lambda_dict.get(vol_target_key, None)

        if lambda_constant is None:
            continue

        quarters = np.arange(0.25, 20.25, 0.25)

        risky_duration = pivot_table.select("date").clone()

        # Handle both Polars DataFrame and numpy array cases
        if hasattr(lambda_constant, 'to_numpy'):
            lambda_vals = lambda_constant.to_numpy().flatten()
        else:
            lambda_vals = np.asarray(lambda_constant).flatten()

        if len(lambda_vals) > len(pivot_table):
            lambda_vals = lambda_vals[: len(pivot_table)]
        elif len(lambda_vals) < len(pivot_table):
            continue

        survival_probs = pl.DataFrame({"date": pivot_table["date"]}).with_columns(
            [pl.Series(name=str(q), values=np.exp(-q * lambda_vals)) for q in quarters]
        )

        discount_filtered = quarterly_discount.select(
            ["index"]
            + [str(q) for q in quarters if str(q) in quarterly_discount.columns]
        )

        survival_probs_filtered = survival_probs.filter(
            pl.col("date").is_in(quarterly_discount["index"].to_list())
        )

        discount_filtered = discount_filtered.rename({"index": "date"})

        dates_df = discount_filtered.select("date")
        dates_spf = survival_probs_filtered.select("date")
        dates_df = dates_df.join(dates_spf, on="date", how="inner")
        discount_filtered = discount_filtered.filter(
            pl.col("date").is_in(dates_df["date"].to_list())
        )
        survival_probs_filtered = survival_probs_filtered.filter(
            pl.col("date").is_in(dates_df["date"].to_list())
        )

        date_column = survival_probs_filtered.select("date")
        temp_df = discount_filtered.drop("date") * survival_probs_filtered.drop("date")
        temp_df = temp_df.with_columns(date_column)

        risky_duration = risky_duration.join(temp_df, on="date", how="left")
        risky_duration = risky_duration.fill_null(strategy="backward")
        risky_duration = risky_duration.fill_null(strategy="forward")

        risky_duration = risky_duration.with_columns(
            (0.25 * risky_duration.select(pl.exclude("date")).sum_horizontal())
        )

        risky_duration_shifted = risky_duration.select(pl.all().exclude("date")).shift(1)
        cds_spread_shifted = pivot_table.select(pl.all().exclude("date")).shift(1)

        cds_spread_change = pivot_table.select(
            pl.all().exclude("date")
        ) - pivot_table.select(pl.all().exclude("date")).shift(1)

        cds_return = (
            (cds_spread_shifted / 250)
            + (cds_spread_change * risky_duration_shifted.select("sum"))
        ).drop_nulls()

        date_column = risky_duration.select("date").slice(1)
        cds_return = cds_return.with_columns(date_column)

        cds_return_dict[key] = cds_return

    total_elapsed = time.time() - portfolio_start_time
    print(f"  Completed all {total_portfolios} portfolios in {_format_elapsed_time(total_elapsed)}")
    return cds_return_dict


def calculate_monthly_contract_returns(daily_contract_returns=None):
    """
    Calculates monthly returns for individual CDS contracts.
    """
    if daily_contract_returns is None or daily_contract_returns.is_empty():
        return pl.DataFrame(
            {
                "ticker": pl.Series([], dtype=pl.Utf8),
                "tenor": pl.Series([], dtype=pl.Utf8),
                "Month": pl.Series([], dtype=pl.Date),
                "credit_quantile": pl.Series([], dtype=pl.Int64),
                "monthly_return": pl.Series([], dtype=pl.Float64),
            }
        )

    daily_with_month = daily_contract_returns.with_columns(
        pl.col("date").dt.truncate("1mo").alias("Month")
    )

    monthly_returns = (
        daily_with_month.group_by(["ticker", "tenor", "Month", "credit_quantile"])
        .agg(((pl.col("daily_return") + 1).product() - 1).alias("monthly_return"))
        .sort(["ticker", "tenor", "Month"])
    )

    return monthly_returns


def calculate_monthly_returns(daily_returns_dict=None):
    """
    Calculates monthly returns for portfolios with volatility scaling.
    """
    monthly_returns_dict = {}
    fiveY_vol_dict = {}

    for key, df in daily_returns_dict.items():
        if key.startswith("5Y_Q"):
            df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("Month"))

            monthly_returns = (
                df.group_by("Month")
                .agg((pl.col(key) + 1).product() - 1)
                .rename({f"{key}": f"{key} Monthly Return"})
            )

            vol = monthly_returns.select(f"{key} Monthly Return").std().item()
            fiveY_vol_dict[key] = vol

    for key, df in daily_returns_dict.items():
        df = df.with_columns(pl.col("date").dt.truncate("1mo").alias("Month"))

        monthly_returns = (
            df.group_by("Month")
            .agg((pl.col(key) + 1).product() - 1)
            .rename({f"{key}": f"{key} Monthly Return"})
        )

        portfolio_std = monthly_returns.select(f"{key} Monthly Return").std().item()

        vol_target_key = "5Y_Q" + key.split("_Q")[-1]
        target_std = fiveY_vol_dict.get(vol_target_key, None)

        if target_std is not None and portfolio_std > 0:
            scale_factor = target_std / portfolio_std

            monthly_returns = monthly_returns.with_columns(
                (pl.col(f"{key} Monthly Return") * scale_factor).alias(
                    f"{key} Scaled Monthly Return"
                )
            )

        monthly_returns_dict[key] = monthly_returns

    frames = []
    for key, df in monthly_returns_dict.items():
        scaled_col_name = [col for col in df.columns if "Scaled" in col][0]
        small_df = df.select(
            [
                pl.col("Month"),
                pl.col(scaled_col_name).alias(key),
            ]
        )
        frames.append(small_df)

    month_df = frames[0].select("Month")
    value_dfs = [df.select(key) for df, key in zip(frames, monthly_returns_dict.keys())]
    final_df = pl.concat([month_df] + value_dfs, how="horizontal")
    final_df = final_df.sort("Month")

    return final_df


def run_cds_calculation(
    raw_rates=None,
    cds_spreads=None,
    start_date=START_DATE,
    end_date=END_DATE,
    data_dir=DATA_DIR,
):
    """
    Main entry point for CDS return calculation.
    """
    print("\n" + "=" * 60)
    print("Starting CDS return calculation")
    print(f"Date range: {start_date} to {end_date}")
    print("=" * 60)

    overall_start = time.time()

    print("\n1. Loading and filtering CDS data...")
    step_start = time.time()
    contract_data = get_contract_data(start_date, end_date, cds_spreads)
    print(f"   Loaded {len(contract_data):,} contract-date observations")
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    print("\n2. Calculating daily contract-level returns...")
    step_start = time.time()
    daily_contract_returns = calc_cds_return_for_contracts(
        contract_data, raw_rates, start_date, end_date, data_dir=data_dir
    )
    print(f"   Generated {len(daily_contract_returns):,} daily returns")
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    print("\n3. Aggregating to monthly contract returns...")
    step_start = time.time()
    monthly_contract_returns = calculate_monthly_contract_returns(daily_contract_returns)
    print(f"   Generated {len(monthly_contract_returns):,} monthly contract returns")
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    print("\n4. Creating portfolio structure...")
    step_start = time.time()
    portfolio_dict = get_portfolio_dict(start_date, end_date, cds_spreads)
    print(f"   Created {len(portfolio_dict)} portfolios")
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    print("\n5. Calculating daily portfolio returns...")
    step_start = time.time()
    daily_returns_dict = calc_cds_return_for_portfolios(
        portfolio_dict, raw_rates, start_date, end_date, data_dir=data_dir
    )
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    print("\n6. Aggregating and scaling monthly portfolio returns...")
    step_start = time.time()
    monthly_portfolio_returns = calculate_monthly_returns(daily_returns_dict)
    print(f"   Generated returns for {len(monthly_portfolio_returns.columns) - 1} portfolios")
    print(f"   Time: {_format_elapsed_time(time.time() - step_start)}")

    total_elapsed = time.time() - overall_start
    print("\n" + "=" * 60)
    print(f"Total calculation time: {_format_elapsed_time(total_elapsed)}")
    print("=" * 60 + "\n")

    return monthly_contract_returns, monthly_portfolio_returns


def load_portfolio(data_dir=DATA_DIR):
    file_path = Path(data_dir) / "markit_cds_returns.parquet"
    return pd.read_parquet(file_path)


def load_contract_returns(data_dir=DATA_DIR):
    file_path = Path(data_dir) / "markit_cds_contract_returns.parquet"
    return pd.read_parquet(file_path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\nCDS Return Calculation Script")
    print(f"Data directory: {DATA_DIR}")

    print("\nLoading interest rate data...")
    raw_rates = pull_fed_yield_curve.load_fed_yield_curve(data_dir=DATA_DIR)

    print("Loading CDS spread data...")
    cds_spreads = pl.scan_parquet(DATA_DIR / "markit_cds.parquet")

    contract_returns, portfolio_returns = run_cds_calculation(
        raw_rates=raw_rates,
        cds_spreads=cds_spreads,
        start_date=START_DATE,
        end_date=END_DATE,
        data_dir=DATA_DIR,
    )

    print("\nSaving results...")
    contract_returns.write_parquet(DATA_DIR / "markit_cds_contract_returns.parquet")
    print(f"  Saved contract returns")

    portfolio_returns.write_parquet(DATA_DIR / "markit_cds_returns.parquet")
    print(f"  Saved portfolio returns")

    print("\nCalculation complete!")


if __name__ == "__main__":
    main()
