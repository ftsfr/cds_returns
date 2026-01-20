"""
Pull Markit CDS data from WRDS.
Code adapted from Kausthub Kesheva.
"""

import sys
from pathlib import Path

sys.path.insert(0, "./src")

import pandas as pd
import wrds
from thefuzz import fuzz

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

WRDS_USERNAME = chartbook.env.get("WRDS_USERNAME")
START_DATE = pd.Timestamp("2001-01-01")
END_DATE = pd.Timestamp("2025-01-01")


def get_cds_data_as_dict(wrds_username=WRDS_USERNAME):
    """
    Fetches CDS data for each year from 2001 to 2023 from WRDS Markit tables.
    """
    db = wrds.Connection(wrds_username=wrds_username)
    cds_data = {}
    for year in range(2001, 2024):
        table_name = f"markit.CDS{year}"
        query = f"""
        SELECT DISTINCT
            date,
            ticker,
            RedCode,
            parspread,
            convspreard,
            tenor,
            country,
            creditdv01,
            riskypv01,
            irdv01,
            rec01,
            dp,
            jtd,
            dtz
        FROM
            {table_name}
        WHERE
            currency = 'USD' AND
            docclause LIKE 'XR%%' AND
            CompositeDepth5Y >= 3 AND
            tenor IN ('1Y', '3Y', '5Y', '7Y', '10Y')
        """
        print(f"  Pulling {year}...")
        cds_data[year] = db.raw_sql(query, date_cols=["date"])
    db.close()
    return cds_data


def combine_cds_data(cds_data: dict) -> pd.DataFrame:
    """
    Combines the CDS data stored in a dictionary into a single DataFrame.
    """
    dataframes = []
    for year, df in cds_data.items():
        if df is not None and not df.empty:
            df_with_year = df.copy()
            df_with_year["year"] = year
            dataframes.append(df_with_year)

    if not dataframes:
        return pd.DataFrame()

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def pull_cds_data(wrds_username=WRDS_USERNAME):
    """Pull and combine CDS data from WRDS."""
    print(">> Pulling Markit CDS data from WRDS...")
    cds_data = get_cds_data_as_dict(wrds_username=wrds_username)
    combined_df = combine_cds_data(cds_data)
    print(f">> Total records: {len(combined_df):,}")
    return combined_df


def pull_markit_red_crsp_link(wrds_username=WRDS_USERNAME):
    """
    Link Markit RED data with CRSP data.
    Returns a table that can be used to link Markit CDS data with CRSP data.
    """
    print(">> Pulling Markit RED-CRSP link from WRDS...")
    conn = wrds.Connection(wrds_username=wrds_username)

    # Get red entity information
    redent = conn.get_table(library="markit", table="redent")

    # Get information from CRSP header table
    crspHdr = conn.raw_sql(
        """SELECT
            permno, permco, hdrcusip, ticker, issuernm
        FROM
            crsp.stksecurityinfohdr
        """
    )
    crspHdr["cusip6"] = crspHdr.hdrcusip.str[:6]
    crspHdr = crspHdr.rename(columns={"ticker": "crspTicker"})

    # First Route - Link with 6-digit cusip
    _cdscrsp1 = pd.merge(
        redent, crspHdr, how="left", left_on="entity_cusip", right_on="cusip6"
    )

    _cdscrsp_cusip = _cdscrsp1.loc[_cdscrsp1.permno.notna()].copy()
    _cdscrsp_cusip["flg"] = "cusip"

    _cdscrsp2 = (
        _cdscrsp1.loc[_cdscrsp1.permno.isna()]
        .copy()
        .drop(
            columns=["permno", "permco", "hdrcusip", "crspTicker", "issuernm", "cusip6"]
        )
    )

    # Second Route - Link with Ticker
    _cdscrsp3 = pd.merge(
        _cdscrsp2, crspHdr, how="left", left_on="ticker", right_on="crspTicker"
    )
    _cdscrsp_ticker = _cdscrsp3.loc[_cdscrsp3.permno.notna()].copy()
    _cdscrsp_ticker["flg"] = "ticker"

    # Consolidate Output
    cdscrsp = pd.concat([_cdscrsp_cusip, _cdscrsp_ticker], ignore_index=True, axis=0)

    # Check similarity ratio of company names
    crspNameLst = cdscrsp.issuernm.str.upper().tolist()
    redNameLst = cdscrsp.shortname.str.upper().tolist()

    nameRatio = []
    for i in range(len(redNameLst)):
        ratio = fuzz.partial_ratio(redNameLst[i], crspNameLst[i])
        nameRatio.append(ratio)

    cdscrsp["nameRatio"] = nameRatio
    conn.close()
    print(f">> Link table records: {len(cdscrsp):,}")
    return cdscrsp


def right_merge_cds_crsp(
    cds_data: pd.DataFrame, cds_crsp_link: pd.DataFrame, ratio_threshold: int = 50
):
    """
    Right merge the CDS data with the CRSP data.
    """
    columns_to_keep = ["redcode", "permno", "permco", "flg", "nameRatio"]
    merged_df = pd.merge(
        cds_data, cds_crsp_link[columns_to_keep], how="right", on="redcode"
    )
    merged_df = merged_df[merged_df["nameRatio"] >= ratio_threshold]
    return merged_df


def load_cds_data(data_dir=DATA_DIR):
    path = data_dir / "markit_cds.parquet"
    return pd.read_parquet(path)


def load_cds_crsp_link(data_dir=DATA_DIR):
    path = data_dir / "markit_red_crsp_link.parquet"
    return pd.read_parquet(path)


def load_cds_subsetted_to_crsp(data_dir=DATA_DIR):
    path = data_dir / "markit_cds_subsetted_to_crsp.parquet"
    return pd.read_parquet(path)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cds_data = pull_cds_data(wrds_username=WRDS_USERNAME)
    cds_data.to_parquet(DATA_DIR / "markit_cds.parquet")
    print(f">> Saved markit_cds.parquet")

    cds_crsp_link = pull_markit_red_crsp_link(wrds_username=WRDS_USERNAME)
    cds_crsp_link.to_parquet(DATA_DIR / "markit_red_crsp_link.parquet")
    print(f">> Saved markit_red_crsp_link.parquet")

    cds_crsp_merged = right_merge_cds_crsp(cds_data, cds_crsp_link, ratio_threshold=50)
    cds_crsp_merged.to_parquet(DATA_DIR / "markit_cds_subsetted_to_crsp.parquet")
    print(f">> Saved markit_cds_subsetted_to_crsp.parquet")


if __name__ == "__main__":
    main()
