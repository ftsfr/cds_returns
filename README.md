# CDS Returns

Monthly CDS portfolio returns following He, Kelly, and Manela (2017) methodology.

## Overview

This pipeline calculates monthly CDS returns using the He-Kelly formula:

```
CDS Return_t = CDS_{t-1}/250 + ΔCDS_t × RD_{t-1}
```

Where:
- First term: Carry return (daily accrual from previous spread)
- Second term: Spread change × Risky duration

## Data Sources

- **WRDS Markit CDS**: Single-name CDS spreads (USD, XR restructuring clause)
- **Federal Reserve**: Zero-coupon yield curve (GSW model)
- **FRED**: Short-term Treasury yields (3M, 6M)

## Outputs

- `ftsfr_cds_portfolio_returns.parquet`: 20 portfolios (4 tenors × 5 credit quintiles)
- `ftsfr_cds_contract_returns.parquet`: Individual contract-level returns

## Requirements

- WRDS account with access to Markit CDS data
- Python 3.10+

## Setup

1. Copy `.env.example` to `.env` and add your WRDS username
2. Ensure your WRDS password is in `~/.pgpass`
3. Install dependencies: `pip install -r requirements.txt`
4. Run pipeline: `doit`

## References

- He, Z., Kelly, B., & Manela, A. (2017). Intermediary asset pricing: New evidence from many asset classes. *Journal of Financial Economics*.
- Palhares, D. (2013). Cash-flow maturity and risk premia in CDS markets. Working paper.
