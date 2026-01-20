# %%
"""
# CDS Returns Summary

This notebook summarizes the CDS portfolio returns calculated following
He, Kelly, and Manela (2017) methodology.
"""

# %%
import sys
sys.path.insert(0, "./src")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import chartbook

BASE_DIR = chartbook.env.get_project_root()
DATA_DIR = BASE_DIR / "_data"

# %%
"""
## Data Overview

This pipeline produces monthly CDS portfolio returns following He, Kelly, and Manela (2017):

- **20 portfolios**: 4 tenors (3Y, 5Y, 7Y, 10Y) × 5 credit quintiles (Q1-Q5)
- **Tenor**: CDS contract maturity
- **Credit quintile**: Based on 5Y spreads (Q1=safest, Q5=riskiest)

### CDS Return Formula

$$
\\text{CDS Return}_t = \\frac{\\text{CDS}_{t-1}}{250} + \\Delta \\text{CDS}_t \\times \\text{RD}_{t-1}
$$

Where:
- First term: Carry return (daily accrual)
- Second term: Spread change × Risky duration

### Data Sources

- WRDS Markit CDS data (USD contracts, XR restructuring clause)
- Federal Reserve zero-coupon yield curve (GSW model)
- FRED Treasury yields for short maturities
"""

# %%
"""
## Portfolio Returns
"""

# %%
df_portfolio = pd.read_parquet(DATA_DIR / "ftsfr_cds_portfolio_returns.parquet")
print(f"Shape: {df_portfolio.shape}")
print(f"Columns: {df_portfolio.columns.tolist()}")
print(f"\nDate range: {df_portfolio['ds'].min()} to {df_portfolio['ds'].max()}")
print(f"Number of portfolios: {df_portfolio['unique_id'].nunique()}")

# %%
# Show portfolio IDs
print("\nPortfolio IDs:")
for pid in sorted(df_portfolio['unique_id'].unique()):
    print(f"  {pid}")

# %%
"""
### Portfolio Return Statistics
"""

# %%
# Pivot to wide format for analysis
portfolio_wide = df_portfolio.pivot(index='ds', columns='unique_id', values='y')
portfolio_stats = portfolio_wide.describe().T
portfolio_stats['skewness'] = portfolio_wide.skew()
portfolio_stats['kurtosis'] = portfolio_wide.kurtosis()
print(portfolio_stats[['mean', 'std', 'min', 'max', 'skewness', 'kurtosis']].round(4).to_string())

# %%
"""
### Portfolio Return Time Series
"""

# %%
# Plot select portfolios
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Representative portfolios
portfolios = [
    ('5Y_Q1', '5Y CDS - Q1 (Safest)'),
    ('5Y_Q5', '5Y CDS - Q5 (Riskiest)'),
    ('3Y_Q3', '3Y CDS - Q3 (Middle)'),
    ('10Y_Q3', '10Y CDS - Q3 (Middle)')
]

for ax, (port_id, label) in zip(axes.flat, portfolios):
    if port_id in portfolio_wide.columns:
        ax.plot(portfolio_wide.index, portfolio_wide[port_id], alpha=0.7)
        ax.set_title(f'{label}')
        ax.set_ylabel('Monthly Return')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(DATA_DIR.parent / "_output" / "cds_portfolio_returns.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
### Cumulative Returns by Credit Quintile (5Y Tenor)
"""

# %%
fig, ax = plt.subplots(figsize=(12, 6))

quintile_cols = [f'5Y_Q{i}' for i in range(1, 6)]
for col in quintile_cols:
    if col in portfolio_wide.columns:
        cumret = (1 + portfolio_wide[col]).cumprod() - 1
        ax.plot(portfolio_wide.index, cumret, label=col, alpha=0.8)

ax.set_title('Cumulative Returns: 5Y CDS by Credit Quintile')
ax.set_ylabel('Cumulative Return')
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(DATA_DIR.parent / "_output" / "cds_cumulative_returns.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
### Correlation Heatmap
"""

# %%
fig, ax = plt.subplots(figsize=(12, 10))
corr = portfolio_wide.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
            annot_kws={'size': 8})
ax.set_title('CDS Portfolio Correlations')
plt.tight_layout()
plt.savefig(DATA_DIR.parent / "_output" / "cds_correlation.png", dpi=150, bbox_inches='tight')
plt.show()

# %%
"""
## Contract-Level Returns
"""

# %%
df_contracts = pd.read_parquet(DATA_DIR / "ftsfr_cds_contract_returns.parquet")
print(f"Shape: {df_contracts.shape}")
print(f"Columns: {df_contracts.columns.tolist()}")
print(f"\nDate range: {df_contracts['ds'].min()} to {df_contracts['ds'].max()}")
print(f"Number of unique contracts: {df_contracts['unique_id'].nunique()}")

# %%
df_contracts.describe()

# %%
"""
## Data Definitions

### Portfolio Returns (ftsfr_cds_portfolio_returns)

| Variable | Description |
|----------|-------------|
| unique_id | Portfolio identifier (e.g., 5Y_Q1 = 5-year tenor, quintile 1) |
| ds | Month-end date |
| y | Monthly scaled return |

### Contract Returns (ftsfr_cds_contract_returns)

| Variable | Description |
|----------|-------------|
| unique_id | Contract identifier (ticker_tenor) |
| ds | Month-end date |
| y | Monthly return |

### Portfolio Naming Convention

- Format: `{tenor}_Q{quintile}`
- Tenor: 3Y, 5Y, 7Y, 10Y
- Quintile: 1 (safest) to 5 (riskiest)
"""
