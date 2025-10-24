#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/valuation.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 11:34:59 am                                                #
# Modified   : Friday October 24th 2025 12:55:58 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# ============================================================================
# DCF VALUATION AS OF 1/1/1997
# Converting Revenue Forecasts to Enterprise Value
# ============================================================================

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ============================================================================
# 1. LOAD FORECAST DATA
# ============================================================================
print("=" * 80)
print("DCF VALUATION - LOADING FORECAST DATA")
print("=" * 80)

# Load the reconciled forecasts (from previous script output)
forecast_df = pd.read_csv("forecasts_5year_1997_2002.csv")
forecast_df["ds"] = pd.to_datetime(forecast_df["ds"])

print(f"\nForecasts loaded:")
print(f"  Shape: {forecast_df.shape}")
print(f"  Date range: {forecast_df['ds'].min()} to {forecast_df['ds'].max()}")
print(f"  Unique series: {forecast_df['unique_id'].nunique()}")

# ============================================================================
# 2. DCF ASSUMPTIONS
# ============================================================================
print("\n" + "=" * 80)
print("DCF ASSUMPTIONS")
print("=" * 80)

# Valuation date
VALUATION_DATE = pd.Timestamp("1997-01-01")


# Operating assumptions (these should be calibrated to your business)
ASSUMPTIONS = {
    # Profitability margins
    "gross_margin": 0.35,  # Gross profit / Revenue
    "operating_margin": 0.10,  # EBIT / Revenue (typical retail: 5-10%)
    "tax_rate": 0.35,  # Corporate tax rate (1997 US rate)
    # Working capital (as % of revenue change)
    "working_capital_pct": 0.10,  # Change in NWC / Change in Revenue
    # Capital expenditures (as % of revenue)
    "capex_pct": 0.03,  # CapEx / Revenue (retail: 2-4%)
    # Discount rate components
    "risk_free_rate": 0.06,  # 1997 10-year Treasury rate
    "market_risk_premium": 0.08,  # Historical equity risk premium
    "beta": 1.2,  # Retail sector beta
    # Terminal value assumptions
    "terminal_growth_rate": 0.03,  # Long-term growth (GDP + inflation)
    "terminal_fcf_multiple": 12,  # Alternative: P/FCF multiple approach
}

# Calculate WACC (using unlevered firm for simplicity)
WACC = ASSUMPTIONS["risk_free_rate"] + ASSUMPTIONS["beta"] * ASSUMPTIONS["market_risk_premium"]
ASSUMPTIONS["wacc"] = WACC

print("\nKey Assumptions:")
print(f"  Operating Margin: {ASSUMPTIONS['operating_margin']:.1%}")
print(f"  Tax Rate: {ASSUMPTIONS['tax_rate']:.1%}")
print(f"  CapEx % of Revenue: {ASSUMPTIONS['capex_pct']:.1%}")
print(f"  Working Capital % of Rev Change: {ASSUMPTIONS['working_capital_pct']:.1%}")
print(f"  WACC: {WACC:.2%}")
print(f"  Terminal Growth Rate: {ASSUMPTIONS['terminal_growth_rate']:.1%}")

# ============================================================================
# 3. SELECT FORECAST MODEL TO USE
# ============================================================================
print("\n" + "=" * 80)
print("SELECTING FORECAST MODEL")
print("=" * 80)

# Get available model columns (reconciled forecasts preferred)
model_cols = [col for col in forecast_df.columns if col not in ["unique_id", "ds", "level"]]

print(f"\nAvailable forecast models: {model_cols}")

# Use reconciled forecast if available, otherwise base forecast
if "LGBMRegressor/MinTrace_method-ols" in model_cols:
    FORECAST_COL = "LGBMRegressor/MinTrace_method-ols"
    print(f"âœ… Using reconciled forecast: {FORECAST_COL}")
elif "LGBMRegressor" in model_cols:
    FORECAST_COL = "LGBMRegressor"
    print(f"âœ… Using base forecast: {FORECAST_COL}")
else:
    FORECAST_COL = model_cols[0]
    print(f"âš ï¸  Using first available: {FORECAST_COL}")

# ============================================================================
# 4. AGGREGATE TO TOTAL COMPANY LEVEL
# ============================================================================
print("\n" + "=" * 80)
print("AGGREGATING TO TOTAL COMPANY REVENUE")
print("=" * 80)

# Filter to only bottom-level forecasts (store_category combinations)
# to avoid double-counting
if "level" in forecast_df.columns:
    bottom_forecasts = forecast_df[forecast_df["level"] == "bottom"].copy()
else:
    # If no level column, assume all are bottom level
    bottom_forecasts = forecast_df[forecast_df["unique_id"].str.contains("_")].copy()

print(f"\nBottom-level series: {bottom_forecasts['unique_id'].nunique()}")

# Aggregate to total company revenue by week
company_revenue_weekly = bottom_forecasts.groupby("ds")[FORECAST_COL].sum().reset_index()
company_revenue_weekly.columns = ["ds", "revenue"]
company_revenue_weekly = company_revenue_weekly.sort_values("ds")

print(f"\nTotal company weekly revenue forecasts: {len(company_revenue_weekly)} weeks")
print(f"  Average weekly revenue: ${company_revenue_weekly['revenue'].mean():,.0f}")
print(f"  Total 5-year revenue: ${company_revenue_weekly['revenue'].sum():,.0f}")

# ============================================================================
# 5. CONVERT TO ANNUAL CASH FLOWS
# ============================================================================
print("\n" + "=" * 80)
print("CONVERTING TO ANNUAL FREE CASH FLOWS")
print("=" * 80)

# Extract year from date
company_revenue_weekly["year"] = company_revenue_weekly["ds"].dt.year

# Aggregate to annual revenue
annual_revenue = company_revenue_weekly.groupby("year")["revenue"].sum().reset_index()
annual_revenue.columns = ["year", "revenue"]

print("\nAnnual Revenue Forecasts:")
print(annual_revenue.to_string(index=False))


# Calculate Free Cash Flow for each year
def calculate_fcf(revenue, prior_revenue, assumptions):
    """
    Calculate Free Cash Flow from revenue

    FCF = Revenue
          * Operating Margin
          * (1 - Tax Rate)
          - CapEx
          - Change in Net Working Capital
    """
    # NOPAT (Net Operating Profit After Tax)
    ebit = revenue * assumptions["operating_margin"]
    nopat = ebit * (1 - assumptions["tax_rate"])

    # Capital Expenditures
    capex = revenue * assumptions["capex_pct"]

    # Change in Net Working Capital
    revenue_change = revenue - prior_revenue
    nwc_change = revenue_change * assumptions["working_capital_pct"]

    # Free Cash Flow
    fcf = nopat - capex - nwc_change

    return {
        "revenue": revenue,
        "ebit": ebit,
        "nopat": nopat,
        "capex": capex,
        "nwc_change": nwc_change,
        "fcf": fcf,
    }


# Build DCF model
dcf_model = []
prior_revenue = 0

for idx, row in annual_revenue.iterrows():
    year = row["year"]
    revenue = row["revenue"]

    # Calculate FCF components
    fcf_calc = calculate_fcf(revenue, prior_revenue, ASSUMPTIONS)

    # Calculate discount period (mid-year convention)
    years_from_valuation = year - VALUATION_DATE.year + 0.5

    # Calculate discount factor
    discount_factor = 1 / (1 + WACC) ** years_from_valuation

    # Calculate present value
    pv_fcf = fcf_calc["fcf"] * discount_factor

    # Store results
    dcf_model.append(
        {
            "year": year,
            "years_from_val": years_from_valuation,
            **fcf_calc,
            "discount_factor": discount_factor,
            "pv_fcf": pv_fcf,
        }
    )

    prior_revenue = revenue

dcf_df = pd.DataFrame(dcf_model)

print("\n" + "=" * 80)
print("DCF MODEL - ANNUAL PROJECTIONS")
print("=" * 80)
print(dcf_df.to_string(index=False))

# ============================================================================
# 6. CALCULATE TERMINAL VALUE
# ============================================================================
print("\n" + "=" * 80)
print("TERMINAL VALUE CALCULATION")
print("=" * 80)

# Get terminal year FCF (2002)
terminal_fcf = dcf_df.iloc[-1]["fcf"]
terminal_year = dcf_df.iloc[-1]["year"]

# Method 1: Perpetuity Growth Model
# TV = FCF_terminal * (1 + g) / (WACC - g)
terminal_value_perpetuity = (terminal_fcf * (1 + ASSUMPTIONS["terminal_growth_rate"])) / (
    WACC - ASSUMPTIONS["terminal_growth_rate"]
)

# Method 2: Exit Multiple Method (alternative)
# TV = FCF_terminal * Exit_Multiple
terminal_value_multiple = terminal_fcf * ASSUMPTIONS["terminal_fcf_multiple"]

# Use perpetuity method as primary
terminal_value = terminal_value_perpetuity

# Discount terminal value to present
years_to_terminal = terminal_year - VALUATION_DATE.year + 0.5
discount_factor_terminal = 1 / (1 + WACC) ** years_to_terminal
pv_terminal_value = terminal_value * discount_factor_terminal

print(f"\nTerminal Year (2002) FCF: ${terminal_fcf:,.0f}")
print(f"\nMethod 1 - Perpetuity Growth:")
print(f"  Terminal Value: ${terminal_value_perpetuity:,.0f}")
print(f"  PV of Terminal Value: ${pv_terminal_value:,.0f}")
print(f"\nMethod 2 - Exit Multiple ({ASSUMPTIONS['terminal_fcf_multiple']}x):")
print(f"  Terminal Value: ${terminal_value_multiple:,.0f}")
print(f"  PV of Terminal Value: ${terminal_value_multiple * discount_factor_terminal:,.0f}")

# ============================================================================
# 7. CALCULATE ENTERPRISE VALUE
# ============================================================================
print("\n" + "=" * 80)
print("ENTERPRISE VALUE CALCULATION")
print("=" * 80)

# Sum of PV of projected FCFs
pv_forecast_period = dcf_df["pv_fcf"].sum()

# Enterprise Value = PV of Forecast Period + PV of Terminal Value
enterprise_value = pv_forecast_period + pv_terminal_value

print(f"\nPV of Forecast Period FCFs (1997-2002): ${pv_forecast_period:,.0f}")
print(f"PV of Terminal Value: ${pv_terminal_value:,.0f}")
print(f"\n{'='*80}")
print(f"ENTERPRISE VALUE (as of {VALUATION_DATE.date()}): ${enterprise_value:,.0f}")
print(f"{'='*80}")

# Value breakdown
forecast_period_pct = pv_forecast_period / enterprise_value * 100
terminal_value_pct = pv_terminal_value / enterprise_value * 100

print(f"\nValue Composition:")
print(f"  Forecast Period: {forecast_period_pct:.1f}%")
print(f"  Terminal Value: {terminal_value_pct:.1f}%")

# ============================================================================
# 8. EQUITY VALUE (if you have debt/cash information)
# ============================================================================
print("\n" + "=" * 80)
print("BRIDGE TO EQUITY VALUE")
print("=" * 80)

# These would need to be provided from balance sheet data
# Using example values - replace with actual
NET_DEBT = 0  # Total Debt - Cash (as of valuation date)
MINORITY_INTEREST = 0
OTHER_ADJUSTMENTS = 0

equity_value = enterprise_value - NET_DEBT - MINORITY_INTEREST + OTHER_ADJUSTMENTS

print(f"\nEnterprise Value: ${enterprise_value:,.0f}")
print(f"Less: Net Debt: ${NET_DEBT:,.0f}")
print(f"Less: Minority Interest: ${MINORITY_INTEREST:,.0f}")
print(f"Plus/Less: Other Adjustments: ${OTHER_ADJUSTMENTS:,.0f}")
print(f"\n{'='*80}")
print(f"EQUITY VALUE (as of {VALUATION_DATE.date()}): ${equity_value:,.0f}")
print(f"{'='*80}")

# ============================================================================
# 9. SENSITIVITY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("SENSITIVITY ANALYSIS")
print("=" * 80)

# Sensitivity to WACC and Terminal Growth Rate
wacc_range = np.arange(WACC - 0.02, WACC + 0.025, 0.005)
tgr_range = np.arange(
    ASSUMPTIONS["terminal_growth_rate"] - 0.01, ASSUMPTIONS["terminal_growth_rate"] + 0.015, 0.005
)

sensitivity_results = []

for wacc_test in wacc_range:
    for tgr_test in tgr_range:
        # Skip if WACC <= terminal growth (invalid)
        if wacc_test <= tgr_test:
            continue

        # Recalculate PV of forecast period
        test_pv_fcf = sum(
            row["fcf"] / (1 + wacc_test) ** row["years_from_val"] for _, row in dcf_df.iterrows()
        )

        # Recalculate terminal value
        test_tv = (terminal_fcf * (1 + tgr_test)) / (wacc_test - tgr_test)
        test_pv_tv = test_tv / (1 + wacc_test) ** years_to_terminal

        # Recalculate enterprise value
        test_ev = test_pv_fcf + test_pv_tv

        sensitivity_results.append(
            {"wacc": wacc_test, "terminal_growth": tgr_test, "enterprise_value": test_ev}
        )

sensitivity_df = pd.DataFrame(sensitivity_results)

# Create sensitivity table
sensitivity_table = sensitivity_df.pivot(
    index="terminal_growth", columns="wacc", values="enterprise_value"
)

print("\nEnterprise Value Sensitivity (in $ millions):")
print("Rows: Terminal Growth Rate | Columns: WACC")
print((sensitivity_table / 1_000_000).round(2))

# ============================================================================
# 10. VISUALIZATIONS
# ============================================================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS")
print("=" * 80)

fig = plt.figure(figsize=(16, 12))

# 1. Annual Revenue Forecast
ax1 = plt.subplot(3, 3, 1)
ax1.bar(dcf_df["year"], dcf_df["revenue"] / 1_000_000, color="steelblue", alpha=0.7)
ax1.set_title("Annual Revenue Forecast", fontsize=12, fontweight="bold")
ax1.set_xlabel("Year")
ax1.set_ylabel("Revenue ($M)")
ax1.grid(axis="y", alpha=0.3)

# 2. Free Cash Flow
ax2 = plt.subplot(3, 3, 2)
colors = ["green" if x > 0 else "red" for x in dcf_df["fcf"]]
ax2.bar(dcf_df["year"], dcf_df["fcf"] / 1_000_000, color=colors, alpha=0.7)
ax2.set_title("Annual Free Cash Flow", fontsize=12, fontweight="bold")
ax2.set_xlabel("Year")
ax2.set_ylabel("FCF ($M)")
ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
ax2.grid(axis="y", alpha=0.3)

# 3. FCF Components Waterfall (for 2002)
ax3 = plt.subplot(3, 3, 3)
last_year = dcf_df.iloc[-1]
components = ["Revenue", "EBIT", "NOPAT", "CapEx", "NWC Δ", "FCF"]
values = [
    last_year["revenue"] / 1_000_000,
    last_year["ebit"] / 1_000_000,
    last_year["nopat"] / 1_000_000,
    -last_year["capex"] / 1_000_000,
    -last_year["nwc_change"] / 1_000_000,
    last_year["fcf"] / 1_000_000,
]
colors_comp = ["steelblue", "lightblue", "green", "red", "red", "darkgreen"]
ax3.bar(range(len(components)), values, color=colors_comp, alpha=0.7)
ax3.set_xticks(range(len(components)))
ax3.set_xticklabels(components, rotation=45, ha="right")
ax3.set_title(f'FCF Components ({int(last_year["year"])})', fontsize=12, fontweight="bold")
ax3.set_ylabel("Amount ($M)")
ax3.grid(axis="y", alpha=0.3)

# 4. Present Value Contributions
ax4 = plt.subplot(3, 3, 4)
ax4.bar(
    dcf_df["year"], dcf_df["pv_fcf"] / 1_000_000, color="purple", alpha=0.7, label="Forecast Period"
)
ax4.bar(
    [terminal_year],
    [pv_terminal_value / 1_000_000],
    color="orange",
    alpha=0.7,
    label="Terminal Value",
)
ax4.set_title("Present Value Contributions", fontsize=12, fontweight="bold")
ax4.set_xlabel("Year")
ax4.set_ylabel("PV ($M)")
ax4.legend()
ax4.grid(axis="y", alpha=0.3)

# 5. Value Bridge
ax5 = plt.subplot(3, 3, 5)
bridge_labels = ["Forecast\nPeriod", "Terminal\nValue", "Enterprise\nValue"]
bridge_values = [pv_forecast_period / 1_000_000, pv_terminal_value / 1_000_000, 0]
bridge_cumulative = np.cumsum(bridge_values)
ax5.bar(
    range(len(bridge_labels)), bridge_values, color=["purple", "orange", "darkgreen"], alpha=0.7
)
for i in range(len(bridge_labels) - 1):
    ax5.plot([i, i + 1], [bridge_cumulative[i], bridge_cumulative[i]], "k--", alpha=0.3)
ax5.axhline(y=enterprise_value / 1_000_000, color="green", linestyle="-", linewidth=2, label="EV")
ax5.set_xticks(range(len(bridge_labels)))
ax5.set_xticklabels(bridge_labels)
ax5.set_title("Enterprise Value Bridge", fontsize=12, fontweight="bold")
ax5.set_ylabel("Value ($M)")
ax5.legend()
ax5.grid(axis="y", alpha=0.3)

# 6. Sensitivity Heatmap (WACC vs Terminal Growth)
ax6 = plt.subplot(3, 3, 6)
sns.heatmap(
    sensitivity_table / 1_000_000,
    annot=True,
    fmt=".1f",
    cmap="RdYlGn",
    ax=ax6,
    cbar_kws={"label": "EV ($M)"},
)
ax6.set_title("Sensitivity: WACC vs Terminal Growth", fontsize=12, fontweight="bold")
ax6.set_xlabel("WACC")
ax6.set_ylabel("Terminal Growth Rate")

# 7. Operating Margins Over Time
ax7 = plt.subplot(3, 3, 7)
ax7.plot(
    dcf_df["year"],
    dcf_df["ebit"] / dcf_df["revenue"] * 100,
    marker="o",
    color="steelblue",
    linewidth=2,
    label="Operating Margin",
)
ax7.axhline(
    y=ASSUMPTIONS["operating_margin"] * 100,
    color="red",
    linestyle="--",
    alpha=0.5,
    label="Assumed Margin",
)
ax7.set_title("Operating Margin Trend", fontsize=12, fontweight="bold")
ax7.set_xlabel("Year")
ax7.set_ylabel("Margin (%)")
ax7.legend()
ax7.grid(alpha=0.3)

# 8. Discount Factors
ax8 = plt.subplot(3, 3, 8)
ax8.plot(dcf_df["year"], dcf_df["discount_factor"], marker="o", color="darkred", linewidth=2)
ax8.set_title("Discount Factors Over Time", fontsize=12, fontweight="bold")
ax8.set_xlabel("Year")
ax8.set_ylabel("Discount Factor")
ax8.grid(alpha=0.3)

# 9. Weekly Revenue Trend
ax9 = plt.subplot(3, 3, 9)
ax9.plot(
    company_revenue_weekly["ds"],
    company_revenue_weekly["revenue"] / 1000,
    color="steelblue",
    alpha=0.7,
    linewidth=1,
)
ax9.set_title("Weekly Revenue Forecast (1997-2002)", fontsize=12, fontweight="bold")
ax9.set_xlabel("Date")
ax9.set_ylabel("Revenue ($K)")
ax9.grid(alpha=0.3)

plt.suptitle(
    f"DCF Valuation Dashboard - Enterprise Value: ${enterprise_value/1_000_000:.1f}M (as of {VALUATION_DATE.date()})",
    fontsize=14,
    fontweight="bold",
    y=0.995,
)
plt.tight_layout()
plt.savefig("dcf_valuation_dashboard.png", dpi=300, bbox_inches="tight")
print("âœ… Dashboard saved to: dcf_valuation_dashboard.png")

# ============================================================================
# 11. EXPORT RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("EXPORTING RESULTS")
print("=" * 80)

# Export DCF model
dcf_df.to_csv("dcf_model_annual.csv", index=False)
print("âœ… DCF model saved to: dcf_model_annual.csv")

# Export sensitivity analysis
sensitivity_table.to_csv("dcf_sensitivity_analysis.csv")
print("âœ… Sensitivity analysis saved to: dcf_sensitivity_analysis.csv")

# Create summary report
summary_report = {
    "Valuation Date": VALUATION_DATE.date(),
    "Forecast Period": "1997-2002",
    "WACC": f"{WACC:.2%}",
    "Terminal Growth Rate": f'{ASSUMPTIONS["terminal_growth_rate"]:.2%}',
    "Operating Margin": f'{ASSUMPTIONS["operating_margin"]:.2%}',
    "Tax Rate": f'{ASSUMPTIONS["tax_rate"]:.2%}',
    "Total Forecast Revenue": f'${dcf_df["revenue"].sum():,.0f}',
    "Total Forecast FCF": f'${dcf_df["fcf"].sum():,.0f}',
    "PV Forecast Period": f"${pv_forecast_period:,.0f}",
    "PV Terminal Value": f"${pv_terminal_value:,.0f}",
    "Enterprise Value": f"${enterprise_value:,.0f}",
    "Equity Value": f"${equity_value:,.0f}",
}

summary_df = pd.DataFrame([summary_report]).T
summary_df.columns = ["Value"]
summary_df.to_csv("dcf_valuation_summary.csv")
print("âœ… Summary report saved to: dcf_valuation_summary.csv")

print("\n" + "=" * 80)
print("âœ… DCF VALUATION COMPLETE!")
print("=" * 80)
print(f"\nFINAL VALUATION (as of {VALUATION_DATE.date()}):")
print(f"  Enterprise Value: ${enterprise_value:,.0f}")
print(f"  Equity Value: ${equity_value:,.0f}")
print("\nKey Outputs:")
print("  1. dcf_model_annual.csv - Detailed annual projections")
print("  2. dcf_sensitivity_analysis.csv - Sensitivity table")
print("  3. dcf_valuation_summary.csv - Summary report")
print("  4. dcf_valuation_dashboard.png - Visual dashboard")
