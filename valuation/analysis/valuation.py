#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/valuation.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 01:00:33 pm                                                #
# Modified   : Saturday October 25th 2025 05:35:16 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""DCF Valuation Class using Financial Statements."""

from typing import Dict, Optional, Tuple

from dataclasses import dataclass
from pathlib import Path
import warnings

from IPython.display import display
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from valuation.analysis.financials import Financials
from valuation.core.dataclass import DataClass

warnings.filterwarnings("ignore")
# ------------------------------------------------------------------------------------------------ #
REPORT_DIRECTORY = Path("report")
# ------------------------------------------------------------------------------------------------ #


@dataclass
class ValuationAssumptions(DataClass):
    """Holds core assumptions for the DCF valuation."""

    # WACC Components (used if WACC is not directly provided)
    risk_free_rate: float = 0.06  # Example: 1997 10-year Treasury
    market_risk_premium: float = 0.05  # Example: Historical average
    beta: float = 0.6  # Example: Retail sector beta

    # Terminal Value Assumptions
    terminal_growth_rate: float = 0.03  # Perpetual growth rate after forecast period
    terminal_fcf_multiple: float = 12  # Exit multiple (alternative TV method)

    # Minority Interest and Other Adjustments
    minority_interest: float = 0.0  # Minority interest to subtract from EV
    other_adjustments: float = 0.0  # Other adjustments to bridge EV to equity value

    wacc: Optional[float] = None  # Weighted average cost of capital

    def __post_init__(self):
        self.wacc = self.risk_free_rate + self.beta * self.market_risk_premium


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ValuationAnalysisResults(DataClass):
    """Dataclass to hold valuation analysis results."""

    enterprise_value: float
    equity_value: float
    pv_forecast_period: float
    pv_terminal_value: float
    wacc: float
    terminal_growth_rate: float


# ------------------------------------------------------------------------------------------------ #
class ValuationDCF:
    """
    Discounted Cash Flow (DCF) Valuation Model.

    Converts revenue forecasts into enterprise value using financial statement
    data and standard DCF methodology.
    """

    def __init__(
        self,
        financials: Financials,
        assumptions: ValuationAssumptions,
        forecast_df: pd.DataFrame,
        forecast_col: str,
        valuation_date: pd.Timestamp,
    ):
        """
        Initialize DCF Valuation.

        Args:
            financials: Financials dataclass object with company financial data
            forecast_df: DataFrame with revenue forecasts
            forecast_col: Column name to use for valuation
            valuation_date: Date of valuation
            wacc: Weighted average cost of capital (calculated from financials if None)
            terminal_growth_rate: Long-term growth rate for terminal value
            terminal_fcf_multiple: Alternative exit multiple for terminal value
            net_debt: Total debt minus cash
            minority_interest: Minority interest to subtract from EV
            other_adjustments: Other adjustments to bridge EV to equity value
        """
        self.financials = financials
        self.assumptions = assumptions
        self.forecast_df = forecast_df.copy()
        self.forecast_col = forecast_col
        self.valuation_date = valuation_date

        # Initialize containers
        self.company_revenue_weekly_df = None
        self.annual_revenue = None
        self.dcf_df = None
        self.terminal_value = None
        self.pv_terminal_value = None
        self.enterprise_value = None
        self.equity_value = None
        self.sensitivity_df = None
        self.net_debt = self.financials.total_debt - self.financials.cash_and_equivalents

        logger.info("=" * 80)
        logger.info("DCF VALUATION INITIALIZED")
        logger.info("=" * 80)
        logger.info(f"Valuation Date: {self.valuation_date.date()}")
        logger.info(f"Forecast Column: {self.forecast_col}")
        logger.info(f"WACC: {self.assumptions.wacc:.2%}")

    def aggregate_revenue(self) -> None:
        """
        Aggregate bottom-level forecasts to total company revenue.

        Returns:
            DataFrame with weekly company revenue
        """
        logger.info("=" * 80)
        logger.info("AGGREGATING TO TOTAL COMPANY REVENUE")
        logger.info("=" * 80)

        is_bottom_level = self.forecast_df["unique_id"].str.contains("/")

        bottom_panel = self.forecast_df[is_bottom_level].copy()

        self.company_revenue_weekly_df = (
            bottom_panel.groupby("ds")[[self.forecast_col]].sum().reset_index()
        )
        self.company_revenue_weekly_df.columns = ["ds", "revenue"]
        self.company_revenue_weekly_df = self.company_revenue_weekly_df.sort_values("ds")

        logger.info(
            f"Total company weekly revenue forecasts: {len(self.company_revenue_weekly_df)} weeks"
        )
        logger.info(
            f"Average weekly revenue: ${self.company_revenue_weekly_df['revenue'].mean():,.0f}"
        )
        logger.info(
            f"Total 5-year revenue: ${self.company_revenue_weekly_df['revenue'].sum():,.0f}"
        )

    def convert_to_annual(self) -> pd.DataFrame:
        """
        Convert weekly revenue to annual revenue.

        Returns:
            DataFrame with annual revenue
        """
        logger.info("=" * 80)
        logger.info("CONVERTING TO ANNUAL REVENUE")
        logger.info("=" * 80)

        if self.company_revenue_weekly_df is None:
            self.aggregate_revenue()
        if self.company_revenue_weekly_df is None:
            raise ValueError("Company revenue weekly data is not available.")
        # Extract year from date
        self.company_revenue_weekly_df["year"] = self.company_revenue_weekly_df["ds"].dt.year

        # Aggregate to annual revenue
        self.annual_revenue = (
            self.company_revenue_weekly_df.groupby("year")["revenue"].sum().reset_index()
        )
        self.annual_revenue.columns = ["year", "revenue"]

        logger.info("Annual Revenue Forecasts:")
        for _, row in self.annual_revenue.iterrows():
            logger.info(f"  {int(row['year'])}: {row['revenue']:,.0f}")

        return self.annual_revenue

    def calculate_fcf(self, revenue: float, prior_revenue: float) -> Dict[str, float]:
        """
        Calculate Free Cash Flow from revenue.

        FCF = NOPAT - CapEx - Change in NWC

        Args:
            revenue: Current period revenue
            prior_revenue: Prior period revenue

        Returns:
            Dictionary with FCF components
        """
        # NOPAT (Net Operating Profit After Tax)
        ebit = revenue * self.financials.operating_margin
        nopat = ebit * (1 - self.financials.tax_rate)
        # Capital Expenditures
        capex = revenue * self.financials.capex_pct

        # Change in Net Working Capital
        revenue_change = revenue - prior_revenue
        nwc_change = revenue_change * self.financials.working_capital_pct

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

    def build_dcf_model(self) -> pd.DataFrame:
        """
        Build the DCF model with annual projections.

        Returns:
            DataFrame with DCF model
        """
        logger.info("=" * 80)
        logger.info("BUILDING DCF MODEL")
        logger.info("=" * 80)

        if self.annual_revenue is None:
            self.convert_to_annual()

        dcf_model = []
        prior_revenue = 0
        wacc = self.assumptions.wacc if self.assumptions.wacc is not None else 0.1

        for idx, row in self.annual_revenue.iterrows():  # type: ignore
            year = row["year"].astype(int)
            revenue = row["revenue"]

            # Calculate FCF components
            fcf_calc = self.calculate_fcf(revenue, prior_revenue)

            # Calculate discount period (mid-year convention)
            years_from_valuation = year - self.valuation_date.year + 0.5

            # Calculate discount factor
            discount_factor = 1 / (1.0 + wacc) ** years_from_valuation

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

        formatter = {
            "year": "{:,.0f}",
            "years_from_val": "{:.1f}",
            "revenue": "${:,.0f}",
            "ebit": "${:,.0f}",
            "nopat": "${:,.0f}",
            "capex": "${:,.0f}",
            "nwc_change": "{:.1f}",
            "fcf": "${:,.0f}",
            "discount_factor": "{:.4f}",
            "pv_fcf": "${:,.0f}",
        }

        self.dcf_df = pd.DataFrame(dcf_model)
        dcf_df_formatted = self.dcf_df.style.format(formatter)
        display(dcf_df_formatted)

        logger.info("DCF Model - Annual Projections:")
        logger.info(f"\n{self.dcf_df.to_string(index=False)}")

        return self.dcf_df

    def calculate_terminal_value(self) -> Tuple[float, float]:
        """
        Calculate terminal value and its present value.

        Returns:
            Tuple of (terminal_value, pv_terminal_value)
        """
        logger.info("=" * 80)
        logger.info("CALCULATING TERMINAL VALUE")
        logger.info("=" * 80)

        if self.dcf_df is None:
            self.build_dcf_model()

        # Get terminal year FCF
        terminal_fcf = self.dcf_df.iloc[-1]["fcf"]  # type: ignore
        terminal_year = self.dcf_df.iloc[-1]["year"]  # type: ignore
        wacc = self.assumptions.wacc if self.assumptions.wacc is not None else 0.1
        g = self.assumptions.terminal_growth_rate

        # Method 1: Perpetuity Growth Model
        # TV = FCF_terminal * (1 + g) / (WACC - g)
        terminal_value_perpetuity = (
            (terminal_fcf * (1 + g)) / (self.assumptions.wacc - g)
            if self.assumptions.wacc is not None
            else 0.1
        )

        # Method 2: Exit Multiple Method
        terminal_value_multiple = terminal_fcf * self.assumptions.terminal_fcf_multiple

        # Use perpetuity method as primary
        self.terminal_value = terminal_value_perpetuity

        # Discount terminal value to present
        years_to_terminal = terminal_year - self.valuation_date.year + 0.5
        discount_factor_terminal = 1 / (1 + wacc) ** years_to_terminal
        self.pv_terminal_value = self.terminal_value * discount_factor_terminal

        logger.info(f"Terminal Year ({int(terminal_year)}) FCF: ${terminal_fcf:,.0f}")
        logger.info(f"\nMethod 1 - Perpetuity Growth:")
        logger.info(f"  Terminal Value: ${terminal_value_perpetuity:,.0f}")
        logger.info(f"  PV of Terminal Value: ${self.pv_terminal_value:,.0f}")
        logger.info(f"\nMethod 2 - Exit Multiple ({self.assumptions.terminal_fcf_multiple}x):")
        logger.info(f"  Terminal Value: ${terminal_value_multiple:,.0f}")
        logger.info(f"  PV: ${terminal_value_multiple * discount_factor_terminal:,.0f}")

        return self.terminal_value, self.pv_terminal_value

    def calculate_enterprise_value(self) -> float:
        """
        Calculate enterprise value.

        Returns:
            Enterprise value
        """
        logger.info("=" * 80)
        logger.info("CALCULATING ENTERPRISE VALUE")
        logger.info("=" * 80)

        if self.pv_terminal_value is None:
            self.calculate_terminal_value()

        # Sum of PV of projected FCFs
        if self.dcf_df is None:
            self.build_dcf_model()
        if self.dcf_df is None:
            raise ValueError("DCF model data is not available.")
        pv_forecast_period = self.dcf_df["pv_fcf"].sum()

        # Enterprise Value = PV of Forecast Period + PV of Terminal Value
        self.enterprise_value = pv_forecast_period + self.pv_terminal_value

        # Value breakdown
        forecast_period_pct = pv_forecast_period / self.enterprise_value * 100
        terminal_value_pct = self.pv_terminal_value / self.enterprise_value * 100

        logger.info(f"PV of Forecast Period FCFs: ${pv_forecast_period:,.0f}")
        logger.info(f"PV of Terminal Value: ${self.pv_terminal_value:,.0f}")
        logger.info("=" * 80)
        logger.info(f"ENTERPRISE VALUE: ${self.enterprise_value:,.0f}")
        logger.info("=" * 80)
        logger.info(f"\nValue Composition:")
        logger.info(f"  Forecast Period: {forecast_period_pct:.1f}%")
        logger.info(f"  Terminal Value: {terminal_value_pct:.1f}%")

        return self.enterprise_value

    def calculate_equity_value(self) -> float:
        """
        Calculate equity value from enterprise value.

        Returns:
            Equity value
        """
        logger.info("=" * 80)
        logger.info("BRIDGING TO EQUITY VALUE")
        logger.info("=" * 80)

        if self.enterprise_value is None:
            self.calculate_enterprise_value()

        if self.enterprise_value is None:
            raise ValueError("Enterprise value is not available.")

        self.equity_value = (
            self.enterprise_value
            - self.net_debt
            - self.assumptions.minority_interest
            + self.assumptions.other_adjustments
        )

        logger.info(f"Enterprise Value: ${self.enterprise_value:,.0f}")
        logger.info(f"Less: Net Debt: ${self.net_debt:,.0f}")
        logger.info(f"Less: Minority Interest: ${self.assumptions.minority_interest:,.0f}")
        logger.info(f"Plus/Less: Other Adjustments: ${self.assumptions.other_adjustments:,.0f}")
        logger.info("=" * 80)
        logger.info(f"EQUITY VALUE: ${self.equity_value:,.0f}")
        logger.info("=" * 80)

        return self.equity_value

    def run_sensitivity_analysis(
        self, wacc_range: Optional[np.ndarray] = None, tgr_range: Optional[np.ndarray] = None
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis on WACC and terminal growth rate.

        Args:
            wacc_range: Array of WACC values to test
            tgr_range: Array of terminal growth rates to test

        Returns:
            DataFrame with sensitivity results
        """
        logger.info("=" * 80)
        logger.info("RUNNING SENSITIVITY ANALYSIS")
        logger.info("=" * 80)

        if self.dcf_df is None:
            self.build_dcf_model()
            self.calculate_terminal_value()

        wacc = self.assumptions.wacc if self.assumptions.wacc is not None else 0.1
        g = self.assumptions.terminal_growth_rate

        # Default ranges
        if wacc_range is None:
            wacc_range = np.arange(wacc - 0.02, wacc + 0.025, 0.005)
        if tgr_range is None:
            tgr_range = np.arange(g - 0.01, g + 0.015, 0.005)

        if self.dcf_df is None:
            raise ValueError("DCF model data is not available.")

        terminal_fcf = self.dcf_df.iloc[-1]["fcf"]
        terminal_year = self.dcf_df.iloc[-1]["year"]
        years_to_terminal = terminal_year - self.valuation_date.year + 0.5

        sensitivity_results = []

        for wacc_test in wacc_range:
            for tgr_test in tgr_range:
                # Skip if WACC <= terminal growth (invalid)
                if wacc_test <= tgr_test:
                    continue

                # Recalculate PV of forecast period
                test_pv_fcf = sum(
                    row["fcf"] / (1 + wacc_test) ** row["years_from_val"]
                    for _, row in self.dcf_df.iterrows()
                )

                # Recalculate terminal value
                test_tv = (terminal_fcf * (1 + tgr_test)) / (wacc_test - tgr_test)
                test_pv_tv = test_tv / (1 + wacc_test) ** years_to_terminal

                # Recalculate enterprise value
                test_ev = test_pv_fcf + test_pv_tv

                sensitivity_results.append(
                    {"wacc": wacc_test, "terminal_growth": tgr_test, "enterprise_value": test_ev}
                )

        self.sensitivity_df = pd.DataFrame(sensitivity_results)

        # Create sensitivity table
        sensitivity_table = self.sensitivity_df.pivot(
            index="terminal_growth", columns="wacc", values="enterprise_value"
        )

        logger.info("Enterprise Value Sensitivity (in $ millions):")
        logger.info("Rows: Terminal Growth Rate | Columns: WACC")
        logger.info(f"\n{(sensitivity_table / 1_000_000).round(2)}")

        return self.sensitivity_df

    def create_visualizations(
        self, output_path: str = REPORT_DIRECTORY / "dcf_valuation_dashboard.png"
    ) -> None:
        """
        Create comprehensive DCF visualization dashboard.

        Args:
            output_path: Path to save the dashboard
        """
        logger.info("=" * 80)
        logger.info("CREATING VISUALIZATIONS")
        logger.info("=" * 80)

        if self.sensitivity_df is None:
            self.run_sensitivity_analysis()

        fig = plt.figure(figsize=(16, 12))

        if self.dcf_df is None:
            self.build_dcf_model()
        if self.dcf_df is None:
            raise ValueError("DCF model data is not available.")
        # 1. Annual Revenue Forecast
        ax1 = plt.subplot(3, 3, 1)
        ax1.bar(
            self.dcf_df["year"], self.dcf_df["revenue"] / 1_000_000, color="steelblue", alpha=0.7
        )
        ax1.set_title("Annual Revenue Forecast", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Year")
        ax1.set_ylabel("Revenue ($M)")
        ax1.grid(axis="y", alpha=0.3)

        # 2. Free Cash Flow
        ax2 = plt.subplot(3, 3, 2)
        colors = ["green" if x > 0 else "red" for x in self.dcf_df["fcf"]]
        ax2.bar(self.dcf_df["year"], self.dcf_df["fcf"] / 1_000_000, color=colors, alpha=0.7)
        ax2.set_title("Annual Free Cash Flow", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Year")
        ax2.set_ylabel("FCF ($M)")
        ax2.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax2.grid(axis="y", alpha=0.3)

        # 3. FCF Components Waterfall
        ax3 = plt.subplot(3, 3, 3)
        last_year = self.dcf_df.iloc[-1]
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

        if self.pv_terminal_value is None:
            self.calculate_terminal_value()
        if self.pv_terminal_value is None:
            raise ValueError("Present value of terminal value is not available.")
        # 4. Present Value Contributions
        ax4 = plt.subplot(3, 3, 4)
        terminal_year = self.dcf_df.iloc[-1]["year"]
        ax4.bar(
            self.dcf_df["year"],
            self.dcf_df["pv_fcf"] / 1_000_000,
            color="purple",
            alpha=0.7,
            label="Forecast Period",
        )
        ax4.bar(
            [terminal_year],
            [self.pv_terminal_value / 1_000_000],
            color="orange",
            alpha=0.7,
            label="Terminal Value",
        )
        ax4.set_title("Present Value Contributions", fontsize=12, fontweight="bold")
        ax4.set_xlabel("Year")
        ax4.set_ylabel("PV ($M)")
        ax4.legend()
        ax4.grid(axis="y", alpha=0.3)

        # Enterprise Value line
        if self.enterprise_value is None:
            self.calculate_enterprise_value()

        if self.enterprise_value is None:
            raise ValueError("Enterprise value is not available.")

        # 5. Value Bridge
        ax5 = plt.subplot(3, 3, 5)
        pv_forecast = self.dcf_df["pv_fcf"].sum()
        bridge_labels = ["Forecast\nPeriod", "Terminal\nValue", "Enterprise\nValue"]
        bridge_values = [pv_forecast / 1_000_000, self.pv_terminal_value / 1_000_000, 0]
        bridge_cumulative = np.cumsum(bridge_values)
        ax5.bar(
            range(len(bridge_labels)),
            bridge_values,
            color=["purple", "orange", "darkgreen"],
            alpha=0.7,
        )
        for i in range(len(bridge_labels) - 1):
            ax5.plot([i, i + 1], [bridge_cumulative[i], bridge_cumulative[i]], "k--", alpha=0.3)
        ax5.axhline(
            y=self.enterprise_value / 1_000_000,
            color="green",
            linestyle="-",
            linewidth=2,
            label="EV",
        )
        ax5.set_xticks(range(len(bridge_labels)))
        ax5.set_xticklabels(bridge_labels)
        ax5.set_title("Enterprise Value Bridge", fontsize=12, fontweight="bold")
        ax5.set_ylabel("Value ($M)")
        ax5.legend()
        ax5.grid(axis="y", alpha=0.3)

        # Sensitivity Data
        if self.sensitivity_df is None:
            self.run_sensitivity_analysis()
        if self.sensitivity_df is None:
            raise ValueError("Sensitivity analysis data is not available.")

        # 6. Sensitivity Heatmap
        ax6 = plt.subplot(3, 3, 6)
        sensitivity_table = self.sensitivity_df.pivot(
            index="terminal_growth", columns="wacc", values="enterprise_value"
        )
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
            self.dcf_df["year"],
            self.dcf_df["ebit"] / self.dcf_df["revenue"] * 100,
            marker="o",
            color="steelblue",
            linewidth=2,
            label="Operating Margin",
        )
        ax7.axhline(
            y=(
                self.financials.operating_margin * 100
                if self.financials.operating_margin is not None
                else 0.0
            ),
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
        ax8.plot(
            self.dcf_df["year"],
            self.dcf_df["discount_factor"],
            marker="o",
            color="darkred",
            linewidth=2,
        )
        ax8.set_title("Discount Factors Over Time", fontsize=12, fontweight="bold")
        ax8.set_xlabel("Year")
        ax8.set_ylabel("Discount Factor")
        ax8.grid(alpha=0.3)

        # Weekly Revenue Data Check
        if self.company_revenue_weekly_df is None:
            self.aggregate_revenue()
        if self.company_revenue_weekly_df is None:
            raise ValueError("Company revenue weekly data is not available.")

        # 9. Weekly Revenue Trend
        ax9 = plt.subplot(3, 3, 9)
        ax9.plot(
            self.company_revenue_weekly_df["ds"],
            self.company_revenue_weekly_df["revenue"] / 1000,
            color="steelblue",
            alpha=0.7,
            linewidth=1,
        )
        ax9.set_title("Weekly Revenue Forecast", fontsize=12, fontweight="bold")
        ax9.set_xlabel("Date")
        ax9.set_ylabel("Revenue ($K)")
        ax9.grid(alpha=0.3)

        plt.suptitle(
            f"DCF Valuation Dashboard - Enterprise Value: ${self.enterprise_value/1_000_000:.1f}M "
            f"(as of {self.valuation_date.date()})",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Dashboard saved to: {output_path}")

    def export_results(
        self,
        dcf_path: str = REPORT_DIRECTORY / "dcf_model_annual.csv",
        sensitivity_path: str = REPORT_DIRECTORY / "dcf_sensitivity_analysis.csv",
        summary_path: str = REPORT_DIRECTORY / "dcf_valuation_summary.csv",
    ) -> None:
        """
        Export DCF results to CSV files.

        Args:
            dcf_path: Path for DCF model export
            sensitivity_path: Path for sensitivity analysis export
            summary_path: Path for summary report export
        """
        logger.info("=" * 80)
        logger.info("EXPORTING RESULTS")
        logger.info("=" * 80)

        # Export DCF model
        if self.dcf_df is None:
            self.build_dcf_model()
        if self.dcf_df is None:
            raise ValueError("DCF model data is not available.")
        self.dcf_df.to_csv(dcf_path, index=False)
        logger.info(f"DCF model saved to: {dcf_path}")

        # Export sensitivity analysis
        if self.sensitivity_df is not None:
            sensitivity_table = self.sensitivity_df.pivot(
                index="terminal_growth", columns="wacc", values="enterprise_value"
            )
            sensitivity_table.to_csv(sensitivity_path)
            logger.info(f"Sensitivity analysis saved to: {sensitivity_path}")

        # Create summary report
        summary_report = {
            "Valuation Date": self.valuation_date.date(),
            "WACC": f"{self.assumptions.wacc:.2%}",
            "Terminal Growth Rate": f"{self.assumptions.terminal_growth_rate:.2%}",
            "Operating Margin": f"{self.financials.operating_margin:.2%}",
            "Tax Rate": f"{self.financials.tax_rate:.2%}",
            "Total Forecast Revenue": f'${self.dcf_df["revenue"].sum():,.0f}',
            "Total Forecast FCF": f'${self.dcf_df["fcf"].sum():,.0f}',
            "PV Forecast Period": f'${self.dcf_df["pv_fcf"].sum():,.0f}',
            "PV Terminal Value": f"${self.pv_terminal_value:,.0f}",
            "Enterprise Value": f"${self.enterprise_value:,.0f}",
            "Equity Value": f"${self.equity_value:,.0f}",
        }

        summary_df = pd.DataFrame([summary_report]).T
        summary_df.columns = ["Value"]
        summary_df.to_csv(summary_path)
        logger.info(f"Summary report saved to: {summary_path}")

    def run_full_valuation(
        self, create_viz: bool = True, export: bool = True
    ) -> ValuationAnalysisResults:
        """
        Run complete DCF valuation workflow.

        Args:
            create_viz: Whether to create visualizations
            export: Whether to export results

        Returns:
            Dictionary with key valuation metrics
        """
        logger.info("=" * 80)
        logger.info("RUNNING FULL DCF VALUATION")
        logger.info("=" * 80)

        # Step 1: Aggregate revenue
        self.aggregate_revenue()

        # Step 2: Convert to annual
        self.convert_to_annual()

        # Step 3: Build DCF model
        self.build_dcf_model()

        # Step 4: Calculate terminal value
        self.calculate_terminal_value()

        # Step 5: Calculate enterprise value
        self.calculate_enterprise_value()

        # Step 6: Calculate equity value
        self.calculate_equity_value()

        # Step 7: Sensitivity analysis
        self.run_sensitivity_analysis()

        # Step 8: Create visualizations
        if create_viz:
            self.create_visualizations()

        # Step 9: Export results
        if export:
            self.export_results()

        valuation_results = {
            "enterprise_value": (
                self.enterprise_value.round(2) if self.enterprise_value is not None else None
            ),
            "equity_value": self.equity_value.round(2) if self.equity_value is not None else None,
            "pv_forecast_period": (
                round(self.dcf_df["pv_fcf"].sum(), 2) if self.dcf_df is not None else None
            ),
            "pv_terminal_value": (
                self.pv_terminal_value.round(2) if self.pv_terminal_value is not None else None
            ),
            "wacc": round(self.assumptions.wacc, 2) if self.assumptions.wacc is not None else 0.1,
            "terminal_growth_rate": round(self.assumptions.terminal_growth_rate, 2),
        }

        logger.info("=" * 80)
        logger.info("DCF VALUATION COMPLETE!")
        logger.info("=" * 80)
        logger.info(valuation_results)

        return ValuationAnalysisResults(**valuation_results)
