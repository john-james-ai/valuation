#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/financials.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 05:32:44 pm                                              #
# Modified   : Friday October 24th 2025 12:31:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for company financial analysis class."""
from __future__ import annotations

from dataclasses import dataclass

from valuation.core.dataclass import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Financials(DataClass):
    """Dataclass representing the financial statements of a company."""

    fiscal_year: int = 1997
    # INCOME STATEMENT
    revenue: float = 2511962
    cogs: float = 1932994
    sga: float = 491359
    gross_profit: float = 578968
    operating_income: float = 77109
    interest_expense: float = 67555
    earnings_before_taxes: float = 6813
    income_tax_expense: float = 7385
    net_income: float = -6932

    # BALANCE SHEET
    # Assets
    cash_and_equivalents: float = 32735
    accounts_receivable: float = 16723
    inventory: float = 203411
    prepaid_expenses: float = 21860
    other_current_assets: float = 459252
    current_assets: float = 274729
    property_plant_equipment: float = 368224
    total_assets: float = 1102205
    average_assets: float = 0.0
    # Liabilities
    accounts_payable: float = 187787
    short_term_debt: float = 16723
    long_term_debt: float = 400644
    current_liabilities: float = 308434
    current_liabilities_excl_debt: float = 298382
    total_liabilities: float = 1102205
    total_debt: float = 540700

    average_equity: float = 0.0
    total_equity: float = 179071

    # cash_flow_statement
    net_income: float = -6932
    operating_activities: float = 41120
    investing_activities: float = -48561
    financing_activities: float = -15375
    net_change_in_cash: float = -22816
    capital_expenditures: float = -49588
    deprecated_and_amortization: float = 45924

    # TAXES
    tax_rate: float = 0.35

    # INTEREST
    effective_interest_rate: float = 9.2

    # HISTORICAL FIGURES
    prior_total_assets: float = 1086423
    prior_current_assets: float = 274318
    prior_inventory: float = 182880
    prior_cogs: float = 1884161
    prior_revenue: float = 2433724
    prior_current_liabilities: float = 308504
    prior_total_equity: float = 139850

    # PROFITABILITY METRICS
    ebitda: float = 114900
    ebit: float = 0
    return_on_assets: float = 0.0
    return_on_equity: float = 0.0
    return_on_invested_capital: float = 0.0

    # PROFITABIITY RATIOS
    gross_profit_margin: float = 0.0
    operating_margin: float = 0.0
    net_profit_margin: float = 0.0
    ebit_margin: float = 0.0
    ebitda_margin: float = 0.0

    # DEPRECIATION METRICS
    depreciation_percentage: float = 0.0

    # LIQUIDITY RATIOS
    working_capital: float = 0.0
    current_ratio: float = 0.0
    quick_ratio: float = 0.0

    # LEVERAGE RATIOS
    debt_ratio: float = 0.0
    debt_to_equity_ratio: float = 0.0

    # EFFICIENCY RATIOS
    inventory_turnover: float = 0.0
    days_inventory_outstanding: float = 0.0
    days_payables_outstanding: float = 0.0
    sga_to_sales: float = 0.0
    dna_to_sales: float = 0.0
    cogs_growth_rate: float = 0.0

    # GROWTH METRICS
    sales_growth: float = 0.0
    sales_growth_rate: float = 0.0
    capex_to_sales: float = 0.0
    nwc_to_sales: float = 0.0

    # LIQUIDITY METRICS
    net_working_capital: float = 0.0
    net_working_capital_change: float = 0.0
    net_working_capital_change_per_pct_sales_growth: float = 0.0

    def __post_init__(self):
        """Calculates derived financial metrics after initialization."""
        # Profiability Metrics
        self.ebit = self.ebitda - self.deprecated_and_amortization
        # Profitability Ratios
        self.gross_profit_margin = self.gross_profit / self.revenue * 100 if self.revenue else 0.0
        self.operating_margin = self.operating_income / self.revenue * 100 if self.revenue else 0.0
        self.net_profit_margin = self.net_income / self.revenue * 100 if self.revenue else 0.0

        self.ebit_margin = self.ebit / self.revenue * 100 if self.revenue else 0.0
        self.ebitda_margin = self.ebitda / self.revenue * 100 if self.revenue else 0.0
        # Average Assets & Equity
        self.average_assets = (
            (self.total_assets + self.prior_total_assets) / 2
            if self.total_assets and self.prior_total_assets
            else 0.0
        )
        self.return_on_assets = (
            self.net_income / self.average_assets * 100 if self.average_assets else 0.0
        )
        self.return_on_equity = (
            self.net_income / self.average_equity * 100 if self.average_equity else 0.0
        )
        self.return_on_invested_capital = self._calculate_roic()
        self.depreciation_percentage = (
            self.deprecated_and_amortization / self.property_plant_equipment * 100
            if self.property_plant_equipment
            else 0.0
        )

        # Liquidity Ratios
        self.working_capital = self.current_assets - self.current_liabilities
        self.current_ratio = (
            self.current_assets / self.current_liabilities if self.current_liabilities else 0.0
        )
        self.quick_ratio = (
            (self.current_assets - self.inventory) / self.current_liabilities
            if self.current_liabilities
            else 0.0
        )
        self.debt_ratio = self.total_liabilities / self.total_assets if self.total_assets else 0.0
        self.debt_to_equity_ratio = (
            self.total_debt / self.total_equity if self.total_equity else 0.0
        )

        # Efficiency Ratios
        self.inventory_turnover = self.cogs / self.inventory if self.inventory else 0.0
        self.days_inventory_outstanding = (
            365
            * (
                (self.inventory + self.prior_inventory) / 2
                if self.prior_inventory
                else self.inventory
            )
            / self.cogs
            if self.cogs > 0
            else 0
        )
        self.days_payables_outstanding = (
            365 * self.accounts_payable / self.cogs if self.cogs > 0 else 0
        )
        self.sga_to_sales = self.sga / self.revenue * 100 if self.revenue else 0.0
        self.dna_to_sales = (
            self.deprecated_and_amortization / self.revenue * 100 if self.revenue else 0.0
        )
        self.cogs_growth_rate = (
            (self.cogs - self.prior_cogs) / self.prior_cogs * 100 if self.prior_cogs else 0.0
        )

        # Growth Metrics
        self.sales_growth = self.revenue - self.prior_revenue
        self.sales_growth_rate = (
            (self.revenue - self.prior_revenue) / self.prior_revenue * 100
            if self.prior_revenue
            else 0.0
        )
        self.capex_to_sales = (
            self.capital_expenditures / self.revenue * 100 if self.revenue else 0.0
        )
        self.nwc_to_sales = self.capital_expenditures / self.revenue * 100 if self.revenue else 0.0

        # Liquidity Metrics
        self.net_working_capital = self.current_assets - self.current_liabilities
        self.net_working_capital_change = self._calculate_change_in_net_working_capital()
        self.net_working_capital_change_per_pct_sales_growth = (
            self._calculate_net_working_capital_change_per_pct_sales_growth()
        )

    @property
    def income_statement(self) -> IncomeStatement:
        """Returns the income statement as a dataclass."""
        return IncomeStatement(
            revenue=self.revenue,
            cogs=self.cogs,
            sga=self.sga,
            gross_profit=self.gross_profit,
            operating_income=self.operating_income,
            interest_expense=self.interest_expense,
            earnings_before_taxes=self.earnings_before_taxes,
            income_tax_expense=self.income_tax_expense,
            net_income=self.net_income,
        )

    @property
    def balance_sheet(self) -> BalanceSheet:
        """Returns the balance sheet as a dataclass."""
        return BalanceSheet(
            cash_and_equivalents=self.cash_and_equivalents,
            accounts_receivable=self.accounts_receivable,
            inventory=self.inventory,
            prepaid_expenses=self.prepaid_expenses,
            other_current_assets=self.other_current_assets,
            current_assets=self.current_assets,
            property_plant_equipment=self.property_plant_equipment,
            total_assets=self.total_assets,
            accounts_payable=self.accounts_payable,
            short_term_debt=self.short_term_debt,
            long_term_debt=self.long_term_debt,
            total_debt=self.total_debt,
            current_liabilities=self.current_liabilities,
            current_liabilities_excl_debt=self.current_liabilities_excl_debt,
            total_liabilities=self.total_liabilities,
            total_equity=self.total_equity,
        )

    @property
    def cashflow_statement(self) -> CashflowStatement:
        """Returns the cashflow statement as a dataclass."""
        return CashflowStatement(
            net_income=self.net_income,
            operating_activities=self.operating_activities,
            investing_activities=self.investing_activities,
            financing_activities=self.financing_activities,
            net_change_in_cash=self.net_change_in_cash,
            capital_expenditures=self.capital_expenditures,
            deprecated_and_amortization=self.deprecated_and_amortization,
        )

    @classmethod
    def from_dict(cls, data: dict) -> Financials:
        """Creates Finances instance from a dictionary."""
        return cls(**data)

    def _calculate_roic(self) -> float:
        """Calculates Return on Invested Capital (ROIC)."""
        nopat = self.operating_income * (1 - self.tax_rate)
        invested_capital = self.average_assets - self.current_liabilities_excl_debt
        return (nopat / invested_capital * 100) if invested_capital else 0.0

    def _calculate_change_in_net_working_capital(self) -> float:
        """Calculates Change in Net Working Capital."""
        prior_net_working_capital = (
            self.prior_current_assets - self.prior_current_liabilities
            if self.prior_current_assets and self.prior_current_liabilities
            else 0.0
        )
        return self.net_working_capital - prior_net_working_capital

    def _calculate_net_working_capital_change_per_pct_sales_growth(self) -> float:
        """Calculates Change in Net Working Capital per Percentage of Sales Growth."""

        nwc_to_sales = self.net_working_capital / self.revenue * 100 if self.revenue else 0.0
        sales_growth_pct = self.sales_growth_rate
        return nwc_to_sales * sales_growth_pct if sales_growth_pct and nwc_to_sales else 0.0


@dataclass
class IncomeStatement(DataClass):
    """Dataclass representing the income statement of a company."""

    revenue: float  # Total Revenue
    cogs: float  # Cost of Goods Sold
    sga: float  # Selling, General & Administrative Expenses
    gross_profit: float  # Gross Profit
    operating_income: float  # Operating Income (~EBIT)
    interest_expense: float  # Interest Expense
    earnings_before_taxes: float  # Earnings Before Taxes (EBT)
    income_tax_expense: float  # Income Tax Expense
    net_income: float  # Net Income


@dataclass
class BalanceSheet(DataClass):
    """Dataclass representing the balance sheet of a company."""

    cash_and_equivalents: float  # Cash and Cash Equivalents
    accounts_receivable: float  # Accounts Receivable
    inventory: float  # Inventory
    prepaid_expenses: float  # Prepaid Expenses
    other_current_assets: float  # Other Current Assets
    current_assets: float  # Total Current Assets
    property_plant_equipment: float  # Property, Plant & Equipment (PP&E)
    total_assets: float  # Total Assets
    accounts_payable: float  # Accounts Payable
    short_term_debt: float  # Short-Term Debt
    long_term_debt: float  # Long-Term Debt
    total_debt: float  # Total Debt
    current_liabilities: float  # Total Current Liabilities
    current_liabilities_excl_debt: float  # Current Liabilities Excluding Debt
    total_liabilities: float  # Total Liabilities
    total_equity: float  # Total Equity


@dataclass
class CashflowStatement(DataClass):
    """Dataclass representing the cashflow statement of a company."""

    net_income: float  # Net Income
    operating_activities: float  # Cash Flow from Operating Activities
    investing_activities: float  # Cash Flow from Investing Activities
    financing_activities: float  # Cash Flow from Financing Activities
    net_change_in_cash: float  # Net Change in class
    capital_expenditures: float  # Capital Expenditures (CapEx)
    deprecated_and_amortization: float  # Depreciation & Amortization (D&A)


@dataclass
class ProfitabilityRatios(DataClass):
    """Dataclass representing profitability ratios of a company."""

    gross_profit_margin: float  # Gross Profit Margin (%)
    operating_margin: float  # Operating Margin (%)
    net_profit_margin: float  # Net Profit Margin (%)
    ebit_margin: float  # EBIT Margin (%)
    ebitda_margin: float  # EBITDA Margin (%)


@dataclass
class ProfitabilityMetrics(DataClass):
    """Dataclass representing profitability metrics of a company."""

    ebitda: float  # Earnings Before Interest, Taxes, Depreciation & Amortization (EBITDA)
    ebit: float  # Earnings Before Interest & Taxes (EBIT)
    return_on_assets: float  # Return on Assets (ROA) (%)
    return_on_equity: float  # Return on Equity (ROE) (%)
    return_on_invested_capital: float  # Return on Invested Capital (ROIC) (%)


@dataclass
class LiquidityRatios(DataClass):
    """Dataclass representing liquidity ratios of a company."""

    working_capital: float  # Working Capital
    current_ratio: float  # Current Ratio
    quick_ratio: float  # Quick Ratio


@dataclass
class LiquidityMetrics(DataClass):
    """Dataclass representing liquidity metrics of a company."""

    net_working_capital: float  # Net Working Capital
    net_working_capital_change: float  # Change in Net Working Capital
    net_working_capital_change_per_pct_sales_growth: float  # Change in NWC per % Sales Growth


@dataclass
class LeverageRatios(DataClass):
    """Dataclass representing leverage ratios of a company."""

    debt_ratio: float  # Debt Ratio
    debt_to_equity_ratio: float  # Debt to Equity Ratio


@dataclass
class EfficiencyRatios(DataClass):
    """Dataclass representing efficiency ratios of a company."""

    inventory_turnover: float  # Inventory Turnover
    days_inventory_outstanding: float  # Days Inventory Outstanding
    days_payables_outstanding: float  # Days Payables Outstanding
    sga_to_sales: float  # SGA to Sales (%)
    dna_to_sales: float  # D&A to Sales (%)
    cogs_growth_rate: float  # COGS Growth Rate (%)


@dataclass
class GrowthMetrics(DataClass):
    """Dataclass representing growth metrics of a company."""

    sales_growth: float  # Sales Growth
    sales_growth_rate: float  # Sales Growth Rate (%)
    capex_to_sales: float  # CapEx to Sales (%)
    nwc_to_sales: float  # NWC to Sales (%)
