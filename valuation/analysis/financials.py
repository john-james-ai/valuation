#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/financials.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 05:32:44 pm                                              #
# Modified   : Saturday October 11th 2025 09:10:38 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for company financial analysis class."""
from pydantic.dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IncomeStatement:
    """Dataclass for storing income statement data."""

    revenue: float  # Total Revenue
    cogs: float  # Cost of Goods Sold
    sga: float  # Selling, General & Administrative Expenses
    gross_profit: float  # Gross Profit
    operating_income: float  # Operating Income (EBIT)
    ebit: float  # Earnings Before Interest and Taxes
    interest_expense: float  # Interest Expense
    earnings_before_taxes: float  # Earnings Before Taxes (EBT)
    income_tax_expense: float  # Income Tax Expense
    net_income: float  # Net Income


# ------------------------------------------------------------------------------------------------ #
@dataclass
class BalanceSheet:
    """Dataclass for storing balance sheet data."""

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
    current_liabilities: float  # Total Current Liabilities
    total_liabilities: float  # Total Liabilities
    total_equity: float  # Total Equity


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CashFlowStatement:
    """Dataclass for storing cash flow statement data."""

    net_income: float  # Net Income
    operating_activities: float  # Cash Flow from Operating Activities
    investing_activities: float  # Cash Flow from Investing Activities
    financing_activities: float  # Cash Flow from Financing Activities
    net_change_in_cash: float  # Net Change in class
    capital_expenditures: float  # Capital Expenditures (CapEx)
    deprecated_and_amortization: float  # Depreciation & Amortization (D&A)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FinancialRatios:
    """Dataclass for storing financial ratios."""

    # Profitability Ratios
    gross_profit_margin: float  # Gross Profit Margin
    operating_margin: float  # Operating Margin
    net_profit_margin: float  # Net Profit Margin
    ebit_margin: float  # EBIT Margin
    ebitda: float  # Earnings Before Interest, Taxes, Depreciation & Amortization (EBITDA)
    ebitda_margin: float
    return_on_assets: float  # Return on Assets (ROA)
    return_on_equity: float  # Return on Equity (ROE)
    return_on_invested_capital: float  # Return on Invested Capital (ROIC)
    effective_interest_rate: float  # Effective Interest Rate
    depreciation_percentage: float  # Depreciation as a Percentage of Sales
    # Liquidity Ratios
    working_capital: float  # Working Capital
    current_ratio: float  # Current Ratio
    quick_ratio: float  # Quick Ratio
    # Leverage Ratios
    debt_to_equity_ratio: float  # Debt to Equity Ratio
    # Efficiency Ratios
    inventory_turnover: float  # Inventory Turnover
    dio: float  # Days Inventory Outstanding (DIO)
    dpo: float  # Days Payables Outstanding (DPO)
    sga_percentage: float  # Selling, General & Administrative Expenses as a Percentage of Sales
    cogs_growth_rate: float  # Cost of Goods Sold Growth Rate


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FinancialMetrics:
    """Dataclass for storing financial metrics."""

    # Growth Metrics
    sales_growth: float  # Revenue Growth
    sales_growth_rate: float  # Revenue Growth Rate
    capex_percentage: float  # Capital Expenditures as a Percentage of Sales
    # Liquidity Metrics
    net_working_capital_change: float  # Change in Net Working
    net_working_capital_change_per_pct_sales_growth: (
        float  # Change in Net Working Capital per Percentage of Sales Growth
    )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FinancialPerformance:
    """Dataclass for storing overall financial performance."""

    name: str
    fiscal_year: int
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow_statement: CashFlowStatement
    financial_ratios: FinancialRatios
    financial_metrics: FinancialMetrics

    @classmethod
    def from_dict(cls, data: dict) -> "FinancialPerformance":
        """Creates a FinancialPerformance instance from a dictionary."""
        return cls(
            name=data["name"],
            fiscal_year=data["fiscal_year"],
            income_statement=IncomeStatement(**data["income_statement"]),
            balance_sheet=BalanceSheet(**data["balance_sheet"]),
            cash_flow_statement=CashFlowStatement(**data["cash_flow_statement"]),
            financial_ratios=FinancialRatios(**data["financial_ratios"]),
            financial_metrics=FinancialMetrics(**data["financial_metrics"]),
        )
