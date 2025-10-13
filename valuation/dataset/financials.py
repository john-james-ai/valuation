#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/financials.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 05:32:44 pm                                              #
# Modified   : Sunday October 12th 2025 11:17:16 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for company financial analysis class."""
from __future__ import annotations

from dataclasses import dataclass

from valuation.utils.data import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Financials(DataClass):
    """Dataclass representing the financial statements of a company."""

    name: str  # Company Name
    fiscal_year: int  # Fiscal Year

    # Income Statement
    revenue: float  # Total Revenue
    cogs: float  # Cost of Goods Sold
    sga: float  # Selling, General & Administrative Expenses
    gross_profit: float  # Gross Profit
    operating_income: float  # Operating Income (~EBIT)
    interest_expense: float  # Interest Expense
    earnings_before_taxes: float  # Earnings Before Taxes (EBT)
    income_tax_expense: float  # Income Tax Expense
    net_income: float  # Net Income

    # Balance Sheet
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
    current_liabilities_excl_debt: float  # Current Liabilities Excluding Debt
    total_liabilities: float  # Total Liabilities
    total_equity: float  # Total Equity

    # Cashflow Statement
    net_income: float  # Net Income
    operating_activities: float  # Cash Flow from Operating Activities
    investing_activities: float  # Cash Flow from Investing Activities
    financing_activities: float  # Cash Flow from Financing Activities
    net_change_in_cash: float  # Net Change in class
    capital_expenditures: float  # Capital Expenditures (CapEx)
    deprecated_and_amortization: float  # Depreciation & Amortization (D&A)

    # Taxes
    tax_rate: float  # Tax Rate

    # Interest
    effective_interest_rate: float  # Effective Interest Rate

    # Debt Metrics
    total_debt: float  # Total Debt

    # Historical Figures
    prior_total_assets: float  # Prior Year Total Assets
    prior_current_assets: float  # Prior Year Current Assets
    prior_current_liabilities: float  # Prior Year Current Liabilities
    prior_inventory: float  # Prior Year Inventory
    prior_cogs: float  # Prior Year Cost of Goods Sold
    prior_revenue: float  # Prior Year Revenue
    prior_total_equity: float  # Prior Year Total Equity

    # Profitability Metrics and Ratios
    ebitda: float  # Earnings Before Interest, Taxes, Depreciation & Amortization (EBITDA)

    # Profitability Ratios
    @property
    def gross_profit_margin(self) -> float:
        """Calculates Gross Profit Margin."""
        return self.gross_profit / self.revenue * 100 if self.revenue else 0.0

    @property
    def operating_margin(self) -> float:
        """Calculates Operating Margin."""
        return self.operating_income / self.revenue * 100 if self.revenue else 0.0

    @property
    def net_profit_margin(self) -> float:
        """Calculates Net Profit Margin."""
        return self.net_income / self.revenue * 100 if self.revenue else 0.0 * 100

    @property
    def ebit(self) -> float:
        """Calculates Earnings Before Interest and Taxes (EBIT)."""
        return self.ebitda - self.deprecated_and_amortization

    @property
    def ebit_margin(self) -> float:  # EBIT Margin
        """Calculates EBIT Margin."""
        return self.ebit / self.revenue * 100 if self.revenue else 0.0

    @property
    def ebitda_margin(self) -> float:
        """Calculates EBITDA Margin."""
        return self.ebitda / self.revenue * 100 if self.revenue else 0.0

    @property
    def average_assets(self) -> float:
        """Calculates Average Assets."""
        return (
            (self.total_assets + self.prior_total_assets) / 2
            if self.total_assets and self.prior_total_assets
            else 0.0
        )

    @property
    def average_equity(self) -> float:
        """Calculates Average Equity."""
        return (
            (self.total_equity + self.prior_total_equity) / 2
            if self.total_equity and self.prior_total_equity
            else 0.0
        )

    @property
    def ruturn_on_assets(self) -> float:
        """Calculates Return on Assets (ROA)."""
        return self.net_income / self.average_assets * 100 if self.average_assets else 0.0

    @property
    def return_on_equity(self) -> float:
        """Calculates Return on Equity (ROE)."""
        return self.net_income / self.average_equity * 100 if self.average_equity else 0.0

    @property
    def return_on_invested_capital(self) -> float:
        """Calculates Return on Invested Capital (ROIC)."""
        invested_capital = self.total_equity + self.total_debt
        nopat = self.operating_income * (1 - self.tax_rate) * 100
        return nopat / invested_capital if invested_capital else 0.0

    @property
    def depreciation_percentage(self) -> float:
        """Calculates Depreciation as a Percentage of PP&E."""
        return (
            self.deprecated_and_amortization / self.property_plant_equipment * 100
            if self.property_plant_equipment
            else 0.0
        )

    # Liquidity Ratios
    @property
    def working_capital(self) -> float:
        """Calculates Working Capital."""
        return self.current_assets - self.current_liabilities

    @property
    def current_ratio(self) -> float:
        """Calculates Current Ratio."""
        return self.current_assets / self.current_liabilities if self.current_liabilities else 0.0

    @property
    def quick_ratio(self) -> float:
        """Calculates Quick Ratio."""
        return (
            (self.current_assets - self.inventory) / self.current_liabilities
            if self.current_liabilities
            else 0.0
        )

    # Leverage Ratios
    @property
    def debt_ratio(self) -> float:
        """Calculates Debt Ratio."""
        return self.total_liabilities / self.total_assets if self.total_assets else 0.0

    @property
    def debt_to_equity_ratio(self) -> float:
        """Calculates Debt to Equity Ratio."""
        return self.total_debt / self.total_equity if self.total_equity else 0.0

    # Efficiency Ratios
    @property
    def inventory_turnover(self) -> float:
        """Calculates Inventory Turnover."""
        return self.cogs / self.inventory if self.inventory else 0.0

    @property
    def days_inventory_outstanding(self) -> float:
        """Calculates Days Inventory Outstanding (DIO)."""
        average_inventory = (
            (self.inventory + self.prior_inventory) / 2 if self.prior_inventory else self.inventory
        )
        return 365 * average_inventory / self.cogs if self.cogs > 0 else 0

    @property
    def days_payables_outstanding(self) -> float:
        """Calculates Days Payables Outstanding (DPO)."""
        return 365 * self.accounts_payable / self.cogs if self.cogs > 0 else 0

    @property
    def sga_to_sales(self) -> float:
        """Calculates SGA as a Percentage of Sales."""
        return self.sga / self.revenue * 100 if self.revenue else 0.0

    @property
    def dna_to_sales(self) -> float:
        """Calculates Depreciation & Amortization as a Percentage of Sales."""
        return self.deprecated_and_amortization / self.revenue * 100 if self.revenue else 0.0

    @property
    def cogs_growth_rate(self) -> float:
        """Calculates COGS Growth Rate."""
        return (self.cogs - self.prior_cogs) / self.prior_cogs * 100 if self.prior_cogs else 0.0

    # Growth Metrics
    @property
    def sales_growth(self) -> float:
        """Calculates Sales Growth."""
        return self.revenue - self.prior_revenue

    @property
    def sales_growth_rate(self) -> float:
        """Calculates Sales Growth Rate."""
        return (
            (self.revenue - self.prior_revenue) / self.prior_revenue * 100
            if self.prior_revenue
            else 0.0
        )

    @property
    def capex_to_sales(self) -> float:
        """Calculates Capital Expenditures as a Percentage of Sales."""
        return self.capital_expenditures / self.revenue * 100 if self.revenue else 0.0

    @property
    def nwc_to_sales(self) -> float:
        """Calculates Net Working Capital as a Percentage of Sales."""
        return self.capital_expenditures / self.revenue * 100 if self.revenue else 0.0

    # Liquidity Metrics
    @property
    def net_working_capital(self) -> float:
        """Calculates Net Working Capital."""
        return self.current_assets - self.current_liabilities

    @property
    def net_working_capital_change(self) -> float:
        """Calculates Change in Net Working Capital."""
        prior_net_working_capital = (
            self.prior_current_assets - self.prior_current_liabilities
            if self.prior_current_assets and self.prior_current_liabilities
            else 0.0
        )
        return self.net_working_capital - prior_net_working_capital

    @property
    def net_working_capital_change_per_pct_sales_growth(self) -> float:
        """Calculates Change in Net Working Capital per Percentage of Sales Growth."""

        nwc_to_sales = self.net_working_capital / self.revenue * 100 if self.revenue else 0.0
        sales_growth_pct = self.sales_growth_rate
        return nwc_to_sales * sales_growth_pct if sales_growth_pct and nwc_to_sales else 0.0

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
