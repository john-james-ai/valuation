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
# Modified   : Friday October 24th 2025 07:26:49 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
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
    revenue: int = 2511962000
    cogs: int = 1932994000
    sga: int = 491359000
    gross_profit: int = 578968000
    operating_income: int = 77109000
    interest_expense: int = 67555000
    earnings_before_taxes: int = 6813000
    income_tax_expense: int = 7385000
    net_income: int = -6932000

    # BALANCE SHEET
    # Assets
    cash_and_equivalents: int = 32735000
    accounts_receivable: int = 16723000
    inventory: int = 203411000
    prepaid_expenses: int = 21860000
    current_assets: int = 274729000
    property_plant_equipment: int = 368224000
    total_assets: int = 1102205000
    average_assets: float = 0.0

    # Liabilities
    accounts_payable: int = 187787000
    current_liabilities: int = 308434000
    current_liabilities_excl_debt: int = 298382000
    total_liabilities: int = 0
    total_liabilities_and_equity: int = 1102205000
    total_debt: int = 540700000
    total_equity: int = 179071000

    average_equity: float = 0.0

    # cash_flow_statement
    # net_income: int = -6932
    operating_activities: int = 41120000
    investing_activities: int = -48561000
    financing_activities: int = -15375000
    net_change_in_cash: int = -22816000
    capital_expenditures: int = -49588000
    depreciation_and_amortization: int = 45924000

    # FREE CASH FLOW
    free_cash_flow: int = 0

    # TAXES
    tax_rate: float = 0.35

    # INTEREST
    effective_interest_rate: float = 9.2

    # HISTORICAL FIGURES
    prior_total_assets: int = 1086423000
    prior_current_assets: int = 274318000
    prior_inventory: int = 182880000
    prior_cogs: int = 1884161000
    prior_revenue: int = 2433724000
    prior_current_liabilities_excl_debt: int = 298376000
    prior_total_equity: int = 139850000

    # PROFITABILITY METRICS
    ebitda: int = 114900000
    ebit: int = 0
    nopat: int = 0
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
    working_capital: int = 0
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
    net_working_capital: int = 0
    net_working_capital_change: int = 0
    net_working_capital_change_per_pct_sales_growth: float = 0.0

    def __post_init__(self):
        """Calculates derived financial metrics after initialization."""
        # Balance Sheet
        self.total_liabilities = self.total_liabilities_and_equity - self.total_equity
        # Profiability Metrics
        self.ebit = self.ebitda - self.depreciation_and_amortization
        # Profitability Ratios
        self.gross_profit_margin = self.gross_profit / self.revenue * 100 if self.revenue else 0.0
        self.operating_margin = self.operating_income / self.revenue * 100 if self.revenue else 0.0
        self.net_profit_margin = self.net_income / self.revenue * 100 if self.revenue else 0.0

        self.ebit_margin = self.ebit / self.revenue * 100 if self.revenue else 0.0
        self.ebitda_margin = self.ebitda / self.revenue * 100 if self.revenue else 0.0
        self.nopat = int(self.ebit * (1 - self.tax_rate))
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
            self.depreciation_and_amortization / self.property_plant_equipment * 100
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
            self.depreciation_and_amortization / self.revenue * 100 if self.revenue else 0.0
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
            current_assets=self.current_assets,
            property_plant_equipment=self.property_plant_equipment,
            total_assets=self.total_assets,
            accounts_payable=self.accounts_payable,
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
            depreciation_and_amortization=self.depreciation_and_amortization,
        )

    @property
    def profitability_ratios(self) -> ProfitabilityRatios:
        """Returns the profitability ratios as a dataclass."""
        return ProfitabilityRatios(
            gross_profit_margin=self.gross_profit_margin,
            operating_margin=self.operating_margin,
            net_profit_margin=self.net_profit_margin,
            ebit_margin=self.ebit_margin,
            ebitda_margin=self.ebitda_margin,
        )

    @property
    def profitability_metrics(self) -> ProfitabilityMetrics:
        """Returns the profitability metrics as a dataclass."""
        return ProfitabilityMetrics(
            ebitda=self.ebitda,
            ebit=self.ebit,
            return_on_assets=self.return_on_assets,
            return_on_equity=self.return_on_equity,
            return_on_invested_capital=self.return_on_invested_capital,
        )

    @property
    def liquidity_ratios(self) -> LiquidityRatios:
        """Returns the liquidity ratios as a dataclass."""
        return LiquidityRatios(
            working_capital=self.working_capital,
            current_ratio=self.current_ratio,
            quick_ratio=self.quick_ratio,
        )

    @property
    def liquidity_metrics(self) -> LiquidityMetrics:
        """Returns the liquidity metrics as a dataclass."""
        return LiquidityMetrics(
            net_working_capital=self.net_working_capital,
            net_working_capital_change=self.net_working_capital_change,
            net_working_capital_change_per_pct_sales_growth=self.net_working_capital_change_per_pct_sales_growth,
        )

    @property
    def leverage_ratios(self) -> LeverageRatios:
        """Returns the leverage ratios as a dataclass."""
        return LeverageRatios(
            debt_ratio=self.debt_ratio,
            debt_to_equity_ratio=self.debt_to_equity_ratio,
        )

    @property
    def efficiency_ratios(self) -> EfficiencyRatios:
        """Returns the efficiency ratios as a dataclass."""
        return EfficiencyRatios(
            inventory_turnover=self.inventory_turnover,
            days_inventory_outstanding=self.days_inventory_outstanding,
            days_payables_outstanding=self.days_payables_outstanding,
            sga_to_sales=self.sga_to_sales,
            dna_to_sales=self.dna_to_sales,
            cogs_growth_rate=self.cogs_growth_rate,
        )

    @property
    def growth_metrics(self) -> GrowthMetrics:
        """Returns the growth metrics as a dataclass."""
        return GrowthMetrics(
            sales_growth=self.sales_growth,
            sales_growth_rate=self.sales_growth_rate,
            capex_to_sales=self.capex_to_sales,
            nwc_to_sales=self.nwc_to_sales,
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

    def _calculate_change_in_net_working_capital(self) -> int:
        """Calculates Change in Net Working Capital."""
        prior_net_working_capital = (
            self.prior_current_assets - self.prior_current_liabilities_excl_debt
            if self.prior_current_assets and self.prior_current_liabilities_excl_debt
            else 0
        )
        return self.net_working_capital - prior_net_working_capital

    def _calculate_net_working_capital_change_per_pct_sales_growth(self) -> float:
        """Calculates Change in Net Working Capital per Percentage of Sales Growth."""

        nwc_to_sales = self.net_working_capital / self.revenue * 100 if self.revenue else 0.0
        sales_growth_pct = self.sales_growth_rate
        return nwc_to_sales * sales_growth_pct if sales_growth_pct and nwc_to_sales else 0.0

    def _calculate_free_cash_flow(self) -> int:
        """Calculates Free Cash Flow (FCF)."""
        return (
            self.nopat
            + self.depreciation_and_amortization
            + self.capital_expenditures
            - self.net_working_capital_change
        )


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

    cash_and_equivalents: int  # Cash and Cash Equivalents
    accounts_receivable: int  # Accounts Receivable
    inventory: int  # Inventory
    prepaid_expenses: int  # Prepaid Expenses
    current_assets: int  # Total Current Assets
    property_plant_equipment: int  # Property, Plant & Equipment (PP&E)
    total_assets: int  # Total Assets
    accounts_payable: int  # Accounts Payable
    total_debt: int  # Total Debt
    current_liabilities: int  # Total Current Liabilities
    current_liabilities_excl_debt: int  # Current Liabilities Excluding Debt
    total_liabilities: int  # Total Liabilities
    total_equity: int  # Total Equity


@dataclass
class CashflowStatement(DataClass):
    """Dataclass representing the cashflow statement of a company."""

    net_income: int  # Net Income
    operating_activities: int  # Cash Flow from Operating Activities
    investing_activities: int  # Cash Flow from Investing Activities
    financing_activities: int  # Cash Flow from Financing Activities
    net_change_in_cash: int  # Net Change in class
    capital_expenditures: int  # Capital Expenditures (CapEx)
    depreciation_and_amortization: int  # Depreciation & Amortization (D&A)


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

    ebitda: int  # Earnings Before Interest, Taxes, Depreciation & Amortization (EBITDA)
    ebit: int  # Earnings Before Interest & Taxes (EBIT)
    return_on_assets: float  # Return on Assets (ROA) (%)
    return_on_equity: float  # Return on Equity (ROE) (%)
    return_on_invested_capital: float  # Return on Invested Capital (ROIC) (%)


@dataclass
class LiquidityRatios(DataClass):
    """Dataclass representing liquidity ratios of a company."""

    working_capital: int  # Working Capital
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
