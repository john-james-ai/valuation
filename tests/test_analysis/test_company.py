#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_analysis/test_company.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 08:17:34 pm                                              #
# Modified   : Sunday October 12th 2025 03:07:14 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect
import math

from loguru import logger
import pandas as pd
import pytest

from valuation.analysis.company import Company
from valuation.analysis.financials import (
    BalanceSheet,
    CashflowStatement,
    Financials,
    IncomeStatement,
)

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.company
class TestCompany:  # pragma: no cover
    # ============================================================================================ #
    def test_financials(self, financials, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        fin = Financials.from_dict(financials)
        # Check that the financials object is created correctly
        assert math.isclose(fin.cash_and_equivalents, 32735.00)
        assert math.isclose(fin.accounts_receivable, 16723.00)
        assert math.isclose(fin.inventory, 203411.00)
        assert math.isclose(fin.prepaid_expenses, 21860.00)
        assert math.isclose(fin.other_current_assets, 459252.00)
        assert math.isclose(fin.current_assets, 274729.00)
        assert math.isclose(fin.property_plant_equipment, 368224.00)
        assert math.isclose(fin.total_assets, 1102205.00)
        assert math.isclose(fin.accounts_payable, 187787.00)
        assert math.isclose(fin.short_term_debt, 16723)
        assert math.isclose(fin.long_term_debt, 400644.00)
        assert math.isclose(fin.current_liabilities, 308434.00)
        assert math.isclose(fin.total_liabilities, 1102205.00)
        assert math.isclose(fin.total_equity, 179071.00)
        assert math.isclose(fin.average_equity, 159460.50)
        assert math.isclose(fin.net_income, -6932.00)
        assert math.isclose(fin.operating_activities, 41120.00)
        assert math.isclose(fin.investing_activities, -48561.00)
        assert math.isclose(fin.financing_activities, -15375.00)
        assert math.isclose(fin.net_change_in_cash, -22816.00)
        assert math.isclose(fin.capital_expenditures, -49588.00)
        assert math.isclose(fin.deprecated_and_amortization, 45924.00)
        assert math.isclose(fin.revenue, 2511962.00)
        assert math.isclose(fin.cogs, 1932994.00)
        assert math.isclose(fin.sga, 491359.00)
        assert math.isclose(fin.gross_profit, 578968.00)
        assert math.isclose(fin.operating_income, 77109.00)
        assert math.isclose(fin.ebit, 68976.00)
        assert math.isclose(fin.interest_expense, 67555.00)
        assert math.isclose(fin.earnings_before_taxes, 6813.00)
        assert math.isclose(fin.income_tax_expense, 7385.00)
        assert math.isclose(fin.net_income, -6932.00)
        assert math.isclose(fin.sales_growth, 78238.00)
        assert math.isclose(fin.sales_growth_rate, 3.214, rel_tol=0.05)
        assert math.isclose(fin.capex_percentage, -1.97, rel_tol=0.05)
        assert math.isclose(fin.net_working_capital_change, 481.00)
        assert math.isclose(fin.gross_profit_margin, 23.05, rel_tol=0.05)
        assert math.isclose(fin.operating_margin, 3.07, rel_tol=0.05)
        assert math.isclose(fin.net_profit_margin, -0.28, rel_tol=0.05)
        assert math.isclose(fin.ebit_margin, 2.75, rel_tol=0.05)
        assert math.isclose(fin.ebitda, 114900.00)
        assert math.isclose(fin.ebitda_margin, 4.57, rel_tol=0.05)
        assert math.isclose(fin.return_on_equity, -4.35, rel_tol=0.05)
        assert math.isclose(fin.return_on_invested_capital, 6.96, rel_tol=0.05)
        assert math.isclose(fin.effective_interest_rate, 9.2, rel_tol=0.05)
        assert math.isclose(fin.depreciation_percentage, 12.47, rel_tol=0.05)
        assert math.isclose(fin.working_capital, -33705.00)
        assert math.isclose(fin.current_ratio, 0.89, rel_tol=0.05)
        assert math.isclose(fin.quick_ratio, 0.23, rel_tol=0.05)
        assert math.isclose(fin.debt_to_equity_ratio, 3.02, rel_tol=0.05)
        assert math.isclose(fin.inventory_turnover, 9.50, rel_tol=0.05)
        assert math.isclose(fin.days_inventory_outstanding, 36.47, rel_tol=0.05)
        assert math.isclose(fin.days_payables_outstanding, 35.46, rel_tol=0.05)
        assert math.isclose(fin.cogs_growth_rate, 2.59, rel_tol=0.05)

        assert isinstance(fin.income_statement, IncomeStatement)
        assert isinstance(fin.balance_sheet, BalanceSheet)
        assert isinstance(fin.cashflow_statement, CashflowStatement)
        assert isinstance(fin.as_dict(), dict)
        logger.info(fin.income_statement)
        logger.info(fin.balance_sheet)
        logger.info(fin.cashflow_statement)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_company_annual_sales(self, financials, sales, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        fin = Financials.from_dict(financials)
        company = Company(financials=fin, sales=sales)
        assert isinstance(company.financials, Financials)
        assert isinstance(company.annual_sales, pd.DataFrame)
        annual_sales = company.annual_sales

        assert not annual_sales.empty
        assert "year" in annual_sales.columns
        assert "yoy_growth" in annual_sales.columns
        assert "revenue" in annual_sales.columns

        # Check that there are no NaN values in the DataFrames
        assert not annual_sales["year"].hasnans
        assert not annual_sales["revenue"].hasnans

        # Check that the years are in ascending order
        years = annual_sales["year"].tolist()
        assert years == sorted(years)

        # Check that the yoy_growth values are floats and within a reasonable range
        yoy_growth_values = annual_sales["yoy_growth"].tolist()
        for value in yoy_growth_values:
            assert isinstance(value, float) or value is None  # Allow None for the first year
            if value is not None and not pd.isna(value):
                assert -100.0 < value < 1000.0  # Assuming growth rates between -100% and +1000%

        # Check that the revenue values are positive floats
        revenue_values = annual_sales["revenue"].tolist()
        for value in revenue_values:
            assert isinstance(value, float)
            assert value >= 0.0  # Revenue should be non-negative

        # Check that the index has the correct name
        assert "year" in annual_sales.columns or annual_sales.index.name == "year"
        assert len(annual_sales) > 0  # Ensure there is at least one year of data

        # Check that the revenue values are in a realistic range
        for value in revenue_values:
            assert (
                0 < value < 1e10
            )  # Assuming revenue should be less than 10 billion for this company

        # Check that the first year's yoy_growth is None
        assert pd.Series(yoy_growth_values[0]).isna().all()

        # Check that the DataFrame has the expected number of rows (years)
        expected_years = sales["year"].nunique()
        assert (
            len(annual_sales) <= expected_years
        )  # Should be less than or equal to the number of unique years in sales data

        # Check that the DataFrame is sorted by year
        assert annual_sales.index.is_monotonic_increasing

        logger.info(annual_sales)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_sss_growth(self, financials, sales, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        fin = Financials.from_dict(financials)
        company = Company(financials=fin, sales=sales)
        assert isinstance(company.financials, Financials)
        # Check that sss_growth is a DataFrame and has the expected structure
        assert isinstance(company.sss_growth, pd.DataFrame)

        # Check that the DataFrame is not empty
        sss_growth = company.sss_growth
        assert not sss_growth.empty

        # Check that the DataFrame has the expected columns
        assert "year" in sss_growth.columns
        assert "sss_growth" in sss_growth.columns
        assert "num_comp_stores" in sss_growth.columns

        # Check that there are no NaN values in the DataFrame
        assert not sss_growth.isnull().values.any()
        # Check that the years are in ascending order
        years = sss_growth["year"].tolist()
        assert years == sorted(years)
        # Check that the sss_growth values are floats and within a reasonable range
        sss_growth_values = sss_growth["sss_growth"].tolist()
        for value in sss_growth_values:
            assert isinstance(value, float) or value is None  # Allow None for the first year
            if value is not None and not pd.isna(value):
                assert -100.0 < value < 1000.0  # Assuming growth rates between -100% and +1000%
        # Check that the num_comp_stores values are positive integers
        num_comp_stores_values = sss_growth["num_comp_stores"].tolist()
        for value in num_comp_stores_values:
            assert isinstance(value, int)
            assert value >= 0  # Store count should be non-negative

        # Check that the index has the correct name
        assert "year" in sss_growth.columns or sss_growth.index.name == "year"
        assert len(sss_growth) > 0  # Ensure there is at least one year of data
        # Check that the DataFrame has the expected number of rows (years)
        expected_years = sales["year"].nunique()
        assert (
            len(sss_growth) <= expected_years
        )  # Should be less than or equal to the number of unique years in sales data
        # Check that the DataFrame is sorted by year
        assert sss_growth.index.is_monotonic_increasing

        logger.info(sss_growth)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
