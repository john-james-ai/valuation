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
# Modified   : Saturday October 11th 2025 09:20:34 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect

from loguru import logger
import pandas as pd
import pytest

from valuation.analysis.company import Company
from valuation.analysis.financials import (
    BalanceSheet,
    CashFlowStatement,
    FinancialMetrics,
    FinancialPerformance,
    FinancialRatios,
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
        fin = FinancialPerformance.from_dict(financials)
        # Check that the financials object is created correctly
        assert isinstance(fin, FinancialPerformance)
        assert isinstance(fin.income_statement, IncomeStatement)
        assert isinstance(fin.balance_sheet, BalanceSheet)
        assert isinstance(fin.cash_flow_statement, CashFlowStatement)
        assert isinstance(fin.financial_ratios, FinancialRatios)
        assert isinstance(fin.financial_metrics, FinancialMetrics)

        assert fin.balance_sheet.cash_and_equivalents == 32735.00
        assert fin.balance_sheet.accounts_receivable == 16723.00
        assert fin.balance_sheet.inventory == 203411.00
        assert fin.balance_sheet.prepaid_expenses == 21860.00
        assert fin.balance_sheet.other_current_assets == 459252.00
        assert fin.balance_sheet.current_assets == 274729.00
        assert fin.balance_sheet.property_plant_equipment == 368224.00
        assert fin.balance_sheet.total_assets == 1102205.00
        assert fin.balance_sheet.accounts_payable == 187787.00
        assert fin.balance_sheet.short_term_debt == 16723
        assert fin.balance_sheet.long_term_debt == 400644.00
        assert fin.balance_sheet.current_liabilities == 308434.00
        assert fin.balance_sheet.total_liabilities == 1102205.00
        assert fin.balance_sheet.total_equity == 179071.00
        assert fin.cash_flow_statement.net_income == -6932.00
        assert fin.cash_flow_statement.operating_activities == 41120.00
        assert fin.cash_flow_statement.investing_activities == -48561.00
        assert fin.cash_flow_statement.financing_activities == -15375.00
        assert fin.cash_flow_statement.net_change_in_cash == -22816.00
        assert fin.cash_flow_statement.capital_expenditures == -49588.00
        assert fin.cash_flow_statement.deprecated_and_amortization == 45924.00
        assert fin.income_statement.revenue == 2511962.00
        assert fin.income_statement.cogs == 1932994.00
        assert fin.income_statement.sga == 491359.00
        assert fin.income_statement.gross_profit == 578968.00
        assert fin.income_statement.operating_income == 77109.00
        assert fin.income_statement.ebit == 88976.00
        assert fin.income_statement.interest_expense == 67555.00
        assert fin.income_statement.earnings_before_taxes == 6813.00
        assert fin.income_statement.income_tax_expense == 7385.00
        assert fin.income_statement.net_income == -6932.00
        assert fin.financial_metrics.sales_growth == 78238.00
        assert fin.financial_metrics.sales_growth_rate == 3.11
        assert fin.financial_metrics.capex_percentage == 1.97
        assert fin.financial_metrics.net_working_capital_change == 481.00
        assert fin.financial_ratios.gross_profit_margin == 23.05
        assert fin.financial_ratios.operating_margin == 3.07
        assert fin.financial_ratios.net_profit_margin == -0.28
        assert fin.financial_ratios.ebit_margin == 3.49
        assert fin.financial_ratios.ebitda == 134900.00
        assert fin.financial_ratios.ebitda_margin == 3.54
        assert fin.financial_ratios.return_on_assets == -0.6
        assert fin.financial_ratios.return_on_equity == -0.6
        assert fin.financial_ratios.return_on_invested_capital == 3.4
        assert fin.financial_ratios.effective_interest_rate == 9.2
        assert fin.financial_ratios.depreciation_percentage == 1.83
        assert fin.financial_ratios.working_capital == -33705.00
        assert fin.financial_ratios.current_ratio == 0.89
        assert fin.financial_ratios.quick_ratio == 0.23
        assert fin.financial_ratios.debt_to_equity_ratio == 0.54
        assert fin.financial_ratios.inventory_turnover == 9.50
        assert fin.financial_ratios.dio == 38.41
        assert fin.financial_ratios.dpo == 35.46
        assert fin.financial_ratios.sga_percentage == 19.56
        assert fin.financial_ratios.cogs_growth_rate == 2.59

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
        fin = FinancialPerformance.from_dict(financials)
        company = Company(financials=fin, sales=sales)
        assert isinstance(company.financials, FinancialPerformance)
        assert isinstance(company.annual_sales, pd.DataFrame)
        annual_sales = company.annual_sales

        assert not annual_sales.empty
        assert "year" in annual_sales.index.names
        assert "yoy_growth" in annual_sales.index.names
        assert "revenue" in annual_sales.index.names

        # Check that there are no NaN values in the DataFrames
        assert not annual_sales.isnull().values.any()

        # Check that the years are in ascending order
        years = annual_sales.index.get_level_values("year").tolist()
        assert years == sorted(years)

        # Check that the yoy_growth values are floats and within a reasonable range
        yoy_growth_values = annual_sales["yoy_growth"].tolist()
        for value in yoy_growth_values:
            assert isinstance(value, float) or value is None  # Allow None for the first year
            if value is not None:
                assert -1.0 < value < 10.0  # Assuming growth rates between -100% and +1000%

        # Check that the revenue values are positive floats
        revenue_values = annual_sales["revenue"].tolist()
        for value in revenue_values:
            assert isinstance(value, float)
            assert value >= 0.0  # Revenue should be non-negative

        # Check that the index has the correct name
        assert annual_sales.index.name == "year"
        assert len(annual_sales) > 0  # Ensure there is at least one year of data

        # Check that the revenue values are in a realistic range
        for value in revenue_values:
            assert (
                0 < value < 1e10
            )  # Assuming revenue should be less than 10 billion for this company

        # Check that the yoy_growth values are calculated correctly
        for i in range(1, len(revenue_values)):
            if revenue_values[i - 1] != 0:  # Prevent division by zero
                expected_growth = (revenue_values[i] / revenue_values[i - 1]) - 1
                assert (
                    abs(yoy_growth_values[i] - expected_growth) < 1e-6
                )  # Allow for floating point precision
            else:
                assert (
                    yoy_growth_values[i] is None
                )  # If previous revenue is zero, growth should be None
        # Check that the first year's yoy_growth is None
        assert yoy_growth_values[0] is None

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
        company = Company(financials=financials, sales=sales)
        assert isinstance(company.financials, FinancialPerformance)
        # Check that sss_growth is a DataFrame and has the expected structure
        assert isinstance(company.sss_growth, pd.DataFrame)

        # Check that the DataFrame is not empty
        sss_growth = company.sss_growth
        assert not sss_growth.empty

        # Check that the DataFrame has the expected columns
        assert "year" in sss_growth.index.names
        assert "sss_growth" in sss_growth.index.names
        assert "store_count" in sss_growth.index.names

        # Check that there are no NaN values in the DataFrame
        assert not sss_growth.isnull().values.any()
        # Check that the years are in ascending order
        years = sss_growth.index.get_level_values("year").tolist()
        assert years == sorted(years)
        # Check that the sss_growth values are floats and within a reasonable range
        sss_growth_values = sss_growth["sss_growth"].tolist()
        for value in sss_growth_values:
            assert isinstance(value, float) or value is None  # Allow None for the first year
            if value is not None:
                assert -1.0 < value < 10.0  # Assuming growth rates between -100% and +1000%
        # Check that the store_count values are positive integers
        store_count_values = sss_growth["store_count"].tolist()
        for value in store_count_values:
            assert isinstance(value, int)
            assert value >= 0  # Store count should be non-negative

        # Check that the index has the correct name
        assert sss_growth.index.name == "year"
        assert len(sss_growth) > 0  # Ensure there is at least one year of data
        # Check that the DataFrame has the expected number of rows (years)
        expected_years = sales["year"].nunique()
        assert (
            len(sss_growth) <= expected_years
        )  # Should be less than or equal to the number of unique years in sales data
        # Check that the DataFrame is sorted by year
        assert sss_growth.index.is_monotonic_increasing

        # Check that the sss_growth values are calculated correctly
        revenue_by_year_store = (
            sales.groupby(["year", "store"])["revenue"].sum().unstack(fill_value=0)
        )
        growth_results = []
        years = sorted(revenue_by_year_store.index)
        for i in range(1, len(years)):
            previous_year = years[i - 1]
            current_year = years[i]
            growth_sum = 0
            store_count = 0
            for store in revenue_by_year_store.columns:
                revenue_prev = revenue_by_year_store.at[previous_year, store]
                revenue_curr = revenue_by_year_store.at[current_year, store]
                if revenue_prev > 0:
                    growth = (revenue_curr / revenue_prev) - 1
                    growth_sum += growth
                    store_count += 1
            avg_growth = (
                (growth_sum / store_count) * 100 if store_count > 0 else None
            )  # Convert to percentage
            growth_results.append(
                {"year": current_year, "sss_growth": avg_growth, "store_count": store_count}
            )
        expected_sss_growth_df = pd.DataFrame(growth_results).set_index("year")
        pd.testing.assert_frame_equal(
            sss_growth, expected_sss_growth_df, check_dtype=False, check_exact=False, rtol=1e-5
        )
        logger.info(sss_growth)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
