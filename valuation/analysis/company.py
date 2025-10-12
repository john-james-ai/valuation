#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/company.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 07:04:50 pm                                              #
# Modified   : Saturday October 11th 2025 08:44:52 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

import pandas as pd

from valuation.analysis.financials import FinancialPerformance


# ------------------------------------------------------------------------------------------------ #
class Company:
    """Class for storing company information."""

    def __init__(self, financials: FinancialPerformance, sales: pd.DataFrame) -> None:
        self._financials = financials
        self._sales = sales
        self._annual_sales = None
        self._sss_growth = None

    @property
    def financials(self) -> FinancialPerformance:
        """Gets the financial performance data."""
        return self._financials

    @property
    def annual_sales(self) -> pd.DataFrame:
        """Gets the annual sales data."""
        if self._annual_sales is None:
            self._compute_annual_sales()
        return self._annual_sales.transpose() if self._annual_sales is not None else pd.DataFrame()

    @property
    def sss_growth(self) -> pd.DataFrame:
        """Gets the same-store sales growth data."""
        if self._sss_growth is None:
            self._compute_same_store_sales_growth()
        return self._sss_growth.transpose() if self._sss_growth is not None else pd.DataFrame()

    def _compute_annual_sales(self) -> None:
        """Calculates same-store sales growth."""
        # 1. Group by year and count the number of unique weeks in each group
        weeks_per_year = self._sales.groupby("year")["week"].nunique()

        # 2. Filter to get a list of years that have more than 50 weeks of data
        full_years = weeks_per_year[weeks_per_year > 50].index

        # 3. You can now filter your DataFrame to work with only these full years
        df_full_years = self._sales[self._sales["year"].isin(full_years)]

        # 4. Group by year and sum the revenue. Ensure data is sorted chronologically.
        self._annual_sales = (
            df_full_years.groupby("year")["revenue"].sum().sort_index().reset_index()
        )

        # 5. Use pct_change() to calculate YoY growth in a single, vectorized operation.
        # This calculates (current_year_revenue / previous_year_revenue) - 1
        self._annual_sales["yoy_growth"] = self._annual_sales["revenue"].pct_change()

    def _compute_same_store_sales_growth(self) -> None:
        """Calculates a time series of year-over-year Same-Store Sales growth.

        Args:
            df (pd.DataFrame): An aggregated DataFrame containing 'year', 'store',
                and 'revenue' columns.

        Returns:
            pd.DataFrame: A DataFrame showing the SSS growth for each year.
        """
        years = sorted(self._sales["year"].unique())
        growth_results = []

        # Iterate through each pair of consecutive years
        for i in range(1, len(years)):
            previous_year = years[i - 1]
            current_year = years[i]

            # 1. Identify stores that were active in both years
            stores_previous = set(
                self._sales[self._sales["year"] == previous_year]["store"].unique()
            )
            stores_current = set(
                self._sales[self._sales["year"] == current_year]["store"].unique()
            )
            comp_stores = stores_previous.intersection(stores_current)

            if not comp_stores:
                continue  # Skip if there are no common stores

            # 2. Filter for only the comparable stores and the two relevant years
            self._sales_period = self._sales[
                self._sales["store"].isin(comp_stores)
                & self._sales["year"].isin([previous_year, current_year])
            ]

            # 3. Aggregate the revenue for these stores in this period
            revenue_by_year = self._sales_period.groupby("year")["revenue"].sum()

            # 4. Calculate the growth rate
            revenue_prev = revenue_by_year.get(previous_year, 0)
            revenue_curr = revenue_by_year.get(current_year, 0)

            if revenue_prev > 0:
                growth = ((revenue_curr / revenue_prev) - 1) * 100  # Convert to percentage
                growth_results.append(
                    {
                        "year": current_year,
                        "sss_growth": growth,
                        "num_comp_stores": len(comp_stores),
                    }
                )

        self._sss_growth = pd.DataFrame(growth_results)
