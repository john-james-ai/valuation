#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/app/analysis/company.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 07:04:50 pm                                              #
# Modified   : Saturday October 18th 2025 11:15:33 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

import pandas as pd

from valuation.analysis.base import Dataset
from valuation.analysis.financials import Financials


# ------------------------------------------------------------------------------------------------ #
class CompanyDataset(Dataset):
    """Class for storing company information."""

    def __init__(self, financials: Financials, sales: pd.DataFrame, min_weeks: int = 50) -> None:
        super().__init__(sales, min_weeks=min_weeks)
        self._financials = financials
        self._annual_sales = None
        self._sss_growth = None

    @property
    def financials(self) -> Financials:
        """Gets the financial performance data."""
        return self._financials

    @property
    def annual_sales(self) -> pd.DataFrame:
        """Gets the annual sales data."""
        self._annual_sales = self._compute_annual_sales()
        return self._annual_sales

    @property
    def sss_growth(self) -> pd.DataFrame:
        """Gets the same-store sales growth data."""
        self._sss_growth = self._compute_same_store_sales_growth()
        return self._sss_growth

    def _compute_annual_sales(self) -> pd.DataFrame:
        """Calculates same-store sales growth."""

        if self._annual_sales is not None:
            return self._annual_sales

        # 1. Group by year and sum the revenue. Ensure data is sorted chronologically.
        annual_sales = self.dataset.data.groupby("year")["revenue"].sum().sort_index().reset_index()

        # 2. Use pct_change() to calculate YoY growth in a single, vectorized operation.
        # This calculates (current_year_revenue / previous_year_revenue) - 1
        annual_sales["yoy_growth"] = annual_sales["revenue"].pct_change() * 100

        return annual_sales

    def _compute_same_store_sales_growth(self) -> pd.DataFrame:
        """Calculates same-store sales growth.
        Same-store sales growth compares revenue from stores that were open in both
        the previous year and the current year, isolating growth from new stores.
        """

        if self._sss_growth is not None:
            return self._sss_growth

        growth_results = []

        # Iterate through each pair of consecutive years
        for i in range(1, len(self.dataset.years)):
            previous_year = self.dataset.years[i - 1]
            current_year = self.dataset.years[i]

            stores_previous = set(
                self.dataset.data[self.dataset.data["year"] == previous_year]["store"].unique()
            )
            stores_current = set(
                self.dataset.data[self.dataset.data["year"] == current_year]["store"].unique()
            )
            comp_stores = stores_previous.intersection(stores_current)

            if not comp_stores:
                continue  # Skip if there are no common stores

            # 2. Filter for only the comparable stores and the two relevant years
            comp_stores_data = self.dataset.data[
                self.dataset.data["store"].isin(comp_stores)
                & self.dataset.data["year"].isin([previous_year, current_year])
            ]

            # 3. Aggregate the revenue for these stores in this period
            revenue_by_year = comp_stores_data.groupby("year")["revenue"].sum()

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

        return pd.DataFrame(growth_results)
