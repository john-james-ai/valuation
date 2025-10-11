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
# Modified   : Saturday October 11th 2025 07:13:56 pm                                              #
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
        self.financials = financials
        self.sales = sales

    def same_store_sales_growth(self) -> pd.Series:
        """Calculates same-store sales growth."""
        self.sales["previous_year_sales"] = self.sales["sales"].shift(1)
        self.sales["same_store_sales_growth"] = (
            (self.sales["sales"] - self.sales["previous_year_sales"])
            / self.sales["previous_year_sales"]
            * 100
        )
        return self.sales[["year", "same_store_sales_growth"]].dropna()
