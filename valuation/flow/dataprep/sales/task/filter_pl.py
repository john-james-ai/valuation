#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/task/filter_pl.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 10:52:13 pm                                              #
# Modified   : Wednesday October 22nd 2025 10:04:45 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for filtering sales data to remove partial years."""
from typing import Optional

import polars as pl

from valuation.flow.dataprep.task import DataPrepTask
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_FILTER = {
    "category": "string",
    "store": "Int64",
    "week": "Int64",
    "year": "Int64",
    "start": "datetime64[ns]",
    "end": "datetime64[ns]",
    "revenue": "float64",
    "gross_profit": "float64",
    "gross_margin_pct": "float64",
}
NON_NEGATIVE_COLUMNS_FILTER = ["revenue", "gross_profit", "gross_margin_pct"]
MIN_WEEKS_PER_YEAR = 50


# ------------------------------------------------------------------------------------------------ #
class FilterPartialYearsTask(DataPrepTask):

    def __init__(
        self,
        min_weeks: int = 50,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(validation=validation)
        self._min_weeks = min_weeks

    def run(self, df: pl.DataFrame) -> pl.DataFrame:
        # Calculate the number of weekly records available for each year
        year_counts = df.group_by("year").agg(
            pl.len().alias("week_count_in_year")  # pl.len() is the row count per group
        )

        # Join the counts back and filter out years with insufficient weeks
        return (
            df.join(year_counts, on="year", how="left")
            .filter(pl.col("week_count_in_year") >= self._min_weeks)
            .drop("week_count_in_year")
        )
