#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/filter.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 10:52:13 pm                                              #
# Modified   : Saturday October 25th 2025 03:38:30 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for filtering sales data to remove partial years."""
from typing import Optional

import polars as pl

from valuation.flow.dataprep.fast.base.task import DataPrepTask
from valuation.flow.dataprep.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_FILTER = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "week": pl.Int64,
    "year": pl.Int64,
    "start": pl.Datetime("ns"),
    "end": pl.Datetime("ns"),
    "revenue": pl.Float64,
    "gross_profit": pl.Float64,
    "gross_margin_pct": pl.Float64,
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
        super().__init__()
        self._validation = validation or Validation()
        self._min_weeks = min_weeks

    def run(self, df: pl.DataFrame, **kwargs) -> pl.DataFrame:
        # Calculate the number of weekly records available for each year
        return df.filter(pl.col("week").n_unique().over("year") >= self._min_weeks)
