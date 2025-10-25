#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/aggregate.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 06:18:02 pm                                              #
# Modified   : Saturday October 25th 2025 03:38:54 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for aggregating sales data to the store-category-week level."""

from typing import Optional, Union

from loguru import logger
import polars as pl

from valuation.flow.dataprep.base.task import DataPrepTask
from valuation.flow.dataprep.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_AGGREGATE = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "week": pl.Int64,
    "year": pl.Int64,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "revenue": pl.Float64,
    "gross_profit": pl.Float64,
    "gross_margin_pct": pl.Float64,
}

NON_NEGATIVE_COLUMNS_AGGREGATE = ["revenue", "gross_profit", "gross_margin_pct"]


# ------------------------------------------------------------------------------------------------ #
class AggregateSalesDataTask(DataPrepTask):
    """
    Aggregates sales data by store, category, and week.

    Calculates summed revenue and gross profit, then derives the true
    gross margin percentage from the aggregated totals.
    """

    def __init__(
        self,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__()
        self._validation = validation or Validation()

    def run(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        lazy: bool = False,
        **kwargs,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Execute the aggregation pipeline on the provided DataFrame.

        Args:
            df: Input DataFrame to be aggregated.
            lazy: If True, return LazyFrame; if False, return DataFrame.
            **kwargs: Additional arguments.

        Returns:
            Union[pl.DataFrame, pl.LazyFrame]: Aggregated DataFrame.
        """
        logger.debug("Aggregating sales data.")

        # Convert to LazyFrame for processing
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        # Group by store, category, and week, summing revenue and gross profit
        aggregated = df.group_by(["store", "category", "week"]).agg(
            [
                pl.col("revenue").sum().alias("revenue"),
                pl.col("gross_profit").sum().alias("gross_profit"),
                pl.col("year").first().alias("year"),
                pl.col("start").first().alias("start"),
                pl.col("end").first().alias("end"),
            ]
        )

        # Calculate the true margin from the summed totals
        # Add a small epsilon to avoid division by zero if revenue is 0
        aggregated = aggregated.with_columns(
            (pl.col("gross_profit") / (pl.col("revenue") + 1e-6) * 100).alias("gross_margin_pct")
        )

        # Sort for reproducibility
        aggregated = aggregated.sort(["store", "category", "week"])

        # Return in requested format
        return aggregated if lazy else aggregated.collect()
