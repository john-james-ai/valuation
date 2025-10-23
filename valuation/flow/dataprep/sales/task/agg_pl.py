#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/task/agg_pl.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 06:18:02 pm                                              #
# Modified   : Wednesday October 22nd 2025 09:48:19 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for aggregating sales data to the store-category-week level."""

from typing import Optional

from loguru import logger
import polars as pl

from valuation.flow.dataprep.task import DataPrepTask
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_AGGREGATE = {
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

NON_NEGATIVE_COLUMNS_AGGREGATE = ["revenue", "gross_profit", "gross_margin_pct"]


# ------------------------------------------------------------------------------------------------ #
class AggregateSalesDataTask(DataPrepTask):
    """Aggregates raw sales records to the store-category-week level.

    This task groups sales records by store, category, and week and computes summed revenue
    and gross profit, takes the first value for year/start/end, and computes gross margin
    percentage as (gross_profit / revenue) * 100.

    Args:
        validation (Optional[Validation]): Validation instance used to validate data.

    """

    def __init__(
        self,
        validation: Optional[Validation] = None,
    ) -> None:
        """Initializes the AggregateSalesDataTask."""
        super().__init__(validation=validation)

    def run(self, df: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """Aggregate the provided LazyFrame and return a collected DataFrame.

        Args:
            df (pl.LazyFrame): Input lazy dataframe to aggregate.
            **kwargs: Additional keyword arguments.

        Return:
            pl.LazyFrame: Aggregated and collected DataFrame.
        """

        logger.debug("Aggregating sales data.")
        # 1: Group by store, category, and week, summing revenue and gross profit
        lazy_agg_plan = (
            df.group_by(["store", "category", "week"])
            .agg(
                [
                    pl.col("revenue").sum(),
                    pl.col("gross_profit").sum(),
                    pl.col("year").first(),
                    pl.col("start").first(),
                    pl.col("end").first(),
                ]
            )
            .with_columns(
                ((pl.col("gross_profit") / (pl.col("revenue") + 1e-6)) * 100).alias(
                    "gross_margin_pct"
                )
            )
            .sort(["store", "category", "week"])
        )

        df = lazy_agg_plan.collect(engine="streaming")
        logger.info("Sales data aggregation complete.")

        return df
