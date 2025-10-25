#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/clean.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Saturday October 25th 2025 03:38:48 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Optional, Union

from loguru import logger
import polars as pl

from valuation.flow.dataprep.fast.base.task import DataPrepTask
from valuation.flow.dataprep.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_CLEAN = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "week": pl.Int64,
    "year": pl.Int64,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "revenue": pl.Float64,
    "gross_profit": pl.Float64,
}

NON_NEGATIVE_COLUMNS_CLEAN = ["revenue"]


# ------------------------------------------------------------------------------------------------ #
class CleanSalesDataTask(DataPrepTask):
    """
    Cleans sales data by removing invalid records, normalizing columns,
    and calculating derived metrics.
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
        Execute the cleaning pipeline on the provided DataFrame.

        Args:
            df: Input DataFrame to be cleaned.
            lazy: If True, return LazyFrame; if False, return DataFrame.
            **kwargs: Additional arguments.

        Returns:
            Union[pl.DataFrame, pl.LazyFrame]: Cleaned DataFrame after all transformations.
        """
        logger.debug("Cleaning sales data.")

        # Convert to LazyFrame for processing
        if isinstance(df, pl.DataFrame):
            df = df.lazy()

        result = (
            df.pipe(self._remove_invalid_records)
            .pipe(self._normalize_columns)
            .pipe(self._calculate_revenue)
            .pipe(self._calculate_gross_profit)
        )

        # Return in requested format
        return result if lazy else result.collect()

    def _remove_invalid_records(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Removes records that do not meet business criteria.

        The rules applied are:
            1. 'ok' flag is 1
            2. 'price' > 0
            3. 'move' > 0
            4. 'qty' >= 1

        Args:
            df: The ingested sales data.

        Returns:
            pl.LazyFrame: The cleaned sales data.
        """
        return df.filter(
            (pl.col("ok") == 1)
            & (pl.col("price") > 0)
            & (pl.col("move") > 0)
            & (pl.col("qty") >= 1)
        )

    def _normalize_columns(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Standardize column names and drop unneeded columns.

        Args:
            df: The cleaned sales data.

        Returns:
            pl.LazyFrame: The cleaned sales data with standardized column names.
        """
        # Rename columns for clarity and drop unneeded ones
        df = df.rename({"profit": "gross_margin_pct"}).drop("sale")

        # Standardize column names to lowercase
        df = df.rename({col: col.lower() for col in df.collect_schema().names()})

        return df

    def _calculate_revenue(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Calculate transaction-level revenue, accounting for product bundles.

        Revenue is derived from bundle price, individual units sold, and bundle size.
        Formula: revenue = (price * move) / qty.

        Args:
            df: The cleaned sales data. Must contain columns 'price', 'move', and 'qty'.

        Returns:
            pl.LazyFrame: DataFrame with an added 'revenue' column.
        """
        return df.with_columns(
            ((pl.col("price") * pl.col("move")) / pl.col("qty")).cast(pl.Float64).alias("revenue")
        )

    def _calculate_gross_profit(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """
        Calculate transaction-level gross profit.

        Gross profit is derived from revenue and gross margin percentage.
        Formula: gross_profit = revenue * (gross_margin_pct / 100).

        Args:
            df: The cleaned sales data. Must contain columns 'revenue' and 'gross_margin_pct'.

        Returns:
            pl.LazyFrame: DataFrame with an added 'gross_profit' column.
        """
        return df.with_columns(
            (pl.col("revenue") * (pl.col("gross_margin_pct") / 100.0))
            .cast(pl.Float64)
            .alias("gross_profit")
        )
