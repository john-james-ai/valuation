#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/ingest.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Saturday October 25th 2025 05:00:41 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Optional, Union

import polars as pl

from valuation.flow.dataprep.fast.base.task import DataPrepTask
from valuation.flow.dataprep.fast.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_INGEST = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "upc": pl.Int64,
    "week": pl.Int64,
    "qty": pl.Int64,
    "move": pl.Int64,
    "ok": pl.Int64,
    "sale": pl.Utf8,
    "price": pl.Float64,
    "profit": pl.Float64,
    "year": pl.Int32,  # changed to Int32 to match actual parquet/csv load
    "start": pl.Date,
    "end": pl.Date,
}

NON_NEGATIVE_COLUMNS_INGEST = ["qty", "move", "price"]


# ------------------------------------------------------------------------------------------------ #
class IngestSalesDataTask(DataPrepTask):
    """
    Ingests a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        week_decode_table: DataFrame containing start and end dates for each week number.
        validation: Optional validation configuration.
    """

    def __init__(
        self,
        week_decode_table: Union[pl.DataFrame, pl.LazyFrame],
        validation: Optional[Validation] = None,
    ) -> None:
        """Initializes the ingestion task with the provided configuration."""
        super().__init__()
        self._validation = validation or Validation()
        self._week_decode_table = week_decode_table

    def run(
        self,
        df: Union[pl.DataFrame, pl.LazyFrame],
        category: str,
        lazy: bool = False,
        **kwargs,
    ) -> Union[pl.DataFrame, pl.LazyFrame]:
        """
        Run the ingestion task.

        Args:
            df: Input sales data
            category: Category name to assign
            lazy: If True, return LazyFrame; if False, return DataFrame
            **kwargs: Additional arguments

        Returns:
            Union[pl.DataFrame, pl.LazyFrame]: Processed sales data with category and dates
        """
        # Convert to LazyFrame if needed for processing
        was_eager = isinstance(df, pl.DataFrame)
        if was_eager:
            df = df.lazy()

        # Add category and dates to the data using polars-compatible calls
        df_out = self._add_category(df, category=category)
        df_out = self._add_dates(df_out, week_dates=self._week_decode_table)

        # Convert column names to lowercase using polars LazyFrame API
        # df_out.columns works for LazyFrame and DataFrame
        cols = list(df_out.collect_schema().names())
        rename_map = {col: col.lower() for col in cols}
        df_out = df_out.rename(rename_map)

        # Return in the requested format
        if lazy:
            return df_out
        else:
            return df_out.collect()

    def _add_dates(
        self,
        df: pl.LazyFrame,
        week_dates: Union[pl.DataFrame, pl.LazyFrame],
    ) -> pl.LazyFrame:
        """
        Adds year, start, and end dates to the DataFrame based on the week number.
        """

        # Collect week_dates if lazy
        week_df = week_dates.collect() if isinstance(week_dates, pl.LazyFrame) else week_dates

        # Parse dates - simple and direct
        week_df = week_df.with_columns(
            [
                pl.col("START").str.to_date("%m/%d/%Y"),
                pl.col("END").str.to_date("%m/%d/%Y"),
                pl.col("WEEK").cast(pl.Int64),
            ]
        )

        # Ensure WEEK column matches
        df = df.with_columns(pl.col("WEEK").cast(pl.Int64))

        # Join
        df = df.join(week_df.lazy(), on="WEEK", how="left")

        # Add year
        df = df.with_columns(pl.col("END").dt.year().alias("YEAR"))

        return df

    def _add_category(
        self,
        df: pl.LazyFrame,
        category: str,
    ) -> pl.LazyFrame:
        """
        Adds a category column to the DataFrame.

        Args:
            df (pl.LazyFrame): The LazyFrame to which the category will be added.
            category (str): The category name to assign.

        Returns:
            pl.LazyFrame: The input DataFrame with an added 'CATEGORY' column.
        """
        return df.with_columns(pl.lit(category).alias("CATEGORY"))
