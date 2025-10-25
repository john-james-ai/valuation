#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/split.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Thursday October 23rd 2025 09:33:17 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Splits data into training/validation and test sets."""
from typing import Dict

import polars as pl


# ------------------------------------------------------------------------------------------------ #
class TimeSeriesDataSplitter:
    """Split data into training/validation and test sets based on year.

    Args:
        year_to_split (int): Year used to split test data (rows with this year become test set).
    """

    def __init__(
        self,
        year_to_split: int = 1996,
    ) -> None:
        self._year_to_split = year_to_split

    def split(self, df: pl.DataFrame | pl.LazyFrame) -> Dict[str, pl.DataFrame]:
        """Split the DataFrame into train_val and test partitions based on the configured year.

        Args:
            df (pl.DataFrame | pl.LazyFrame): Input Polars DataFrame or LazyFrame containing a 'ds' datetime column.

        Returns:
            Dict[str, pl.DataFrame]: Dictionary with keys 'train_val' and 'test' containing respective Polars DataFrames.
        """
        # Materialize LazyFrame if necessary
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # Ensure 'ds' exists
        if "ds" not in df.columns:
            raise ValueError("Input DataFrame must contain a 'ds' column with datetimes.")

        # Try to extract year; if ds is not datetime, attempt to parse common M/D/YYYY strings
        try:
            df = df.with_columns(pl.col("ds").dt.year().alias("year"))
        except Exception:
            # Trim whitespace and attempt parsing M/D/YYYY (non-strict) then extract year
            df = df.with_columns(
                pl.col("ds")
                .str.replace_all(r"^\s+|\s+$", "")
                .str.strptime(pl.Datetime("ns"), "%m/%d/%Y", strict=False)
                .alias("ds")
            )
            df = df.with_columns(pl.col("ds").dt.year().alias("year"))

        # Partition
        train_val_df = df.filter(pl.col("year") < self._year_to_split).drop("year")
        test_df = df.filter(pl.col("year") == self._year_to_split).drop("year")

        return {"train_val": train_val_df, "test": test_df}
