#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/densify.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 07:02:20 am                                              #
# Modified   : Saturday October 25th 2025 03:38:41 am                                              #
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
REQUIRED_COLUMNS_DENSIFY = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "week": pl.Int64,
    "revenue": pl.Float64,
}


# ------------------------------------------------------------------------------------------------ #
class DensifySalesDataTask(DataPrepTask):
    """
    Create a dense panel (store x category x week) for aggregated sales.

    Produces a complete grid of all store-category-week combinations,
    filling missing revenue values with zeros or using imputation.
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
        Run densification to produce a dense panel with revenue filled.

        Args:
            df: Input sales data
            lazy: If True, return LazyFrame; if False, return DataFrame
            **kwargs: Additional arguments

        Returns:
            Union[pl.DataFrame, pl.LazyFrame]: Densified sales panel
        """
        logger.debug("Creating Feature Engineered Dataset.")

        # Convert to DataFrame for processing (densification requires eager operations)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # --- 1. Scaffolding ---
        # Get unique week-date lookup
        week_date_lookup = df.select(["week", "end"]).unique()

        # Get actual store-category combinations that exist in the data
        actual_combinations = df.select(["store", "category"]).unique()

        # Get all unique weeks
        all_weeks_df = df.select("week").unique()

        # Cross join to create full scaffold (all combinations)
        scaffold = actual_combinations.join(all_weeks_df, how="cross")

        # Add date information
        df_panel = scaffold.join(week_date_lookup, on="week", how="left")

        # --- 2. Merge Original Data ---
        df_panel = df_panel.join(
            df.select(["store", "category", "week", "revenue"]),
            on=["store", "category", "week"],
            how="left",
        )

        # --- 3. Sort for Time Series Operations ---
        df_panel = df_panel.sort(["store", "category", "week"])

        # --- 4. Imputation / Zero-Filling Logic ---
        df_final = df_panel.with_columns(pl.col("revenue").fill_null(0.0).alias("revenue"))

        logger.debug(f"Densified sales data with shape: {df_final.shape}")

        # Return in requested format
        return df_final.lazy() if lazy else df_final
