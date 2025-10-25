#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/task/feature.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Saturday October 25th 2025 03:38:36 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Optional

import polars as pl

from valuation.flow.dataprep.fast.base.task import DataPrepTask
from valuation.flow.dataprep.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_INGEST = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "week": pl.Int64,
    "revenue": pl.Float64,
    "year": pl.Int64,
    "start": pl.Datetime("ns"),
    "end": pl.Datetime("ns"),
}


# ------------------------------------------------------------------------------------------------ #
class FeatureEngineeringTask(DataPrepTask):

    def __init__(
        self,
        validation: Optional[Validation] = None,
    ) -> None:
        """Initializes the ingestion task with the provided configuration."""
        super().__init__()
        self._validation = validation or Validation()

    def run(self, df: pl.LazyFrame, **kwargs) -> pl.LazyFrame:
        """Adds year, start and end dates to the DataFrame based on the week number.

        Args:
            df (pl.LazyFrame): The lazy DataFrame to which dates will be added.
                Must contain a 'week' column.

        Returns:
            pl.LazyFrame: The input lazy DataFrame with added columns and transformations.
        """

        df = (
            df.with_columns(
                pl.concat_str(
                    [
                        pl.col("store").cast(pl.Utf8),
                        pl.lit("_"),
                        pl.col("category").str.to_lowercase().str.replace_all(" ", "_"),
                    ]
                ).alias("unique_id")
            )
            .sort(["unique_id", "week"])
            .rename({"revenue": "y", "end": "ds"})
            .select(["unique_id", "ds", "y"])
        )

        return df
