#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/aggregate.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 06:18:02 pm                                              #
# Modified   : Tuesday October 21st 2025 05:42:13 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for aggregating sales data to the store-category-week level."""

from loguru import logger
import pandas as pd

from valuation.flow.dataprep.task import DataPrepTaskConfig, SISODataPrepTask
from valuation.infra.store.dataset import DatasetStore

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
class AggregateSalesDataTask(SISODataPrepTask):
    """Aggregates a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(
        self,
        config: DataPrepTaskConfig,
        dataset_store: DatasetStore,
    ) -> None:
        super().__init__(config=config, dataset_store=dataset_store)

    def _execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Runs the ingestion process on the provided DataFrame.
        Args:
            data (pd.DataFrame): The raw sales data DataFrame.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The processed sales data with added category and date information.

        """
        logger.debug("Aggregating sales data.")
        # 1: Group by store, category, and week, summing revenue and gross profit
        aggregated = (
            df.groupby(["store", "category", "week"])
            .agg(
                revenue=("revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
                year=("year", "first"),
                start=("start", "first"),
                end=("end", "first"),
            )
            .reset_index()
        )
        # Step 2: Calculate the true margin from the summed totals
        # We add a small epsilon to avoid division by zero if revenue is 0
        aggregated["gross_margin_pct"] = (
            aggregated["gross_profit"] / (aggregated["revenue"] + 1e-6) * 100
        )

        # Sort for reproducibility
        aggregated = aggregated.sort_values(["store", "category", "week"])

        return aggregated
