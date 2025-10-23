#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/task/densify.py                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 07:02:20 am                                              #
# Modified   : Thursday October 23rd 2025 12:45:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Optional

from loguru import logger
import pandas as pd

from valuation.flow.dataprep.task import DataPrepTask
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_DENSIFY = {
    "category": "string",
    "store": "Int64",
    "week": "Int64",
    "revenue": "float64",
}


# ------------------------------------------------------------------------------------------------ #
class DensifySalesDataTask(DataPrepTask):
    """Create a dense panel (store x category x week) for aggregated sales.

    Args:
        validation (Optional[Validation]): Optional Validation instance used by the task.
    """

    def __init__(
        self,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(validation=validation)

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run densification to produce a dense panel with revenue filled.

        Args:
            df (pd.DataFrame): Input DataFrame containing at minimum the required columns.

        Returns:
            pd.DataFrame: Dense DataFrame with rows for every (store, category, week) combination
                and missing revenues filled with 0.0.
        """
        logger.debug("Creating Feature Engineered Dataset.")
        week_date_lookup = df[["week", "end"]].drop_duplicates()
        # Extract required columns
        # Obtain all unique stores, categories, weeks
        all_stores = df["store"].unique()
        all_categories = df["category"].unique()
        all_weeks = df["week"].unique()

        # Create a MultiIndex of all combinations
        scaffold = pd.MultiIndex.from_product(
            [all_stores, all_categories, all_weeks], names=["store", "category", "week"]
        ).to_frame(index=False)

        # Merge to add start dates back
        df_panel = pd.merge(scaffold, week_date_lookup, on="week", how="left")

        # Merge with original data to create dense panel
        df_panel = pd.merge(
            df_panel,
            df[["store", "category", "week", "revenue"]],
            on=["store", "category", "week"],
            how="left",
        )

        # Fill missing revenue with 0.0
        df_panel["revenue"] = df_panel["revenue"].fillna(0.0)

        logger.debug("Densified sales data with shape: {}", df_panel.shape)

        return df_panel
