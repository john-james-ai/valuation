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
# Modified   : Thursday October 23rd 2025 11:17:54 pm                                              #
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
        # Create a week and date lookup table
        week_date_lookup = df[["week", "end"]].drop_duplicates()
        # Create a dataframe of all actual store/category combinations
        actual_combinations = df[["store", "category"]].drop_duplicates()

        # Create complete week scaffold
        all_weeks_df = pd.DataFrame({"week": df["week"].unique()})
        all_weeks_df["_key"] = 1
        actual_combinations["_key"] = 1

        # Cross join actual combinations with all weeks
        scaffold = actual_combinations.merge(all_weeks_df, on="_key").drop("_key", axis=1)

        # Merge to add dates
        df_panel = scaffold.merge(week_date_lookup, on="week", how="left")

        # Merge with original data to create dense panel
        df_panel = df_panel.merge(
            df[["store", "category", "week", "revenue"]],
            on=["store", "category", "week"],
            how="left",
        )

        logger.debug("Densified sales data with shape: {}", df_panel.shape)
        original_non_null = df_panel["revenue"].notna().sum()
        logger.debug(f"Coverage before imputation: {original_non_null / len(df_panel) * 100:.1f}%")

        # Sort by store, category, and week for proper time series ordering
        df_panel = df_panel.sort_values(["store", "category", "week"]).reset_index(drop=True)

        # Apply imputation per store-category combination
        df_panel["revenue"] = df_panel.groupby(["store", "category"])["revenue"].transform(
            impute_series
        )
        imputed_count = df_panel["revenue"].notna().sum() - original_non_null
        logger.debug(f"Imputed {imputed_count} values ({imputed_count / len(df_panel) * 100:.1f}%)")
        logger.debug(
            f"Coverage after imputation: {df_panel['revenue'].notna().sum() / len(df_panel) * 100:.1f}%"
        )
        logger.debug(f"Remaining NaN values: {df_panel['revenue'].isna().sum()}")

        return df_panel


# Principled imputation for missing revenue values
def impute_series(series):
    """Interpolate gaps in middle, then forward/backward fill edges."""
    # Linear interpolation for gaps within the series
    series = series.interpolate(method="linear", limit_direction="both")
    # Forward fill for gaps at the start
    series = series.ffill()
    # Backward fill for any remaining gaps at the end
    series = series.bfill()
    return series
