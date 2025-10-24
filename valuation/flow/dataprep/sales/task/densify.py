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
# Modified   : Friday October 24th 2025 01:29:50 am                                                #
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
    """Create a dense panel (store x category x week) for aggregated sales."""

    def __init__(
        self,
        impute: bool = False,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(validation=validation)
        self._impute = impute

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Run densification to produce a dense panel with revenue filled."""
        logger.debug("Creating Feature Engineered Dataset.")

        # --- 1. Scaffolding ---
        week_date_lookup = df[["week", "end"]].drop_duplicates()
        actual_combinations = df[["store", "category"]].drop_duplicates()

        all_weeks_df = pd.DataFrame({"week": df["week"].unique()})
        all_weeks_df["_key"] = 1
        actual_combinations["_key"] = 1

        scaffold = actual_combinations.merge(all_weeks_df, on="_key").drop("_key", axis=1)
        df_panel = scaffold.merge(week_date_lookup, on="week", how="left")

        # --- 2. Merge Original Data ---
        df_panel = df_panel.merge(
            df[["store", "category", "week", "revenue"]],
            on=["store", "category", "week"],
            how="left",
        )

        # --- 3. Sort for Time Series Operations ---
        df_panel = df_panel.sort_values(["store", "category", "week"]).reset_index(drop=True)

        # --- 4. Imputation / Zero-Filling Logic ---
        if self._impute:
            # Use interpolation/fill for features (revenue column)
            df_final = self._impute_df(df=df_panel.copy())
        else:
            df_final = df_panel.copy()
            df_final["revenue"] = df_final["revenue"].fillna(value=0.0)

        logger.debug("Densified sales data with shape: {}", df_final.shape)
        return df_final

    def _impute_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing revenue values in the densified DataFrame."""

        original_non_null = df["revenue"].notna().sum()
        logger.debug(f"Coverage before imputation: {original_non_null / len(df) * 100:.1f}%")

        # Apply imputation per store-category combination
        df["revenue"] = df.groupby(["store", "category"])["revenue"].transform(impute_series)

        imputed_count = df["revenue"].notna().sum() - original_non_null
        logger.debug(f"Imputed {imputed_count} values ({imputed_count / len(df) * 100:.1f}%)")

        df["revenue"] = df["revenue"].fillna(0.0)

        return df


# Principled imputation for missing revenue values
def impute_series(series):
    """Interpolate gaps in middle, then forward/backward fill edges.

    Args:
        series (pd.Series): A pandas Series of numeric values (e.g., revenue) possibly containing NaNs.

    Returns:
        pd.Series: The input series with gaps interpolated and edge NaNs filled where possible.
    """
    # Linear interpolation for gaps within the series
    series = series.interpolate(method="linear", limit_direction="both")
    # Forward fill for gaps at the start
    series = series.ffill()
    # Backward fill for any remaining gaps at the end
    series = series.bfill()
    return series
