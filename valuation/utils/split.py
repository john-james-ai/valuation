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

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class TimeSeriesDataSplitter:
    """Split data into training/validation and test sets based on year.

    Args:
        year_to_split (int): Year used to split test data (rows with this year become test set).
        validation (Optional[Validation]): Optional Validation instance used by the task.
    """

    def __init__(
        self,
        year_to_split: int = 1996,
    ) -> None:
        self._year_to_split = year_to_split

    def split(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Split the DataFrame into train_val and test partitions based on the configured year.

        Args:
            df (pd.DataFrame): Input DataFrame containing a 'ds' datetime column.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with keys 'train_val' and 'test' containing respective DataFrames.
        """
        df["year"] = pd.DatetimeIndex(df["ds"]).year
        train_val_df = df[df["year"] < self._year_to_split].reset_index(drop=True)
        test_df = df[df["year"] == self._year_to_split].reset_index(drop=True)

        train_val_df = train_val_df.drop(columns=["year"])
        test_df = test_df.drop(columns=["year"])
        return {"train_val": train_val_df, "test": test_df}
