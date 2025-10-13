#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/profile.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 13th 2025 08:32:34 am                                                #
# Modified   : Monday October 13th 2025 08:46:18 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from valuation.dataprep.base import Task, TaskConfig
from valuation.utils.data import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ProfileTaskConfig(TaskConfig):
    """Holds all parameters for the sales data profiling process."""

    raw_data_directory: Path
    core_features: list[str] = field(
        default_factory=lambda: ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT", "OK"]
    )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Profile(DataClass):
    dataset_name: str
    stores: int
    weeks: int
    num_records: int
    num_columns: int
    missing_values: int
    missing_values_pct: float
    invalid_records: int
    invalid_records_pct: float
    memory_usage_mb: float
    file_size_mb: float


# ------------------------------------------------------------------------------------------------ #
class ProfileTask(Task):
    """Profiles a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(self, config: ProfileTaskConfig) -> None:
        super().__init__(config=config)

    def _execute(self) -> pd.DataFrame:
        """Runs the ingestion process on the provided DataFrame.
        Args:
            data (pd.DataFrame): The raw sales data DataFrame.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The processed sales data with added category and date information.

        """

        df = self._load(filepath=self._config.input_location)

        profile = Profile(
            dataset_name=self._config.dataset_name,
            stores=df["STORE"].nunique() if "STORE" in df.columns else 0,
            weeks=df["WEEK"].nunique() if "WEEK" in df.columns else 0,
            num_records=len(df),
            num_columns=len(df.columns),
            missing_values=df[self._config.core_features].isnull().sum().sum(),  # type: ignore
            missing_values_pct=df[self._config.core_features].isnull().sum().sum() / df.shape[0] * 100,  # type: ignore
            invalid_records=len(df[df["OK"] == 0]) if "OK" in df.columns else 0,
            invalid_records_pct=(
                len(df[df["OK"] == 0]) / df.shape[0] * 100 if "OK" in df.columns else 0
            ),
            memory_usage_mb=df.memory_usage(deep=True).sum() / (1024 * 1024),
            file_size_mb=(
                self._config.input_location.stat().st_size / (1024 * 1024)
                if self._config.input_location.exists()
                else 0
            ),
        )

        return profile.as_df()
