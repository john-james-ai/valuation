#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/filter.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 10:52:13 pm                                              #
# Modified   : Wednesday October 22nd 2025 02:14:58 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for filtering sales data to remove partial years."""
from typing import Optional

from dataclasses import dataclass

import pandas as pd

from valuation.core.dataclass import DataClass
from valuation.flow.dataprep.task import DataPrepTaskResult, SISODataPrepTask
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_FILTER = {
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
NON_NEGATIVE_COLUMNS_FILTER = ["revenue", "gross_profit", "gross_margin_pct"]


# ------------------------------------------------------------------------------------------------ #
@dataclass
class FilterPartialYearsTaskResult(DataPrepTaskResult):
    """Holds the results of the FilterPartialYearsTask execution."""


# ------------------------------------------------------------------------------------------------ #


@dataclass
class FilterPartialYearsTaskConfig(DataClass):
    """Configuration for filtering sales data.

    Args:
        min_weeks (int): Minimum number of weeks a year must have to be included.
    """

    min_weeks: int = 50


# ------------------------------------------------------------------------------------------------ #
class FilterPartialYearsTask(SISODataPrepTask):

    _config: FilterPartialYearsTaskConfig

    def __init__(
        self,
        config: FilterPartialYearsTaskConfig,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(config=config, validation=validation)

    def _execute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executes the filtering of partial years.

        Args:
            df (pd.DataFrame): Sales data DataFrame.

        Returns:
            DatasetContainer: Container with filtered data and list of full years.
        """

        """Returns a DataFrame containing only full years of data."""
        # 1. Group by year and count the number of unique weeks in each group
        weeks_per_year = df.groupby("year")["week"].nunique()

        # 2. Filter to get a sorted list of years that have more than min_weeks of data
        years = weeks_per_year[weeks_per_year >= self._config.min_weeks].index.tolist()

        # 3. Filter the original DataFrame to include only rows from the full years
        df_out = df[df["year"].isin(years)].copy()

        return df_out
