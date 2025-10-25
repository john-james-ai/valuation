#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/task/feature.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Saturday October 25th 2025 03:47:38 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Optional

import pandas as pd

from valuation.flow.dataprep.base.task import DataPrepTask
from valuation.flow.dataprep.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_INGEST = {
    "category": "string",
    "store": "Int64",
    "week": "Int64",
    "revenue": "float64",
    "year": "Int64",
    "start": "datetime64[ns]",
    "end": "datetime64[ns]",
}


# ------------------------------------------------------------------------------------------------ #
class FeatureEngineeringTask(DataPrepTask):

    def __init__(
        self,
        validation: Optional[Validation] = None,
    ) -> None:
        """Initializes the ingestion task with the provided configuration."""
        super().__init__(validation=validation)

    def run(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Adds year, start and end dates to the DataFrame based on the week number.

        Args:
            df (pd.DataFrame): The DataFrame to which dates will be added. Must contain
                a 'week' column.
            week_decode_filepath (Path): The path to the week decode CSV file.
        Returns:
            pd.DataFrame: The input DataFrame with added 'start_date' and 'end_date' columns.
        """

        df["unique_id"] = (
            df["store"].astype("string")
            + "_"
            + df["category"].str.lower().replace(" ", "_").astype("string")
        )
        df = df.sort_values(by=["unique_id", "week"]).reset_index(drop=True)
        df = df.rename(columns={"revenue": "y"})
        df = df.rename(columns={"end": "ds"})

        return df[["unique_id", "ds", "y"]]
