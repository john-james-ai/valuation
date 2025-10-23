#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/ingest.py                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Wednesday October 22nd 2025 12:07:21 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Optional

import pandas as pd

from valuation.asset.dataset.base import DTYPES
from valuation.flow.dataprep.task import DataPrepTask
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_INGEST = {
    "category": "string",
    "store": "Int64",
    "upc": "Int64",
    "week": "Int64",
    "qty": "Int64",
    "move": "Int64",
    "ok": "Int64",
    "sale": "string",
    "price": "float64",
    "profit": "float64",
    "year": "Int64",
    "start": "datetime64[ns]",
    "end": "datetime64[ns]",
}

NON_NEGATIVE_COLUMNS_INGEST = ["qty", "move", "price"]


# ------------------------------------------------------------------------------------------------ #
class IngestSalesDataTask(DataPrepTask):
    """Ingests a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(
        self,
        week_decode_table: pd.DataFrame,
        validation: Optional[Validation] = None,
    ) -> None:
        """Initializes the ingestion task with the provided configuration."""
        super().__init__(validation=validation)
        self._week_decode_table = week_decode_table

    def run(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        # Add category and dates to the data
        df_out = self._add_category(df=df, category=category).pipe(
            self._add_dates, week_dates=self._week_decode_table
        )
        # Convert column names to lowercase
        df_out.columns = [col.lower() for col in df_out.columns]

        return df_out

    def _add_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Adds a category column to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to which the category will be added.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'category' column.
        """
        df["CATEGORY"] = category

        # Correct dtype
        df["CATEGORY"] = df["CATEGORY"].astype(DTYPES["CATEGORY"])  # type: ignore
        return df

    def _add_dates(self, df: pd.DataFrame, week_dates: pd.DataFrame) -> pd.DataFrame:
        """Adds year, start and end dates to the DataFrame based on the week number.

        Args:
            df (pd.DataFrame): The DataFrame to which dates will be added. Must contain
                a 'week' column.
            week_decode_filepath (Path): The path to the week decode CSV file.
        Returns:
            pd.DataFrame: The input DataFrame with added 'start_date' and 'end_date' columns.
        """

        df = df.merge(week_dates, on="WEEK", how="left")
        # Add year column for trend analysis
        df["YEAR"] = df["END"].dt.year
        df["YEAR"] = df["YEAR"].astype("Int64")

        return df
