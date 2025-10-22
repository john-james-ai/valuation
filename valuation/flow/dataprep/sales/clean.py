#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/clean.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Wednesday October 22nd 2025 02:30:19 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Optional

from dataclasses import dataclass

from loguru import logger
import pandas as pd

from valuation.flow.dataprep.task import (
    DataPrepTaskResult,
    SISODataPrepTask,
    SISODataPrepTaskConfig,
)
from valuation.flow.validation import Validation

# ------------------------------------------------------------------------------------------------ #
REQUIRED_COLUMNS_CLEAN = {
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

NON_NEGATIVE_COLUMNS_CLEAN = ["revenue"]


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CleanSalesDataTaskResult(DataPrepTaskResult):
    """Holds the results of the CleanSalesDataTask execution."""


# ------------------------------------------------------------------------------------------------ #
class CleanSalesDataTask(SISODataPrepTask):
    """Cleans a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        config (SISODataPrepTaskConfig): Task configuration.
        validation (Optional[Validation]): Optional Validation instance for data checks.
    """

    _result: type[CleanSalesDataTaskResult]

    def __init__(
        self,
        config: SISODataPrepTaskConfig,
        result: type[CleanSalesDataTaskResult] = CleanSalesDataTaskResult,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(config=config, validation=validation, result=result)

    def _execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Execute the cleaning pipeline on the provided DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be cleaned.

        Returns:
            pd.DataFrame: Cleaned DataFrame after all transformations.
        """
        logger.debug("Cleaning sales data.")

        return (
            df.pipe(self._remove_invalid_records)
            .pipe(self._normalize_columns)
            .pipe(self._calculate_revenue)
            .pipe(self._calculate_gross_profit)
        )

    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes records that do not meet business criteria.

        The rules applied are:
            1. 'ok' flag is '1'
            2. 'price' > 0
            3. 'move' > 0
            4. 'qty' >= 1

        Args:
            df (pd.DataFrame): The ingested sales data.

        Returns:
            pd.DataFrame: The cleaned sales data.
        """
        logger.debug("Removing invalid records based on business rules.")
        # Define the query string for filtering
        query_string = """
            ok == 1 and \
            price > 0 and \
            move > 0 and \
            qty >= 1
        """

        df_clean = df.query(
            query_string, engine="python"
        ).copy()  # Python engine required for compatability betweeen numexpr and pandas
        return df_clean

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and drop unneeded columns.

        Args:
            df (pd.DataFrame): The cleaned sales data.

        Returns:
            pd.DataFrame: The cleaned sales data with standardized column names.
        """
        logger.debug("Normalizing column names and dropping unneeded columns.")
        df_clean = df.copy()
        # Rename columns for clarity and drop unneeded ones.
        df_clean = (
            df_clean.rename(columns={"profit": "gross_margin_pct"}).drop(columns=["sale"])
        ).copy()
        # Standardize column names to lowercase
        df_clean.columns = df_clean.columns.str.lower()
        return df_clean

    def _calculate_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction-level revenue, accounting for product bundles.

        Revenue is derived from bundle price, individual units sold, and bundle size.
        Formula: revenue = (price * move) / qty.

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain columns 'price', 'move', and 'qty'.

        Returns:
            pd.DataFrame: DataFrame with an added 'revenue' column.
        """
        logger.debug("Calculating transaction-level revenue.")
        df["revenue"] = (df["price"] * df["move"]) / df["qty"]
        return df

    def _calculate_gross_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate transaction-level gross profit.

        Gross profit is derived from revenue and gross margin percentage.
        Formula: gross_profit = revenue * (gross_margin_pct / 100).

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain columns 'revenue' and 'gross_margin_pct'.

        Returns:
            pd.DataFrame: DataFrame with an added 'gross_profit' column.
        """
        logger.debug("Calculating transaction-level gross profit.")
        df["gross_profit"] = df["revenue"] * (df["gross_margin_pct"] / 100.0)
        return df
