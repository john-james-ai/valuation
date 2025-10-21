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
# Modified   : Tuesday October 21st 2025 06:48:34 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Union

from loguru import logger
import pandas as pd

from valuation.flow.dataprep.task import (
    DataPrepTaskResult,
    SISODataPrepTask,
    SISODataPrepTaskConfig,
)
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
class CleanSalesDataTask(SISODataPrepTask):
    """Cleans a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(
        self,
        config: SISODataPrepTaskConfig,
        dataset_store: DatasetStore,
    ) -> None:
        super().__init__(config=config, dataset_store=dataset_store)

    def _execute(self, data: Union[pd.DataFrame, Any], **kwargs) -> Union[pd.DataFrame, Any]:

        logger.debug("Cleaning sales data.")

        return (
            data.pipe(self._remove_invalid_records)
            .pipe(self._normalize_columns)
            .pipe(self._calculate_revenue)
            .pipe(self._calculate_gross_profit)
        )

    def _remove_invalid_records(self, df: pd.DataFrame) -> pd.DataFrame:
        """Removes records that do not meet the following criteria:
        1.  'OK' flag is '1'
        2.  'PRICE' is greater than 0
        3.  'MOVE' is greater than 0
        4.  'QTY' is greater than 0

        Args:
            df (pd.DataFrame): The ingested sales data.

        Returns:
            pd.DataFrame: The cleaned sales data.
        """
        logger.debug("Removing invalid records based on business rules.")
        # Define the query string for filtering
        query_string = """
            OK == 1 and \
            PRICE > 0 and \
            MOVE > 0 and \
            QTY >= 1
        """

        df_clean = df.query(
            query_string, engine="python"
        ).copy()  # Python engine required for compatability betweeen numexpr and pandas
        return df_clean

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes column names and drops unneeded columns.

        Args:
            df (pd.DataFrame): The cleaned sales data.
        Returns:
            pd.DataFrame: The cleaned sales data with standardized column names.
        """
        logger.debug("Normalizing column names and dropping unneeded columns.")
        df_clean = df.copy()
        # Rename columns for clarity and drop unneeded ones.
        df_clean = (
            df_clean.rename(columns={"PROFIT": "GROSS_MARGIN_PCT"}).drop(columns=["SALE"])
        ).copy()
        # Standardize column names to lowercase
        df_clean.columns = df_clean.columns.str.lower()
        return df_clean

    def _calculate_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates transaction-level revenue, accounting for product bundles.

        Revenue is derived from bundle price, individual units sold, and bundle size.
        The formula used is: revenue = (price * move) / qty.

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain lowercase columns
                'price', 'move', and 'qty'.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'revenue' column.
        """
        logger.debug("Calculating transaction-level revenue.")
        df["revenue"] = (df["price"] * df["move"]) / df["qty"]
        return df

    def _calculate_gross_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates transaction-level gross profit.

        Gross profit is derived from revenue and gross margin percentage.
        The formula used is: gross_profit = revenue * (gross_margin_pct / 100).

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain lowercase columns
                'revenue' and 'gross_margin_pct'.
        Returns:
            pd.DataFrame: The input DataFrame with an added 'gross_profit' column.
        """
        logger.debug("Calculating transaction-level gross profit.")
        df["gross_profit"] = df["revenue"] * (df["gross_margin_pct"] / 100.0)
        return df

    def _validate_result(self, result: DataPrepTaskResult) -> DataPrepTaskResult:
        """Validates the output DataFrame structure and integrity.

        Args:
            result (TaskResult): The result object containing the output DataFrame.
        Returns:
            TaskResult: The validated result object.
        """
        COLUMNS = [
            "store",
            "category",
            "week",
            "year",
            "start",
            "end",
            "revenue",
            "gross_profit",
            "gross_margin_pct",
        ]

        data = result.dataset.data
        validation = result.validation

        # 1. Check for mandatory columns
        logger.debug("Validating output DataFrame structure and integrity.")
        # Critical: Must check this first and return if failed, as subsequent steps require these columns.
        for col in COLUMNS:
            if col not in data.columns:
                validation.add_message(f"Missing mandatory column: {col}")

        # Check for negative gross profit
        logger.debug("..checking negative gross profit.")
        negative_profit = data[data["gross_profit"] < 0]
        if not negative_profit.empty:
            reason = f"Aggregated 'gross_profit' contains {len(negative_profit)} negative values."
            validation.add_failed_records(reason=reason, records=negative_profit)

        # Check for division by zero / resulting NaNs/Infs in the margin
        logger.debug("..checking for NaN or infinite gross margin percentages.")
        null_margins = data[data["gross_margin_pct"].isnull()]
        if not null_margins.empty:
            reason = f"'gross_margin_pct' contains {len(null_margins)} NULLs, often indicating a divide by zero error."
            validation.add_failed_records(reason=reason, records=null_margins)

        return result
