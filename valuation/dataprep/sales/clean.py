#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/sales/clean.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Wednesday October 15th 2025 02:21:31 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from typing import Any, Optional, Union

from loguru import logger
import pandas as pd

from valuation.dataprep.base import Task, TaskConfig, Validation


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CleanSalesDataTaskConfig(TaskConfig):
    """Holds all parameters for the sales data cleaning process."""

    return_dask: bool = True

    pass


# ------------------------------------------------------------------------------------------------ #
class CleanSalesDataTask(Task):
    """Cleans a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(self, config: CleanSalesDataTaskConfig) -> None:
        super().__init__(config=config)

    def _execute(self, data: Optional[pd.DataFrame] = None) -> Union[pd.DataFrame, Any]:
        """Runs the ingestion process on the provided DataFrame.
        Args:
            data (pd.DataFrame): The raw sales data DataFrame.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The processed sales data with added category and date information.

        """
        logger.debug("Cleaning sales data.")

        data = (
            data
            if data is not None
            else self._load(
                filepath=self.config.input_location, return_dask=self.config.return_dask  # type: ignore
            )
        )
        return (
            data.pipe(self._remove_invalid_records)
            .pipe(self._normalize_columns)
            .pipe(self._calculate_revenue)
            .pipe(self._calculate_gross_profit)
            .pipe(self._aggregate)
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

    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregates transaction-level data to the store-category-week level.

        This method collapses revenue, gross profit, and gross margin percent
        to the store, category, and week level.

        Args:
            df (pd.DataFrame): DataFrame with transaction-level data, including
                'store', 'category', 'week', and 'revenue' columns.

        Returns:
            pd.DataFrame: An aggregated DataFrame with total revenue, gross_profit, and
                gross margin percent for each store, category, and week.
        """
        logger.debug("Aggregating to store-category-week level.")
        # 1: Group by store, category, and week, summing revenue and gross profit
        aggregated = (
            df.groupby(["store", "category", "week"])
            .agg(
                revenue=("revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
                year=("year", "first"),
                start=("start", "first"),
                end=("end", "first"),
            )
            .reset_index()
        )
        # Step 2: Calculate the true margin from the summed totals
        # We add a small epsilon to avoid division by zero if revenue is 0
        aggregated["gross_margin_pct"] = (
            aggregated["gross_profit"] / (aggregated["revenue"] + 1e-6) * 100
        )

        # Sort for reproducibility
        aggregated = aggregated.sort_values(["store", "category", "week"])

        return aggregated

    def _validate(self, data: pd.DataFrame) -> Validation:
        """
        Validates the output DataFrame for structural integrity and data quality after aggregation.

        This function strictly checks the output against the expected contract,
        ignoring external state like input record counts.

        Checks include:
        1. Presence of all mandatory columns (exits early if missing to prevent KeyError).
        2. Uniqueness on the aggregation key, ensuring aggregation occurred successfully.
        3. Sanity checks on aggregated financial metrics.

        Args:
            data: The output DataFrame generated by _execute.

        Returns:
            A Validation object containing the validation status, messages, and any failed records.
        """
        validation = Validation()
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

        # 1. Check for mandatory columns
        logger.debug("Validating output DataFrame structure and integrity.")
        # Critical: Must check this first and return if failed, as subsequent steps require these columns.
        for col in COLUMNS:
            if col not in data.columns:
                validation.add_message(f"Missing mandatory column: {col}")
                return validation

        # 2. Check for uniqueness on key (Aggregation guarantee)
        logger.debug("Checking for uniqueness on aggregation key.")
        group_cols = ["week", "year", "store", "category"]
        duplicates = data[data.duplicated(subset=group_cols, keep=False)]
        if not duplicates.empty:
            reason = "Output data contains duplicate rows on the aggregation key, indicating incomplete aggregation."
            validation.add_failed_records(reason=reason, records=duplicates)

        # 3. Check integrity of aggregated financial metrics
        logger.debug("Performing sanity checks on aggregated financial metrics.")
        # Check for negative revenue values
        logger.debug("..checking negative revenue.")
        negative_revenue = data[data["revenue"] < 0]
        if not negative_revenue.empty:
            reason = f"Aggregated 'revenue' contains {len(negative_revenue)} negative values."
            validation.add_failed_records(reason=reason, records=negative_revenue)

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

        return validation
