#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Friday October 10th 2025 02:04:19 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from loguru import logger
import pandas as pd
from tqdm import tqdm

from valuation.config import (
    RAW_DATA_DIR,
    SALES_DATA_FILEPATH,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.utils.io import IOService

# ------------------------------------------------------------------------------------------------ #


class SalesDataProcessor:
    """Processes raw sales data into a cleaned and aggregated dataset."""

    def __init__(self, io: IOService = IOService) -> None:
        self._io = io

    def load(self, filename: str, category: str) -> pd.DataFrame:
        """Loads a single data file and adds a category identifier.

        Args:
            filename (str): The name of the file to load from the raw data directory.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data with an added
                'category' column.
        """
        filepath = RAW_DATA_DIR / filename
        df = self._io.read(filepath=filepath)
        df["category"] = category
        return df

    def save(self, df: pd.DataFrame, filepath: str) -> None:
        """Saves a DataFrame to the processed data directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            filepath (str): The name of the file to save the DataFrame to within
                the processed data directory.
        """
        self._io.write(data=df, filepath=filepath)
        logger.success(f"Saved aggregated dataset to {filepath}")

    def clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies a series of cleaning rules to the raw sales data.

        The cleaning process includes:
        1.  Filtering for valid records based on the 'OK' flag.
        2.  Removing records with invalid or missing prices and movement.
        3.  Ensuring bundle quantity ('QTY') is a valid number.
        4.  Standardizing data types and column names.

        Args:
            df (pd.DataFrame): The raw sales data DataFrame.

        Returns:
            pd.DataFrame: A cleaned version of the input DataFrame.
        """
        df_clean = df.copy()

        query_string = """
            OK == 1 and \
            PRICE > 0 and \
            MOVE > 0 and \
            QTY >= 1
        """
        # 1. Remove records with missing critical values
        df_clean = df_clean.query(query_string).copy()

        # 2. Convert to consistent numeric types
        numeric_cols = ["QTY", "PRICE", "PROFIT"]
        for col in numeric_cols:
            df_clean[col] = df_clean[col].astype("float64")

        # 3. Rename columns for clarity and drop unneeded ones.
        df_clean = (
            df_clean.rename(columns={"PROFIT": "GROSS_MARGIN_PCT"}).drop(columns=["SALE"])
        ).copy()

        # 4. Standardize column names to lowercase
        df_clean.columns = df_clean.columns.str.lower()

        return df_clean

    def add_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds week start and end dates to the aggregated dataset.

        This method assumes that the 'week' column contains week numbers (1-52)
        and adds 'week_start_date' and 'week_end_date' columns based on a fixed year.

        Args:
            df (pd.DataFrame): The aggregated DataFrame with a 'week' column.
        Returns:
            pd.DataFrame: The input DataFrame with added 'week_start_date' and
                'week_end_date' columns.
        """
        # Read the week decode table and count number of dates
        week_dates = self._io.read(filepath=WEEK_DECODE_TABLE_FILEPATH)

        # Merge start and end dates into the original DataFrame
        df = df.merge(week_dates, on="week", how="left")

        return df

    def calculate_revenue(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates transaction-level revenue, accounting for product bundles.

        Revenue is derived from bundle price, individual units sold, and bundle size.
        The formula used is: revenue = (price * move) / qty.

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain lowercase columns
                'price', 'move', and 'qty'.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'revenue' column.
        """
        df["revenue"] = (df["price"] * df["move"]) / df["qty"]
        return df

    def calculate_gross_profit(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates transaction-level gross profit.

        Gross profit is derived from revenue and gross margin percentage.
        The formula used is: gross_profit = revenue * (gross_margin_pct / 100).

        Args:
            df (pd.DataFrame): The cleaned sales data. Must contain lowercase columns
                'revenue' and 'gross_margin_pct'.
        Returns:
            pd.DataFrame: The input DataFrame with an added 'gross_profit' column.
        """
        df["gross_profit"] = df["revenue"] * (df["gross_margin_pct"] / 100.0)
        return df

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
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
        # 1: Group by store, category, and week, summing revenue and gross profit
        aggregated = df.groupby(["store", "category", "week"], as_index=False).agg(
            {"revenue": "sum", "gross_profit": "sum"}
        )

        # Step 2: Calculate the true margin from the summed totals
        # We add a small epsilon to avoid division by zero if revenue is 0
        aggregated["gross_margin_pct"] = aggregated["gross_profit"] / (
            aggregated["revenue"] + 1e-6
        )

        # Sort for reproducibility
        aggregated = aggregated.sort_values(["store", "category", "week"])

        return aggregated

    def process(self, category_filenames: dict, force: bool = False) -> None:
        """Processes raw sales data files into a cleaned and aggregated dataset.

        This method orchestrates the loading, cleaning, revenue calculation,
        gross profit calculation, aggregation, and saving of sales data.

        Args:
            category_filenames (dict): A dictionary mapping category names to
                their corresponding filenames.
            force (bool): If True, forces reprocessing even if the output file exists. Defaults to
                False.
        Returns:
            None
        """
        sales_datasets = []

        # Check if output file already exists and not forcing reprocessing
        if not force and SALES_DATA_FILEPATH.exists():
            logger.info(
                f"Processed dataset already exists at {SALES_DATA_FILEPATH}. \
                    Skipping processing.\nTo reprocess, set force=True."
            )
            return

        # If force is True and file exists, log that we are reprocessing and
        # remove the existing file
        if force and SALES_DATA_FILEPATH.exists():
            logger.info(
                f"Force reprocessing enabled. \
                    Existing file at {SALES_DATA_FILEPATH} will be overwritten."
            )
            SALES_DATA_FILEPATH.unlink()  # Remove the existing file

        logger.info("Processing dataset...")

        # Set up the progress bar
        pbar = tqdm(category_filenames.items(), total=len(category_filenames))

        # Iterate through category sales files
        for _, category_info in pbar:
            filename = category_info["filename"]
            category = category_info["category"]
            pbar.set_description(f"Processing category: {category} from file: {filename}")

            # Load, clean, calculate revenue, and aggregate
            processed_df = (
                self.load(filename=filename, category=category)
                .pipe(self.clean_dataset)
                .pipe(self.calculate_revenue)
                .pipe(self.calculate_gross_profit)
                .pipe(self.aggregate)
            )
            sales_datasets.append(processed_df)

        # Concatenate all datasets
        full_dataset = pd.concat(sales_datasets, ignore_index=True)
        logger.info(f"Concatenated dataset has {len(full_dataset)} records.")

        # Save processed dataset
        self.save(df=full_dataset, filepath=SALES_DATA_FILEPATH)
