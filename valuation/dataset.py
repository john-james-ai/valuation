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
# Modified   : Thursday October 9th 2025 02:49:52 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from loguru import logger
import pandas as pd
from tqdm import tqdm
import typer

from valuation.config import (
    CONFIG_CATEGORY_FILEPATH,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    ConfigReader,
)
from valuation.io import IOService

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


def load(filename: str, category: str) -> pd.DataFrame:
    """Loads a single data file and adds a category identifier.

    Args:
        filename (str): The name of the file to load from the raw data directory.
        category (str): The category name to assign to all records in the DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with an added
            'category' column.
    """
    filepath = RAW_DATA_DIR / filename
    df = IOService.read(filepath=filepath)
    df["category"] = category
    return df


def save(df: pd.DataFrame, filename: str) -> None:
    """Saves a DataFrame to the processed data directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): The target filename for the output CSV file.
    """
    filepath = PROCESSED_DATA_DIR / filename
    IOService.write(data=df, filepath=filepath)
    logger.success(f"Saved aggregated dataset to {filepath}")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
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

    # 3. Standardize column names to lowercase
    df_clean.columns = df_clean.columns.str.lower()

    return df_clean


def calculate_revenue(df: pd.DataFrame) -> pd.DataFrame:
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


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates transaction-level data to the store-category-week level.

    This function collapses all individual product transactions within a category
    to calculate the total weekly revenue for that category in each store.

    Args:
        df (pd.DataFrame): DataFrame with transaction-level data, including
            'store', 'category', 'week', and 'revenue' columns.

    Returns:
        pd.DataFrame: An aggregated DataFrame with total revenue for each
            store, category, and week.
    """
    aggregated = df.groupby(["store", "category", "week"], as_index=False).agg({"revenue": "sum"})

    # Rename columns for clarity
    aggregated.columns = ["store_id", "category", "week", "revenue"]

    # Sort for reproducibility
    aggregated = aggregated.sort_values(["store_id", "category", "week"])

    return aggregated


@app.command()
def main():
    """Processes raw sales data into a cleaned and aggregated dataset."""
    sales_datasets = []
    # Obtain categories and filenames from config
    config_reader = ConfigReader()
    category_filenames = config_reader.read(CONFIG_CATEGORY_FILEPATH)

    # ----------------------------------------------
    # Iterate through category sales files
    logger.info("Processing dataset...")
    for _, category_info in tqdm(category_filenames.items(), total=len(category_filenames)):
        filename = category_info["filename"]
        category = category_info["category"]
        logger.info(f"Processing category: {category} from file: {filename}")

        # Load, clean, calculate revenue, and aggregate
        processed_df = (
            load(filename=filename, category=category)
            .pipe(clean_dataset)
            .pipe(calculate_revenue)
            .pipe(aggregate)
        )
        sales_datasets.append(processed_df)

    # Concatenate all datasets
    full_dataset = pd.concat(sales_datasets, ignore_index=True)
    logger.info(f"Concatenated dataset has {len(full_dataset)} records.")

    # Save processed dataset
    save(df=full_dataset, filename="sales_data.csv")


if __name__ == "__main__":
    app()
