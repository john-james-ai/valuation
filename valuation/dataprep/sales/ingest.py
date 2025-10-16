#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/sales/ingest.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Wednesday October 15th 2025 08:11:42 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import pandas as pd
from tqdm import tqdm

from valuation.config.data import DTYPES
from valuation.dataprep.base import Task, TaskConfig, Validation

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IngestSalesDataTaskConfig(TaskConfig):
    """Holds all parameters for the sales data ingestion process."""

    week_decode_table_filepath: Path
    raw_data_directory: Path
    input_engine: str = "pandas"
    output_engine: str = "dask"


# ------------------------------------------------------------------------------------------------ #
class IngestSalesDataTask(Task):
    """Ingests a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(self, config: IngestSalesDataTaskConfig) -> None:
        super().__init__(config=config)

    def _execute(self, data: Union[pd.DataFrame, Any]) -> pd.DataFrame:
        """Runs the ingestion process the dictionary containing category filenames and categories.

        Args:
            data (Dict[str,str]): Dictionary containing category filenames and categories.

        Returns:
            pd.DataFrame: The processed sales data with added category and date information.
        """

        config = data  # Rename for clarity

        self._task_report.records_in = 0  # Initialize record count to zero for accurate tracking

        sales_datasets = []

        # Obtain week decode data for date mapping
        week_dates = self._load(filepath=self._config.week_decode_table_filepath)  # type: ignore

        # Set up the progress bar
        pbar = tqdm(config["category_filenames"].items(), total=len(config["category_filenames"]))

        # Iterate through category sales files
        for _, category_info in pbar:
            filename = category_info["filename"]
            filepath = self._config.raw_data_directory / filename  # type: ignore
            category = category_info["category"]
            pbar.set_description(f"Processing category: {category} from file: {filename}")

            # Load, clean, calculate revenue, and aggregate
            processed_df = (
                self._load(filepath=filepath, kwargs=config.iokwargs.csv.read.as_dict())
                .pipe(self._add_category, category=category)
                .pipe(self._add_dates, week_dates=week_dates)
            )
            sales_datasets.append(processed_df)
            self._update_record_count(data=processed_df)

        # Concatenate all datasets
        full_dataset = pd.concat(sales_datasets, ignore_index=True)

        return full_dataset

    def _validate(self, data: pd.DataFrame) -> Validation:
        """
        Validates the output DataFrame to ensure structural integrity and record count consistency.

        Checks include:
        1. Presence of all mandatory columns.
        2. Non-zero input record count.
        3. Match between input and output record counts.

        Args:
            data: The output DataFrame generated by `_execute`.

        Returns:
            A Validation object containing the validation status and any resulting messages.
        """
        validation = Validation()
        COLUMNS = [
            "CATEGORY",
            "WEEK",
            "YEAR",
            "START",
            "END",
            "STORE",
            "UPC",
            "PRICE",
            "QTY",
            "MOVE",
            "PROFIT",
            "SALE",
            "OK",
        ]

        # Check for zero input records
        if self._task_report.records_in == 0:
            validation.add_message("No records were processed.")
        else:
            # Check presence and types of mandatory columns
            validation = self._validate_columns(
                validation=validation, data=data, required_columns=COLUMNS
            )
            # Check for consistent record count
            if self._task_report.records_in != len(data):
                validation.add_message(
                    f"Number of input records {self._task_report.records_in} does not match number of output records {len(data)}."
                )

        return validation

    def _update_record_count(self, data: pd.DataFrame) -> None:
        self._task_report.records_in += len(data)

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
