#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/workflow/dataprep/sales/ingest.py                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Friday October 17th 2025 04:58:56 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Dict, Union

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm

from valuation.config.data import DTYPES
from valuation.utils.io.config import IOKwargs
from valuation.utils.io.service import IOService
from valuation.workflow.task import Task, TaskConfig, TaskContext, TaskResult

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IngestSalesDataTaskConfig(TaskConfig):
    """Holds all parameters for the sales data ingestion process."""

    week_decode_table_filepath: Path
    raw_data_directory: Path


# ------------------------------------------------------------------------------------------------ #
class IngestSalesDataTask(Task):
    """Ingests a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(self, config: IngestSalesDataTaskConfig, io: type[IOService] = IOService) -> None:
        super().__init__(config=config)
        self._task_context = TaskContext(config=config)
        self._io = io

    def _execute(
        self, data: Union[pd.DataFrame, Any], category: str, week_dates: pd.DataFrame
    ) -> pd.DataFrame:
        """Runs the ingestion process on the provided DataFrame.

        Args:
            data (pd.DataFrame): The raw sales data DataFrame.
            category (str): The category name to assign to all records in the DataFrame.
            week_dates (pd.DataFrame): The week decode table containing start and end dates for each
                week number.

            Returns:
            pd.DataFrame: The processed sales data with added category and date information.
        """
        # Add category and dates to the data
        return self._add_category(df=data, category=category).pipe(
            self._add_dates, week_dates=week_dates
        )

    def run(self, data: Union[pd.DataFrame, Dict[str, Dict[str, str]]]) -> TaskResult:
        """Executes the full task lifecycle: execution, validation, and reporting.

        This method orchestrates the task's operation within a context that
        handles timing, status updates, and error logging. It ensures that a
        complete TaskResult object is returned, whether the task succeeds or fails.

        Args:
            data: The input data to be processed by the task.

        Returns:
            TaskResult: An object containing the final status, metrics,
                validation info, and output data of the task run.

        Raises:
            RuntimeError: If input data is missing or empty, or if the
                validation fails.
        """

        try:
            with self._task_context as result:
                sales_datasets = []

                category_config = data

                # Check input data and set records_in count.
                if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                    raise RuntimeError(f"{self.__class__.__name__} - No input data provided.")

                # Obtain the week decode table for date mapping
                week_dates = self._load(filepath=self._config.week_decode_table_filepath)  # type: ignore

                # Get category filenames and categories from config

                if CONFIG_CATEGORY_INFO_KEY not in category_config:
                    raise RuntimeError(
                        f"{self.__class__.__name__} - Missing '{CONFIG_CATEGORY_INFO_KEY}' in config."
                    )

                # Set up the progress bar to iterate through categories
                pbar = tqdm(
                    category_config[CONFIG_CATEGORY_INFO_KEY].items(),
                    total=len(category_config[CONFIG_CATEGORY_INFO_KEY]),
                )

                logger.debug("Beginning ingestion of sales data by category.")

                # Iterate through category sales files
                for _, category_info in pbar:
                    filename = category_info["filename"]
                    filepath = self._config.raw_data_directory / filename  # type: ignore
                    category = category_info["category"]
                    pbar.set_description(f"Processing category: {category} from file: {filename}")

                    # Load the file and process ingestion
                    data = self._load(
                        filepath=filepath, kwargs=IOKwargs().pandas_csv.read.as_dict()
                    )
                    result.records_in += len(data)

                    # Execute ingestion steps
                    category_data = self._execute(
                        data=data, category=category, week_dates=week_dates
                    )

                    sales_datasets.append(category_data)

                # Concatenate all datasets
                result.data = pd.concat(sales_datasets, ignore_index=True)
                result.records_out = len(result.data)

                # Validate the result by calling the subclass's implementation.
                result = self._validate_result(result=result)

                # Handle validation failure.
                if not result.validation.is_valid:  # type: ignore
                    self._handle_validation_failure(validation=result.validation)
        finally:
            return result

    def _validate_result(self, result: TaskResult) -> TaskResult:
        """Validates the result of the ingestion process.

        Args:
            result (TaskResult): The result object containing data to validate.

        Returns:
            TaskResult: The updated result object with validation info.
        """

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
        if result.records_in == 0:
            result.validation.add_message("No records were processed.")  # type: ignore
        else:
            # Check presence and types of mandatory columns
            validation = self._validate_columns(
                validation=result.validation, data=result.data, required_columns=COLUMNS
            )
            # Check for consistent record count
            if result.records_in != result.records_out:
                validation.add_message(
                    f"Number of input records {result.records_in} does not match number of output records {result.records_out}."
                )
            result.validation = validation

        return result

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

    def _load(self, filepath: Path, **kwargs) -> Union[pd.DataFrame, Dict[str, str]]:
        """Loads a DataFrame from the specified filepath using the I/O service.

        Args:
            filepath: The path to the file to be loaded.
            **kwargs: Additional keyword arguments for the I/O service.

            Returns:
            Union[pd.DataFrame, Any]: The loaded DataFrame or data object."""

        logger.debug(f"Loading data from {filepath}")

        data = self._io.read(filepath=filepath, **kwargs)
        # Ensure correct data types
        if isinstance(data, pd.DataFrame):
            logger.debug(f"Applying data types to loaded DataFrame")
            data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
        else:
            logger.debug(
                f"Loaded data is type {type(data)} and not a DataFrame. Skipping dtype application."
            )
        return data
