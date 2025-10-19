#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/app/dataprep/sales/ingest.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Sunday October 19th 2025 07:05:53 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Dict, Optional, Union

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm

from valuation.app.dataprep.task import DataPrepTask, DataPrepTaskResult, TaskConfig, TaskResult
from valuation.app.state import Status
from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetID, DatasetPassport
from valuation.infra.file.io import IOService
from valuation.infra.store.dataset import DatasetStore

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"
WEEK_DECODE_TABLE_FILEPATH = "data/external/week_decode_table.csv"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IngestSalesDataTaskConfig(TaskConfig):
    """Holds all parameters for the sales data ingestion process."""

    source: str  # Path to categories and filenames mapping
    target: DatasetPassport  # Target ingested dataset passport
    week_decode_table_filepath: Path
    raw_data_directory: Path


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
        config: IngestSalesDataTaskConfig,
        dataset_store: type[DatasetStore] = DatasetStore,
        io: IOService = IOService,
    ) -> None:
        self._config = config
        self._dataset_store = dataset_store()
        self._io = io

    def _execute(self, df: pd.DataFrame, category: str, week_dates: pd.DataFrame) -> pd.DataFrame:
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
        return self._add_category(df=df, category=category).pipe(
            self._add_dates, week_dates=week_dates
        )

    def run(self, dataset: Optional[Dataset] = None, force: bool = False) -> Optional[Dataset]:
        """Executes the sales data ingestion task.

        Args:
            dataset (Optional[Dataset], optional): Not used. Defaults to None.
            force (bool, optional): If True, forces re-ingestion even if the target dataset exists.
                Defaults to False.
        Returns:
            Optional[Dataset]: The ingested sales dataset, or None if ingestion was skipped.
        """

        # Initialize the result object and start the task
        result = DataPrepTaskResult(task_name=self.task_name, config=self._config)
        result.start_task()

        # Check if output dataset alread exists
        dataset_id_out = DatasetID.from_passport(self._config.target)
        if self._dataset_store.exists(dataset_id=dataset_id_out) and not force:
            dataset_out = self._dataset_store.get(dataset_id=dataset_id_out)
            result.status = Status.EXISTS.value
            result.end_task()
            logger.info(result)
            return dataset_out

        sales_datasets = []
        try:

            # Read week decoding table
            week_dates = self._load(filepath=self._config.week_decode_table_filepath)

            # Read category filenames mapping
            category_filenames = self._io.read(filepath=self._config.source)[
                CONFIG_CATEGORY_INFO_KEY
            ]

            # Create tqdm progress bar for categories
            pbar = tqdm(
                category_filenames.items(),
                total=len(category_filenames),
                desc="Ingesting Sales Data by Category",
                unit="category",
            )

            # Iterate through category sales files
            for _, category_info in pbar:
                filename = category_info["filename"]
                filepath = self._config.raw_data_directory / filename  # type: ignore
                category = category_info["category"]
                pbar.set_description(f"Processing category: {category} from file: {filename}")

                # Create a temporary dataset for loading the data
                df_in = self._load(filepath=filepath)
                if isinstance(df_in, pd.DataFrame):
                    result.records_in += len(df_in)  # type: ignore

                # Execute ingestion steps
                category_df = self._execute(df=df_in, category=category, week_dates=week_dates)

                sales_datasets.append(category_df)

            # Concatenate all datasets
            concat_df = pd.concat(sales_datasets, ignore_index=True)

            # Update result with output record counts
            result.records_out = len(concat_df)

            # Create the output dataset
            dataset_out = Dataset(passport=self._config.target, df=concat_df)

            # Add the dataset to the result object and validate
            result.dataset = dataset_out
            result = self._validate_result(result=result)

            # Save the dataset if validation passed
            if result.validation.is_valid:  # type: ignore
                self._dataset_store.add(dataset=dataset_out, overwrite=True)
                logger.debug(f"Saved ingested dataset {dataset_out.passport.label} to the store.")
            else:
                self._handle_validation_failure(validation=result.validation)  # type: ignore

        finally:
            result.end_task()
            logger.info(result)
            return result.dataset  # type: ignore

    def _validate_result(self, result: DataPrepTaskResult) -> TaskResult:
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
            result.validation = self._validate_columns(
                validation=result.validation, data=result.dataset.data, required_columns=COLUMNS
            )
            # Check for consistent record count
            if result.records_in != result.records_out:
                result.validation.add_message(
                    f"Number of input records {result.records_in} does not match number of output records {result.records_out}."
                )

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
