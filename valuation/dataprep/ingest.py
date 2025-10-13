#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/ingest.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Monday October 13th 2025 06:22:50 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from valuation.dataprep.base import Task, TaskConfig

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IngestTaskConfig(TaskConfig):
    """Holds all parameters for the sales data ingestion process."""

    week_decode_table_filepath: Path
    raw_data_directory: Path


# ------------------------------------------------------------------------------------------------ #
class IngestTask(Task):
    """Ingests a raw sales data file.

    The ingestion adds category and date information to the raw sales data.

    Args:
        weeks (pd.DataFrame): The week decode table containing start and end dates for each week
            number.

    """

    def __init__(self, config: IngestTaskConfig) -> None:
        super().__init__(config=config)

    def _execute(self) -> pd.DataFrame:
        """Runs the ingestion process on the provided DataFrame.
        Args:
            data (pd.DataFrame): The raw sales data DataFrame.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The processed sales data with added category and date information.

        """

        sales_datasets = []

        # Obtain week decode data for date mapping
        week_dates = self._load(filepath=self._config.week_decode_filepath)  # type: ignore

        # Obtain the list of category files
        config = self._load(filepath=self._config.input_location)  # type: ignore

        # Set up the progress bar
        pbar = tqdm(config.category_filenames.items(), total=len(config.category_filenames))

        # Iterate through category sales files
        for _, category_info in pbar:
            filename = category_info["filename"]
            filepath = config.raw_data_directory / filename
            category = category_info["category"]
            pbar.set_description(f"Processing category: {category} from file: {filename}")

            # Load, clean, calculate revenue, and aggregate
            processed_df = (
                self._load(filepath=filepath)
                .pipe(self.add_category, category=category)
                .pipe(self.add_dates, week_dates=week_dates)
            )
            sales_datasets.append(processed_df)

        # Concatenate all datasets
        full_dataset = pd.concat(sales_datasets, ignore_index=True)

        # Save processed dataset
        self._save(df=full_dataset, filepath=config.output_location)

        return full_dataset

    def validate(self, data: pd.DataFrame) -> bool:
        super()._validate(data=data)
        return (
            data["CATEGORY"].notnull().all()
            and data["YEAR"].notnull().all()
            and data["START"].notnull().all()
            and data["END"].notnull().all()
        )

    def add_category(self, df: pd.DataFrame, category: str) -> pd.DataFrame:
        """Adds a category column to the DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to which the category will be added.
            category (str): The category name to assign to all records in the DataFrame.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'category' column.
        """
        df["CATEGORY"] = category
        return df

    def add_dates(self, df: pd.DataFrame, week_dates: pd.DataFrame) -> pd.DataFrame:
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
