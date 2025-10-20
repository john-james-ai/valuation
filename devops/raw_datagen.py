#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/raw_datagen.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 12:18:21 am                                                #
# Modified   : Monday October 20th 2025 12:52:23 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Set

from dataclasses import dataclass
from datetime import datetime
from functools import reduce

from loguru import logger
import numpy as np
import pandas as pd
import typer

from valuation.app.dataprep.task import DataPrepTaskResult
from valuation.app.state import Status
from valuation.asset.entity import Entity
from valuation.asset.identity.dataset import DatasetID
from valuation.asset.stage import DatasetStage
from valuation.asset.types import AssetType
from valuation.core.structure import DataClass
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.file.io import IOService
from valuation.infra.loggers import configure_logging
from valuation.utils.file import is_directory_empty


@dataclass
class RawSalesDataConfig(DataClass):
    """Holds data related to the current operating mode."""

    source_dataset: str = "data/prod/ingest/sales_ingest.parquet"
    category_config_filepath: str = "config.yaml"
    target_sample_size: int = 10000
    window_size: int = 52


# ------------------------------------------------------------------------------------------------ #
class RawSalesDataGenerator:
    """Generates ModeSalesDataConfig based on the current operating mode."""

    def __init__(
        self,
        config: RawSalesDataConfig,
        mode: str,
        io: IOService = IOService,
        force: bool = False,
        random_state: int = None,
    ) -> None:
        self._config = config
        self._mode = mode
        self._io = io
        self._force = force
        self._random_state = random_state
        self._file_system = DatasetFileSystem()

        self._dataset_id = None  # Ingest dataset createadd during processing

    def run(self) -> None:
        """Generates the mode sales data."""

        if self._mode == "prod":
            raise RuntimeError("Raw sales data generation is not allowed in 'prod' mode.")

        # Generate stratified sample
        stratified_sample = self.generate_stratified_sample()

        # Generate raw zipped files
        self.generate_raw_zipped_files(stratified_sample=stratified_sample)

    def generate_stratified_sample(self) -> pd.DataFrame:
        """Generates the mode sales data."""
        result = DataPrepTaskResult(task_name="GenrateStratifiedSample", dataset_name="sales")
        if self._mode == "prod":
            raise RuntimeError("Mode sales data generation is not allowed in 'prod' mode.")

        result.started = datetime.now()

        # Load raw sales data
        df = self._io.read(filepath=self._config.source_dataset)
        result.records_in = len(df)
        logger.info(f"Loaded raw sales data with {len(df)} records.")

        # Identify common weeks across all categories
        common_weeks = self._get_common_weeks(df)
        logger.info(f"Identified {len(common_weeks)} common weeks across all categories.")

        # Randomly select a sequential window of weeks
        selected_weeks = self._randomly_select_window(
            common_weeks=common_weeks, window_size=self._config.window_size
        )
        logger.info(f"Selected {len(selected_weeks)} weeks for sampling.")

        # Filter dataset to selected weeks
        filtered_df = self._filter_dataset(df, weeks=selected_weeks)
        logger.info(f"Filtered dataset to {len(filtered_df)} records for selected weeks.")

        # Determine sample proportion
        n = len(filtered_df)
        p = self._get_sample_proportion(n=n, target_sample_size=self._config.target_sample_size)

        # Perform stratified sampling
        sampled_df = self._stratified_sample(filtered_df, p=p)
        logger.info(f"Stratified sampled dataset to {len(sampled_df)} records.")
        categories = sampled_df["CATEGORY"].nunique()
        weeks = sampled_df["WEEK"].nunique()
        logger.info(f"Sampled dataset contains {categories} categories over {weeks} weeks.")

        # Update task result
        result.records_out = len(sampled_df)
        result.ended = datetime.now()
        result.elapsed = (result.ended - result.started).total_seconds()
        result.status = Status.SUCCESS.value
        # Finalize the task result
        result.end_task()
        logger.info(result)

        return sampled_df

    def generate_raw_zipped_files(self, stratified_sample: pd.DataFrame) -> pd.DataFrame:
        """Generates the mode sales data."""

        if self._mode == "prod":
            raise RuntimeError("Mode sales data generation is not allowed in 'prod' mode.")

        result = DataPrepTaskResult(task_name="Generate Zipped Files", dataset_name="sales")
        result.started = datetime.now()

        # Read the config file
        category_config = self._io.read(filepath=self._config.category_config_filepath)

        # Capture records per category
        records_per_category = []

        # Iterate through  each category and save the filtered dataset
        for _, category_info in category_config["category_filenames"].items():
            category = category_info["category"]
            name, format = category_info["filename"].split(".")
            filepath = self._file_system.get_asset_filepath(
                id_or_passport=DatasetID(
                    name=name,
                    stage=DatasetStage.RAW,
                    asset_type=AssetType.DATASET,
                    entity=Entity.SALES,
                ),
                format=str(format),
                mode=self._mode,
            )

            # Filter dataset by category and count records
            df_category = stratified_sample[stratified_sample["CATEGORY"] == category]
            if df_category.empty or df_category is None:
                logger.warning(f"No records found for category '{category}'. Skipping.")
                counts = {"category": category, "records": 0}
                records_per_category.append(counts)
                continue

            counts = {"category": category, "records": len(df_category)}
            records_per_category.append(counts)

            # Cast records_out to int
            result.records_out = result.records_out if result.records_out else 0
            result.records_out += len(df_category)
            # Save the category dataset
            self._io.write(data=df_category, filepath=filepath)
            logger.info(f"Saved raw data for category '{category}' to {filepath}.")

        result.status = Status.SUCCESS.value
        result.end_task()

        counts = pd.DataFrame(records_per_category)
        total_categories = counts["category"].nunique()
        logger.info(
            f"Saved a total of {result.records_out} raw data records for {total_categories} categories."
        )

        logger.info(result)
        logger.info(f"\n{counts}")
        return counts

    def _exists(self) -> bool:
        """Checks if the target dataset already exists.

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        location = self._file_system.asset_location
        return location.exists() and not is_directory_empty(location)

    def _randomly_select_window(self, common_weeks: Set[int], window_size: int) -> Set[int]:
        """Randomly selects a sequential set of week indices from the set of common weeks.

        Args:
            common_weeks (Set[int]): The set of week indices that exist for all categories.
            window_size (int): The size of the window to select.
        Returns:
            Set[int]: The randomly selected set of week indices.
        """
        weeks_list = sorted(common_weeks)
        max_start_index = len(weeks_list) - window_size
        start_index = np.random.randint(0, max_start_index + 1)
        selected_weeks = set(weeks_list[start_index : start_index + window_size])
        return selected_weeks

    def _get_sample_proportion(self, n: int, target_sample_size: int) -> float:
        """Calculates the sample proportion needed to achieve the target sample size.

        Args:
            n (int): The total number of available records.
            target_sample_size (int): The desired sample size.

        Returns:
            float: The sample proportion.
        """

        return target_sample_size / n

    def _get_common_weeks(self, df: pd.DataFrame) -> Set[int]:
        """Finds the set of sequential week indices that exist for ALL categories."""

        return (
            df.groupby("CATEGORY")["WEEK"]
            .apply(lambda x: set(x))
            .pipe(lambda s: reduce(set.intersection, s))
        )

    def _filter_dataset(self, df: pd.DataFrame, weeks: Set[int]) -> pd.DataFrame:
        """Filters the dataset to include only records from the specified weeks."""

        return df[df["WEEK"].isin(weeks)]

    def _stratified_sample(self, df: pd.DataFrame, p: float) -> pd.DataFrame:
        """Performs stratified sampling by CATEGORY and WEEK and returns a sampled DataFrame.

        Args:
            df (pd.DataFrame): The input dataframe to sample from.
            p (float): The fraction of samples to draw from each group.

        Returns:
            pd.DataFrame: The stratified sampled dataframe with reset index.
        """
        return (
            df.groupby(["CATEGORY", "WEEK"], group_keys=False)
            .sample(frac=p, random_state=self._random_state)
            .reset_index(drop=True)
        )


# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
@app.command()
def main(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        case_sensitive=False,
        help="Whether to force reprocessing if the file already exists.",
    ),
    mode: str = typer.Option(
        "test",
        "--mode",
        "-m",
        case_sensitive=False,
        help="Mode: valid values are 'test', 'prod', 'dev'. Defaults to 'test'.",
    ),
    random_state: int = typer.Option(
        None,
        "--random-state",
        "-r",
        help="Mode: valid values are 'test', 'prod', 'dev'. Defaults to 'test'.",
    ),
):
    """Main entry point for the Valuation package."""
    # Configure logging
    configure_logging()
    # Create the mode sales data generator
    config = RawSalesDataConfig()
    generator = RawSalesDataGenerator(
        config=config, mode=mode, force=force, random_state=random_state
    )
    generator.run()


if __name__ == "__main__":
    app()
