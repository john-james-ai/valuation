#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/mode_data.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 12:18:21 am                                                #
# Modified   : Sunday October 19th 2025 03:20:54 am                                                #
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

from valuation.app.state import Status
from valuation.app.task import TaskResult
from valuation.asset.stage import DatasetStage
from valuation.asset.types import AssetType
from valuation.core.structure import DataClass
from valuation.infra.file.file_system import FileSystem
from valuation.infra.file.io import IOService
from valuation.infra.loggers import configure_logging


@dataclass
class ModeSalesDataConfig(DataClass):
    """Holds data related to the current operating mode."""

    source_dataset: str = "data/prod/ingest/sales_ingest.parquet"
    target_sample_size: int = 1000
    window_size: int = 52


# ------------------------------------------------------------------------------------------------ #
class ModeSalesDataGenerator:
    """Generates ModeSalesDataConfig based on the current operating mode."""

    def __init__(self, config: ModeSalesDataConfig, io: IOService = IOService) -> None:
        self._config = config
        self._io = io
        self._file_system = FileSystem(asset_type=AssetType.DATASET)
        self._result = TaskResult(task_name="ModeSalesDataGenerator", dataset_name="sales")

    def run(self, mode: str) -> TaskResult:
        """Generates the mode sales data."""

        self._result.started = datetime.now()

        # Load raw sales data
        df = self._io.read(filepath=self._config.source_dataset)
        self._result.records_in = len(df)

        # Identify common weeks across all categories
        common_weeks = self._get_common_weeks(df)

        # Randomly select a sequential window of weeks
        selected_weeks = self._randomly_select_window(
            common_weeks=common_weeks, window_size=self._config.window_size
        )

        # Filter dataset to selected weeks
        filtered_df = self._filter_dataset(df, weeks=selected_weeks)

        # Determine sample proportion
        n = len(filtered_df)
        p = self._get_sample_proportion(n=n, target_sample_size=self._config.target_sample_size)

        # Perform stratified sampling
        sampled_df = self._stratified_sample(filtered_df, p=p)

        # Save the sampled dataset
        output_location = self._file_system.get_asset_filepath(
            passport_or_stage=DatasetStage.INGEST, name="sales", format="parquet", mode=mode
        )

        self._io.write(data=sampled_df, filepath=output_location)

        # Update task result
        self._result.records_out = len(sampled_df)
        self._result.ended = datetime.now()
        self._result.elapsed = (self._result.ended - self._result.started).total_seconds()
        self._result.status = Status.SUCCESS.value

        return self._result

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
            df.groupby("category")["week"]
            .apply(lambda x: set(x))
            .pipe(lambda s: reduce(set.intersection, s))
        )

    def _filter_dataset(self, df: pd.DataFrame, weeks: Set[int]) -> pd.DataFrame:
        """Filters the dataset to include only records from the specified weeks."""

        return df[df["week"].isin(weeks)]

    def _stratified_sample(self, df: pd.DataFrame, p: float) -> pd.DataFrame:

        return df.groupby(["category", "week"], group_keys=False).apply(lambda x: x.sample(frac=p))


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
):
    """Main entry point for the Valuation package."""
    # Configure logging
    configure_logging()
    # Create the mode sales data generator
    config = ModeSalesDataConfig()
    generator = ModeSalesDataGenerator(config=config)

    # Generate the mode sales data
    result = generator.run(mode=mode)

    # Log the result
    logger.info(result)


if __name__ == "__main__":
    app()
