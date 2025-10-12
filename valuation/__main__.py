#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/__main__.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 11:01:16 pm                                               #
# Modified   : Sunday October 12th 2025 05:59:02 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""
from typing import Dict

import typer

from valuation.config.filepaths import (
    CATEGORY_DATA_FILEPATH,
    CONFIG_CATEGORY_FILEPATH,
    DATASET_PROFILE_FILEPATH,
    RAW_DATA_DIR,
    SALES_DATA_FILEPATH,
    STORE_DATA_FILEPATH,
    TEST_DATA_FILEPATH,
    TRAIN_DATA_FILEPATH,
    VALIDATION_DATA_FILEPATH,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.config.reader import ConfigReader
from valuation.pipeline.aggregate import KPIDataPrep, KPIDataPrepConfig
from valuation.pipeline.config import DataPrepCoreConfig
from valuation.pipeline.ingest import SalesDataPrep, SalesDataPrepConfig
from valuation.pipeline.profile import ProfileConfig, SalesDataProfile
from valuation.pipeline.split import DatasetSplitter, PathsConfig, SplitterConfig

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
def prepare_category_kpi_data(force: bool) -> None:
    """Prepares, cleans and aggregates the category data.
    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """

    # Create configuration for category data processing
    core_config = DataPrepCoreConfig(task_name="Category KPI Data Preparation", force=force)
    config = KPIDataPrepConfig(
        core_config=core_config,
        input_filepath=TRAIN_DATA_FILEPATH,
        output_filepath=CATEGORY_DATA_FILEPATH,
        groupby="category",
    )
    # Instantiate the KPI data processor
    processor = KPIDataPrep()

    # Run the category kpi data preparation pipeline
    processor.prepare(config=config)


# ------------------------------------------------------------------------------------------------ #
def prepare_store_kpi_data(force: bool) -> None:
    """Prepares, cleans and aggregates the store data.
    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """

    # Create configuration for category data processing
    core_config = DataPrepCoreConfig(task_name="Store KPI Data Preparation", force=force)
    config = KPIDataPrepConfig(
        core_config=core_config,
        input_filepath=TRAIN_DATA_FILEPATH,
        output_filepath=STORE_DATA_FILEPATH,
        groupby="store",
    )
    # Instantiate the KPI data processor
    processor = KPIDataPrep()

    # Run the category kpi data preparation pipeline
    processor.prepare(config=config)


# ------------------------------------------------------------------------------------------------ #
def split_sales_data(force: bool) -> None:
    """Splits the sales data into training, validation and test datasets.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    core_config = DataPrepCoreConfig(task_name="Sales Data Splitting", force=force)
    paths_config = PathsConfig(
        input_filepath=SALES_DATA_FILEPATH,
        train_filepath=TRAIN_DATA_FILEPATH,
        validation_filepath=VALIDATION_DATA_FILEPATH,
        test_filepath=TEST_DATA_FILEPATH,
    )

    split_config = SplitterConfig(
        core_config=core_config,
        paths=paths_config,
        val_col="week",
        train_size=0.7,
        val_size=0.15,
        shuffle=False,
        random_state=42,
    )
    splitter = DatasetSplitter()
    splitter.prepare(config=split_config)


# ------------------------------------------------------------------------------------------------ #
def prepare_sales_data(category_filenames: Dict, force: bool) -> None:
    """Prepares, cleans and aggregates the sales data.

    Args:
        category_filenames (Dict): A dictionary mapping category names to their respective file
            paths.
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for sales data processing
    core_config = DataPrepCoreConfig(task_name="Sales Data Preparation", force=force)
    config = SalesDataPrepConfig(
        core_config=core_config,
        week_decode_filepath=WEEK_DECODE_TABLE_FILEPATH,
        output_filepath=SALES_DATA_FILEPATH,
        category_filenames=category_filenames,
        raw_data_directory=RAW_DATA_DIR,
    )
    # Instantiate the sales data processor
    processor = SalesDataPrep()

    # Run the sales data preparation pipeline
    processor.prepare(config=config)


# ------------------------------------------------------------------------------------------------ #
def profile(category_filenames: Dict, force: bool) -> None:
    """Creates the sales data profiling dataset.
    Args:
        category_filenames (Dict): A dictionary mapping category names to their respective file paths.
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Configure the profiler
    core_config = DataPrepCoreConfig(task_name="Sales Data Profiling", force=force)
    config = ProfileConfig(
        core_config=core_config,
        output_filepath=DATASET_PROFILE_FILEPATH,
        category_filenames=category_filenames,
        raw_data_directory=RAW_DATA_DIR,
    )
    # Instantiate the processor
    processor = SalesDataProfile()
    # Run the processor pipeline
    processor.prepare(config=config)


# ------------------------------------------------------------------------------------------------ #
@app.command()
def main(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        case_sensitive=False,
        help="Whether to force reprocessing if the file already exists.",
    )
):
    """Main entry point for the Valuation package."""
    # Configure logging
    configure_logging()
    # Read category filenames from configuration
    config_reader = ConfigReader()
    category_filenames = config_reader.read(CONFIG_CATEGORY_FILEPATH)
    # Run Pipelines
    profile(category_filenames=category_filenames, force=False)
    prepare_sales_data(category_filenames=category_filenames, force=force)
    split_sales_data(force=force)
    prepare_store_kpi_data(force=force)
    prepare_category_kpi_data(force=force)


if __name__ == "__main__":
    app()
