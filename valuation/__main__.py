#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/__main__.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 11:01:16 pm                                               #
# Modified   : Friday October 10th 2025 11:48:38 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""
from typing import Dict

import typer

from valuation.config.data_prep import DataPrepCoreConfig
from valuation.config.filepaths import (
    CONFIG_CATEGORY_FILEPATH,
    DATASET_PROFILE_FILEPATH,
    RAW_DATA_DIR,
    SALES_DATA_FILEPATH,
    TEST_DATA_FILEPATH,
    TRAIN_DATA_FILEPATH,
    VALIDATION_DATA_FILEPATH,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.config.reader import ConfigReader
from valuation.dataset.profile import ProfileConfig, SalesDataProfile
from valuation.dataset.sales import SalesDataPrep, SalesDataPrepConfig
from valuation.dataset.split import DatasetSplitter, PathsConfig, SplitterConfig

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


def split_sales_data(force: bool) -> None:
    """Splits the sales data into training, validation, and test sets."""
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
    # Obtain categories and filenames from config
    config_reader = ConfigReader()
    category_filenames = config_reader.read(CONFIG_CATEGORY_FILEPATH)
    # Create profiling dataset
    profile(category_filenames=category_filenames, force=force)
    prepare_sales_data(category_filenames=category_filenames, force=force)
    split_sales_data(force=force)


if __name__ == "__main__":
    app()
