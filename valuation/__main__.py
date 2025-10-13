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
# Modified   : Monday October 13th 2025 10:28:55 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

import typer

from valuation.config.filepaths import (
    CONFIG_FILEPATH,
    FILEPATH_CUSTOMER_INGEST,
    FILEPATH_CUSTOMER_RAW,
    FILEPATH_SALES_INGEST,
    FILEPATH_STORE_DEMO_INGEST,
    FILEPATH_STORE_DEMO_RAW,
    RAW_DATA_DIR,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.dataprep.base import TaskConfig
from valuation.dataprep.clean import CleanTask, CleanTaskConfig
from valuation.dataprep.ingest import (
    IngestCustomerDataTask,
    IngestSalesDataTask,
    IngestSalesDataTaskConfig,
    IngestStoreDemoDataTask,
)

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
def clean(force: bool) -> None:

    # Create configuration for sales data processing

    config = CleanTaskConfig(
        dataset_name="Dominick's Sales Data - Ingestion",
        input_location=CONFIG_FILEPATH,
        output_location=FILEPATH_SALES_INGEST,
    )
    # Run the sales data processing task
    task = CleanTask(config=config)
    task.run(force=force)


# ------------------------------------------------------------------------------------------------ #
def ingest_store_demo_data(force: bool) -> None:
    """Ingests raw store demographic data file.
    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for store demographic data processing

    config = TaskConfig(
        dataset_name="Dominick's Store Demo Data - Ingestion",
        input_location=FILEPATH_STORE_DEMO_RAW,
        output_location=FILEPATH_STORE_DEMO_INGEST,
    )
    # Run the sales data processing task
    task = IngestStoreDemoDataTask(config=config)
    task.run(force=force)


# ------------------------------------------------------------------------------------------------ #
def ingest_customer_data(force: bool) -> None:
    """Ingests raw customer data files.
    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for sales data processing

    config = TaskConfig(
        dataset_name="Dominick's Customer Data - Ingestion",
        input_location=FILEPATH_CUSTOMER_RAW,
        output_location=FILEPATH_CUSTOMER_INGEST,
    )
    # Run the sales data processing task
    task = IngestCustomerDataTask(config=config)
    task.run(force=force)


# ------------------------------------------------------------------------------------------------ #
def ingest_sales(force: bool) -> None:
    """Ingests raw sales data files.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for sales data processing

    config = IngestSalesDataTaskConfig(
        dataset_name="Dominick's Sales Data - Ingestion",
        input_location=CONFIG_FILEPATH,
        output_location=FILEPATH_SALES_INGEST,
        week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH,
        raw_data_directory=RAW_DATA_DIR,
    )
    # Run the sales data processing task
    task = IngestSalesDataTask(config=config)
    task.run(force=force)


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
    # Run Pipelines
    ingest_sales(force=force)
    ingest_customer_data(force=force)
    ingest_store_demo_data(force=force)
    clean(force=force)


if __name__ == "__main__":
    app()
