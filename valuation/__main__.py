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
# Modified   : Tuesday October 14th 2025 11:00:18 pm                                               #
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
    FILEPATH_SALES_CLEAN,
    FILEPATH_SALES_INGEST,
    FILEPATH_STORE_DEMO_INGEST,
    FILEPATH_STORE_DEMO_RAW,
    RAW_DATA_DIR,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.dataprep.base import Task, TaskConfig
from valuation.dataprep.clean import CleanTask, CleanTaskConfig
from valuation.dataprep.ingest import (
    IngestCustomerDataTask,
    IngestSalesDataTask,
    IngestSalesDataTaskConfig,
    IngestStoreDemoDataTask,
)
from valuation.dataprep.pipeline import DataPrepPipeline

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


# ------------------------------------------------------------------------------------------------ #
def get_clean_sales_data_task() -> Task:

    # Create configuration for sales data processing

    config = CleanTaskConfig(
        dataset_name="Dominick's Sales Data - Clean",
        input_location=FILEPATH_SALES_INGEST,
        output_location=FILEPATH_SALES_CLEAN,
    )
    # Run the sales data processing task
    return CleanTask(config=config)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_store_demo_data_task() -> Task:
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
    return IngestStoreDemoDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_customer_data_task() -> Task:
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
    return IngestCustomerDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_sales_data_task() -> Task:
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
    return IngestSalesDataTask(config=config)


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
    # Construct the data preparation pipeline tasks
    ingest_sales_data_task = get_ingest_sales_data_task()
    clean_sales_data_task = get_clean_sales_data_task()
    ingest_customer_data_task = get_ingest_customer_data_task()
    ingest_store_demo_data_task = get_ingest_store_demo_data_task()

    # Create and populate the data preparation pipeline
    pipeline = DataPrepPipeline()
    pipeline.add_task(ingest_sales_data_task)
    pipeline.add_task(clean_sales_data_task)
    pipeline.add_task(ingest_customer_data_task)
    pipeline.add_task(ingest_store_demo_data_task)

    # Run the data preparation pipeline
    pipeline.run(force=force)


if __name__ == "__main__":
    app()
