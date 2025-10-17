#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/__main__.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 11:01:16 pm                                               #
# Modified   : Friday October 17th 2025 04:26:43 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

import typer

from valuation.config.filepaths import (
    CONFIG_FILEPATH,
    FILEPATH_SALES_PROCESSED_SCW,
    RAW_DATA_DIR,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.workflow.dataprep.sales.aggregate import AggregateSalesDataTask
from valuation.workflow.dataprep.sales.clean import CleanSalesDataTask
from valuation.workflow.dataprep.sales.ingest import IngestSalesDataTask, IngestSalesDataTaskConfig
from valuation.workflow.pipeline import Pipeline, PipelineConfig, PipelineResult
from valuation.workflow.task import Task, TaskConfig

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


# # ------------------------------------------------------------------------------------------------ #
# def get_ingest_store_demo_data_task() -> Task:
#     """Ingests raw store demographic data file.
#     Args:
#         force (bool): Whether to force reprocessing if the file already exists.
#     """
#     # Create configuration for store demographic data processing

#     config = TaskConfig(
#         dataset_name="Dominick's Store Demo Data - Ingestion",
#         input_location=FILEPATH_STORE_DEMO_RAW,
#         output_location=FILEPATH_STORE_DEMO_INGEST,
#     )
#     # Run the sales data processing task
#     return IngestStoreDemoDataTask(config=config)


# # ------------------------------------------------------------------------------------------------ #
# #                                CUSTOMER DATA PIPELINE                                            #
# # ------------------------------------------------------------------------------------------------ #
# def get_ingest_customer_data_task() -> Task:
#     """Ingests raw customer data files.
#     Args:
#         force (bool): Whether to force reprocessing if the file already exists.
#     """
#     # Create configuration for sales data processing

#     config = TaskConfig(
#         dataset_name="Dominick's Customer Data - Ingestion",
#         input_location=FILEPATH_CUSTOMER_RAW,
#         output_location=FILEPATH_CUSTOMER_INGEST,
#     )
#     # Run the sales data processing task
#     return IngestCustomerDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
#                                   SALES DATA PIPELINE                                            #
# ------------------------------------------------------------------------------------------------ #
def get_aggregate_sales_data_task() -> Task:
    """Aggregates cleaned sales data to the store-category-week level.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for sales data processing

    config = TaskConfig(
        task_name="AggregateSalesDataTask",
        dataset_name="Dominick's Sales Data - Aggregation",
    )
    # Run the sales data processing task
    return AggregateSalesDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
def get_clean_sales_data_task() -> Task:

    # Create configuration for sales data processing

    config = TaskConfig(
        task_name="CleanSalesDataTask",
        dataset_name="Dominick's Sales Data - Clean",
    )
    # Run the sales data processing task
    return CleanSalesDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_sales_data_task() -> Task:
    """Ingests raw sales data files.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create configuration for sales data processing

    config = IngestSalesDataTaskConfig(
        task_name="IngestSalesDataTask",
        dataset_name="Dominick's Sales Data - Ingestion",
        week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH,
        raw_data_directory=RAW_DATA_DIR,
    )
    # Run the sales data processing task
    return IngestSalesDataTask(config=config)


def run_sales_data_pipeline(force: bool) -> PipelineResult:
    """Runs the sales data preparation pipeline.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Construct the data preparation pipeline tasks
    ingest_sales_data_task = get_ingest_sales_data_task()
    clean_sales_data_task = get_clean_sales_data_task()
    aggregate_sales_data_task = get_aggregate_sales_data_task()

    # Pipeline Configuration
    config = PipelineConfig(
        name="Sales Data Preparation Pipeline",
        dataset_name="Dominick's Sales Data",
        description="Pipeline to prepare sales data for analysis.",
        input_location=CONFIG_FILEPATH,
        output_location=FILEPATH_SALES_PROCESSED_SCW,
    )

    # Create and run the sales data pipeline
    return (
        Pipeline(config=config)
        .add_task(ingest_sales_data_task)
        .add_task(clean_sales_data_task)
        .add_task(aggregate_sales_data_task)
        .run(force=force)
    )


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
    result = run_sales_data_pipeline(force=force)
    print(result)
    result.summary


if __name__ == "__main__":
    app()
