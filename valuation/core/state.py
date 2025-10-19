#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/state.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 11:01:16 pm                                               #
# Modified   : Saturday October 18th 2025 08:20:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

import typer

from valuation.asset.identity import AssetType, DatasetStage, Passport
from valuation.config.data import DTYPES
from valuation.config.filepaths import (
    CONFIG_FILEPATH,
    DATASET_STORE_DIR,
    FILEPATH_SALES_PROCESSED_SCW,
    RAW_DATA_DIR,
    WEEK_DECODE_TABLE_FILEPATH,
)
from valuation.config.loggers import configure_logging
from valuation.utils.db.dataset import DatasetStore
from valuation.utils.io.service import IOService
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
#         input_filepath=FILEPATH_STORE_DEMO_RAW,
#         output_filepath=FILEPATH_STORE_DEMO_INGEST,
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
#         input_filepath=FILEPATH_CUSTOMER_RAW,
#         output_filepath=FILEPATH_CUSTOMER_INGEST,
#     )
#     # Run the sales data processing task
#     return IngestCustomerDataTask(config=config)


# ------------------------------------------------------------------------------------------------ #
#                                   SALES DATA PIPELINE                                            #
# ------------------------------------------------------------------------------------------------ #
def get_aggregate_sales_data_task(source: Passport) -> Task:
    """Aggregates cleaned sales data to the store-category-week level.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Create the output Passport for the aggregated sales data
    passport = Passport.create(
        name="sales_aggregated",
        description="Dominick's Aggregated Sales Data Dataset",
        stage=DatasetStage.PROCESSED,
        type=AssetType.DATASET,
        format="csv",
    )

    # Create configuration for sales data processing
    config = TaskConfig(
        task_name="AggregateSalesDataTask",
        dataset_name=passport.name,
        description=passport.description,
        source=source,
        target=passport,
    )

    # Instantiate Dataset Store
    dataset_store = DatasetStore(location=DATASET_STORE_DIR)

    # Run the sales data processing task
    return AggregateSalesDataTask(config=config, dataset_store=dataset_store)


# ------------------------------------------------------------------------------------------------ #
def get_clean_sales_data_task(source: Passport) -> Task:

    # Create the output Passport for the cleaned sales data
    passport = Passport.create(
        name="sales_clean",
        description="Dominick's Clean Sales Data Dataset",
        stage=DatasetStage.CLEAN,
        type=AssetType.DATASET,
        format="parquet",
    )

    # Create configuration for sales data processing
    config = TaskConfig(
        task_name="CleanSalesDataTask",
        dataset_name=passport.name,
        description=passport.description,
        source=source,
        target=passport,
    )

    # Instantiate Dataset Store
    dataset_store = DatasetStore(location=DATASET_STORE_DIR)

    # Run the sales data processing task
    return CleanSalesDataTask(config=config, dataset_store=dataset_store)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_sales_data_task() -> Task:
    """Ingests raw sales data files.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Obtain prerequisite data
    week_dates = IOService.read(filepath=WEEK_DECODE_TABLE_FILEPATH, dtype=DTYPES)
    filepaths = IOService.read(filepath=CONFIG_FILEPATH)
    # Create Passport for Target Dataset
    passport = Passport.create(
        name="sales_ingestion",
        description="Dominick's Sales Data Ingestion Dataset",
        stage=DatasetStage.INGEST,
        type=AssetType.DATASET,
        format="parquet",
    )

    config = IngestSalesDataTaskConfig(
        task_name="IngestSalesDataTask",
        description=passport.description,
        dataset_name=passport.name,
        week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH,
        source=filepaths,
        target=passport,
        raw_data_directory=RAW_DATA_DIR,
    )
    # Instantiate Dataset Store
    dataset_store = DatasetStore(location=DATASET_STORE_DIR)
    # Run the sales data processing task
    return IngestSalesDataTask(config=config, dataset_store=dataset_store)


def run_sales_data_pipeline(force: bool) -> PipelineResult:
    """Runs the sales data preparation pipeline.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Construct the data preparation pipeline tasks
    ingest_sales_data_task = get_ingest_sales_data_task()
    clean_sales_data_task = get_clean_sales_data_task(ingest_sales_data_task.config.target)
    aggregate_sales_data_task = get_aggregate_sales_data_task(clean_sales_data_task.config.target)

    # Pipeline Configuration
    config = PipelineConfig(
        name="Sales Data Preparation Pipeline",
        dataset_name="Dominick's Sales Data",
        description="Pipeline to prepare sales data for analysis.",
        input_filepath=CONFIG_FILEPATH,
        output_filepath=FILEPATH_SALES_PROCESSED_SCW,
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
