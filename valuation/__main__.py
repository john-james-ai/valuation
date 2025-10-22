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
# Modified   : Tuesday October 21st 2025 07:02:45 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

import typer

from devops.raw_datagen import MODE
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.entity import Entity
from valuation.core.file import FileFormat
from valuation.core.stage import DatasetStage
from valuation.flow.base.pipeline import PipelineConfig, PipelineResult
from valuation.flow.dataprep.pipeline import DataPrepPipeline
from valuation.flow.dataprep.sales.aggregate import (
    NON_NEGATIVE_COLUMNS_AGGREGATE,
    REQUIRED_COLUMNS_AGGREGATE,
    AggregateSalesDataTask,
)
from valuation.flow.dataprep.sales.clean import (
    NON_NEGATIVE_COLUMNS_CLEAN,
    REQUIRED_COLUMNS_CLEAN,
    CleanSalesDataTask,
)
from valuation.flow.dataprep.sales.ingest import (
    CONFIG_FILEPATH,
    NON_NEGATIVE_COLUMNS_INGEST,
    REQUIRED_COLUMNS_INGEST,
    WEEK_DECODE_TABLE_FILEPATH,
    IngestSalesDataTask,
    IngestSalesDataTaskConfig,
)
from valuation.flow.dataprep.task import DataPrepTask, SISODataPrepTaskConfig
from valuation.flow.validation import ValidationBuilder
from valuation.infra.loggers import configure_logging
from valuation.infra.store.dataset import DatasetStore

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
def get_aggregate_sales_data_task(source: DatasetPassport) -> DataPrepTask:
    """Aggregates cleaned sales data to the store-category-week level.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Build a validator for the clean sales data task
    validation = (
        ValidationBuilder()
        .with_column_type_validator(column_types=REQUIRED_COLUMNS_AGGREGATE)
        .with_missing_column_validator(required_columns=REQUIRED_COLUMNS_AGGREGATE.keys())
        .with_non_negative_column_validator(NON_NEGATIVE_COLUMNS_AGGREGATE)
        .build()
    )

    # Create the output DatasetPassport for the aggregated sales data
    passport = DatasetPassport.create(
        name="sales_aggregated",
        description="Dominick's Aggregated Sales Data Dataset",
        stage=DatasetStage.PROCESSED,
        entity=Entity.SALES,
        file_format=FileFormat.CSV,
    )

    # Create configuration for sales data processing
    config = SISODataPrepTaskConfig(
        source=source,
        target=passport,
    )

    # Instantiate Dataset Store
    dataset_store = DatasetStore()

    # Run the sales data processing task
    return AggregateSalesDataTask(config=config, dataset_store=dataset_store, validation=validation)


# ------------------------------------------------------------------------------------------------ #
def get_clean_sales_data_task(source: DatasetPassport) -> CleanSalesDataTask:

    # Build a validator for the clean sales data task
    validation = (
        ValidationBuilder()
        .with_column_type_validator(column_types=REQUIRED_COLUMNS_CLEAN)
        .with_missing_column_validator(required_columns=REQUIRED_COLUMNS_CLEAN.keys())
        .with_non_negative_column_validator(NON_NEGATIVE_COLUMNS_CLEAN)
        .build()
    )

    # Create the output DatasetPassport for the cleaned sales data
    passport = DatasetPassport.create(
        name="sales_clean",
        description="Dominick's Clean Sales Data Dataset",
        stage=DatasetStage.CLEAN,
        entity=Entity.SALES,
        file_format=FileFormat.PARQUET,
        read_kwargs=source.read_kwargs,
        write_kwargs=source.write_kwargs,
    )
    # Create configuration for sales data processing
    config = SISODataPrepTaskConfig(
        source=source,
        target=passport,
    )

    # Instantiate Dataset Store
    dataset_store = DatasetStore()

    # Run the sales data processing task
    return CleanSalesDataTask(config=config, dataset_store=dataset_store, validation=validation)


# ------------------------------------------------------------------------------------------------ #
def get_ingest_sales_data_task() -> IngestSalesDataTask:
    """Ingests raw sales data files.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Build a validator for the ingest sales data task
    validation = (
        ValidationBuilder()
        .with_column_type_validator(column_types=REQUIRED_COLUMNS_INGEST)
        .with_missing_column_validator(required_columns=REQUIRED_COLUMNS_INGEST.keys())
        .with_non_negative_column_validator(NON_NEGATIVE_COLUMNS_INGEST)
        .build()
    )
    # Create DatasetPassport for Target Dataset
    passport = DatasetPassport.create(
        name="sales_ingest",
        description="Dominick's Ingested Sales Data Dataset",
        stage=DatasetStage.INGEST,
        entity=Entity.SALES,
        file_format=FileFormat.PARQUET,
        read_kwargs={
            "engine": "pyarrow",
            "columns": None,
            "filters": None,
            "use_threads": True,
            "dtype_backend": "pyarrow",
        },
        write_kwargs={
            "engine": "pyarrow",
            "compression": "snappy",
            "index": False,
            "row_group_size": 256_000,
            "partition_cols": ["CATEGORY", "YEAR"],
        },
    )

    config = IngestSalesDataTaskConfig(
        week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH,
        source=CONFIG_FILEPATH,
        target=passport,
    )
    # Instantiate Dataset Store
    dataset_store = DatasetStore()
    # Return the sales data processing task
    return IngestSalesDataTask(config=config, dataset_store=dataset_store, validation=validation)


def run_sales_data_pipeline(force: bool) -> PipelineResult:
    """Runs the sales data preparation pipeline.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.
    """
    # Construct the data preparation pipeline tasks
    ingest_sales_data_task = get_ingest_sales_data_task()
    clean_sales_data_task = get_clean_sales_data_task(ingest_sales_data_task.config.target)  # type: ignore
    aggregate_sales_data_task = get_aggregate_sales_data_task(clean_sales_data_task.config.target)

    # Pipeline Configuration
    config = PipelineConfig(name="Sales Data Preparation Pipeline")

    # Create and run the sales data pipeline
    return (
        DataPrepPipeline(config=config)
        .add_task(ingest_sales_data_task)
        .add_task(clean_sales_data_task)  # type: ignore
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
    if str(MODE).lower() == "prod":
        go = input("This will run in 'prod' mode. Are you sure you want to continue? (y/n):")
        if go.lower() != "y":
            print("Exiting...")
            raise SystemExit(0)
    # Configure logging
    configure_logging()
    # Construct the data preparation pipeline tasks
    result = run_sales_data_pipeline(force=force)
    print(result)


if __name__ == "__main__":
    app()
