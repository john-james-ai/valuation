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
# Modified   : Thursday October 23rd 2025 09:10:58 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

from typing import Optional

import os
from pathlib import Path

from dotenv import load_dotenv
import typer

from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.entity import Entity
from valuation.core.file import FileFormat
from valuation.core.stage import DatasetStage
from valuation.flow.dataprep.sales.pipeline.clean import (
    CleanSalesDataPipelineBuilder,
    CleanSalesDataPipelineResult,
)
from valuation.flow.dataprep.sales.pipeline.model_data import (
    ModelDataPipelineBuilder,
    ModelDataPipelineConfig,
    ModelDataPipelineResult,
)
from valuation.flow.dataprep.sales.pipeline.transform import (
    TransformSalesDataPipelineBuilder,
    TransformSalesDataPipelineResult,
)
from valuation.flow.dataprep.sales.task.filter import MIN_WEEKS_PER_YEAR
from valuation.infra.loggers import configure_logging
from valuation.infra.store.dataset import DatasetStore

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()
load_dotenv()
MODE = os.getenv("MODE", "dev")
# ------------------------------------------------------------------------------------------------ #
CONFIG_FILEPATH = Path("config.yaml")
WEEK_DECODE_TABLE_FILEPATH = Path("reference/week_decode_table.csv")


# ------------------------------------------------------------------------------------------------ #
def run_model_data_pipeline(force: bool = False) -> Optional[ModelDataPipelineResult]:

    source = DatasetPassport.create(
        name="sales_transform",
        description="Transformed Sales Data",
        entity=Entity.SALES,
        stage=DatasetStage.TRANSFORM,
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
            "partition_cols": None,
        },
    )
    train_val_target = DatasetPassport.create(
        name="train_val",
        description="Training and Validation Set",
        entity=Entity.SALES,
        stage=DatasetStage.MODEL,
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
            "partition_cols": None,
        },
    )
    test_target = DatasetPassport.create(
        name="test",
        description="Test Set",
        entity=Entity.SALES,
        stage=DatasetStage.MODEL,
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
            "partition_cols": None,
        },
    )
    config = ModelDataPipelineConfig(
        source=source, train_val_target=train_val_target, test_target=test_target
    )

    pipeline = (
        ModelDataPipelineBuilder().with_config(config=config).with_feature_engineering().build()
    )
    return pipeline.run(force=force)


# ------------------------------------------------------------------------------------------------ #
def run_sales_data_transform_pipeline(
    force: bool = False,
) -> Optional[TransformSalesDataPipelineResult]:
    store = DatasetStore()
    source = store.file_system.get_stage_entity_location(
        stage=DatasetStage.CLEAN, entity=Entity.SALES
    )
    target = DatasetPassport.create(
        name="sales_transform",
        description="Transformed Sales Data",
        entity=Entity.SALES,
        stage=DatasetStage.TRANSFORM,
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
            "partition_cols": None,
        },
    )
    pipeline = (
        TransformSalesDataPipelineBuilder()
        .with_source(source=source)
        .with_target(target=target)
        .with_aggregate_task()
        .with_full_year_filter_task(min_weeks=MIN_WEEKS_PER_YEAR)
        .build()
    )
    return pipeline.run(force=force)


def run_sales_data_clean_pipeline(force: bool = False) -> Optional[CleanSalesDataPipelineResult]:
    """Runs the sales data preparation pipeline.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.

    Returns:
        PipelineResult: Result object from executing the pipeline.
    """
    passport = DatasetPassport.create(
        name="sales_clean",
        description="Cleaned sales data",
        entity=Entity.SALES,
        stage=DatasetStage.CLEAN,
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
            "partition_cols": None,
        },
    )

    pipeline = (
        CleanSalesDataPipelineBuilder()
        .with_source(source=CONFIG_FILEPATH)
        .with_target(target=passport)
        .with_ingest_task(week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH)
        .with_clean_task()
        .with_aggregate_task()
        .with_aggregate_task()
        .build()
    )
    return pipeline.run(force=force)


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
    clean: bool = typer.Option(
        False,
        "--clean",
        "-c",
        case_sensitive=False,
        help="Run Data Clean Pipeline.",
    ),
    transform: bool = typer.Option(
        False,
        "--transform",
        "-t",
        case_sensitive=False,
        help="Run Data Transform Pipeline.",
    ),
    model_data: bool = typer.Option(
        False,
        "--model_data",
        "-m",
        case_sensitive=False,
        help="Run Model Data Prep Pipeline.",
    ),
):
    """Main entry point for the Valuation package."""
    if str(MODE).lower() == "prod":
        go = input("This will run in 'prod' mode. Are you sure you want to continue? (y/n):")
        if go.lower() != "y":
            print("Exiting...")
            raise SystemExit(0)
    # Configure logging
    configure_logging()
    if clean:
        # Only run clean pipeline
        run_sales_data_clean_pipeline(force=force)
    elif transform:
        # Only run transform pipeline
        run_sales_data_transform_pipeline(force=force)
    elif model_data:
        # Only run transform pipeline
        run_model_data_pipeline(force=force)
    else:
        # Default: run both pipelines
        run_sales_data_clean_pipeline(force=force)
        run_sales_data_transform_pipeline(force=force)


if __name__ == "__main__":
    app()
