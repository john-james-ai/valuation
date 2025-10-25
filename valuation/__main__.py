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
# Modified   : Saturday October 25th 2025 04:45:02 pm                                              #
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
from valuation.core.file import FileFormat
from valuation.core.stage import DatasetStage
from valuation.flow.dataprep.pipeline.clean import (
    CleanSalesDataPipelineBuilder,
    CleanSalesDataPipelineResult,
)
from valuation.flow.dataprep.pipeline.model_dataprep import (
    TrainDataPipelineBuilder,
    TrainDataPipelineConfig,
    TrainDataPipelineResult,
)
from valuation.flow.dataprep.pipeline.transform import (
    TransformSalesDataPipelineBuilder,
    TransformSalesDataPipelineResult,
)
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
def run_model_data_pipeline(force: bool = False) -> Optional[TrainDataPipelineResult]:

    source = DatasetPassport.create(
        name="sales",
        description="Transformed Sales Data",
        stage=DatasetStage.TRANSFORM,
        file_format=FileFormat.PARQUET,
    )
    target = DatasetPassport.create(
        name="sales_full",
        description="Full Sales Dataset for Training",
        stage=DatasetStage.TRAIN,
        file_format=FileFormat.PARQUET,
    )
    train_val_target = DatasetPassport.create(
        name="sales_train_val",
        description="Training and Validation Set",
        stage=DatasetStage.TRAIN,
        file_format=FileFormat.PARQUET,
    )
    test_target = DatasetPassport.create(
        name="sales_test",
        description="Test Set",
        stage=DatasetStage.TRAIN,
        file_format=FileFormat.PARQUET,
    )
    config = TrainDataPipelineConfig(
        source=source, target=target, train_val_target=train_val_target, test_target=test_target
    )

    pipeline = (
        TrainDataPipelineBuilder()
        .with_config(config=config)
        .with_densify()
        .with_feature_engineering()
        .build()
    )
    return pipeline.run(force=force)


# ------------------------------------------------------------------------------------------------ #
def run_sales_data_transform_pipeline(
    force: bool = False,
) -> Optional[TransformSalesDataPipelineResult]:
    store = DatasetStore()
    source = store.file_system.get_asset_stage_location(stage=DatasetStage.CLEAN)
    target = DatasetPassport.create(
        name="sales",
        description="Transformed and Aggregated Sales Data",
        stage=DatasetStage.TRANSFORM,
        file_format=FileFormat.PARQUET,
    )
    pipeline = (
        TransformSalesDataPipelineBuilder()
        .with_source(source=source)
        .with_target(target=target)
        .with_full_year_filter_task(min_weeks=50)
        .with_aggregate_task()
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
        name="sales",
        description="Cleaned sales data",
        stage=DatasetStage.CLEAN,
        file_format=FileFormat.PARQUET,
    )

    pipeline = (
        CleanSalesDataPipelineBuilder()
        .with_source(source=CONFIG_FILEPATH)
        .with_target(target=passport)
        .with_ingest_task(week_decode_table_filepath=WEEK_DECODE_TABLE_FILEPATH)
        .with_clean_task()
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
        help="Run Train Data Prep Pipeline.",
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
