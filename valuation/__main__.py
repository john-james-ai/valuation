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
# Modified   : Wednesday October 22nd 2025 11:54:46 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""

from typing import Optional

from pathlib import Path

import typer

from devops.raw_datagen import MODE
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.entity import Entity
from valuation.core.file import FileFormat
from valuation.core.stage import DatasetStage
from valuation.flow.dataprep.sales.pipeline import (
    SalesDataPrepPipelineBuilder,
    SalesDataPrepPipelineResult,
)
from valuation.infra.loggers import configure_logging

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()
# ------------------------------------------------------------------------------------------------ #
CONFIG_FILEPATH = Path("config.yaml")
WEEK_DECODE_TABLE_FILEPATH = Path("reference/week_decode_table.csv")
# ------------------------------------------------------------------------------------------------ #


def run_sales_data_pipeline(force: bool = False) -> Optional[SalesDataPrepPipelineResult]:
    """Runs the sales data preparation pipeline.

    Args:
        force (bool): Whether to force reprocessing if the file already exists.

    Returns:
        PipelineResult: Result object from executing the pipeline.
    """
    passport = DatasetPassport.create(
        name="sales_final",
        description="Final cleaned and aggregated sales data",
        entity=Entity.SALES,
        stage=DatasetStage.FINAL,
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
            "partition_cols": ["category", "year"],
        },
    )

    pipeline = (
        SalesDataPrepPipelineBuilder()
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
    persist: bool = typer.Option(
        False,
        "--persist",
        "-p",
        case_sensitive=False,
        help="Whether to persist intermediate files.",
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
    # Construct the data preparation pipeline tasks
    run_sales_data_pipeline(force=force)


if __name__ == "__main__":
    app()
