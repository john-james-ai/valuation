#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/raw_datagen.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 04:53:07 am                                                #
# Modified   : Sunday October 19th 2025 02:15:14 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import cast

from dataclasses import dataclass
from datetime import datetime

from loguru import logger
import typer

from valuation.app.dataprep.task import DatasetTaskResult
from valuation.app.state import Status
from valuation.asset.stage import DatasetStage
from valuation.asset.types import AssetType
from valuation.core.structure import DataClass
from valuation.infra.file.base import FileSystem
from valuation.infra.file.io import IOService
from valuation.infra.loggers import configure_logging


@dataclass
class RawDataGeneratorConfig(DataClass):
    """Holds data related to the current operating mode."""

    category_config_filepath: str = "config.yaml"


# ------------------------------------------------------------------------------------------------ #
class RawDataGenerator:
    """Generates ModeSalesDataConfig based on the current operating mode."""

    def __init__(self, config: RawDataGeneratorConfig, io: IOService = IOService) -> None:
        self._config = config
        self._io = io
        self._file_system = FileSystem(asset_type=AssetType.DATASET)
        self._result = DatasetTaskResult(task_name=self.__class__.__name__, dataset_name="sales")

    def run(self, mode: str) -> DatasetTaskResult:
        """Generates the mode sales data."""
        if mode == "prod":
            raise RuntimeError("Raw data generation is not allowed in 'prod' mode.")

        self._result.started = datetime.now()

        # Read the config file
        category_config = self._io.read(filepath=self._config.category_config_filepath)

        # Get the dataset path
        dataset_path = self._file_system.get_asset_filepath(
            passport_or_stage=DatasetStage.INGEST, name="sales", format="parquet", mode=mode
        )

        # Read the dataset
        df = self._io.read(filepath=dataset_path)
        self._result.records_in = len(df)

        for _, category_info in category_config["category_filenames"].items():
            category = category_info["category"]
            name, format = category_info["filename"].split(".")
            filepath = self._file_system.get_asset_filepath(
                passport_or_stage=DatasetStage.RAW, name=name, format=format, mode=mode
            )

            # Remove stage and mode from filenames
            substring = f"_{DatasetStage.RAW.value}_{mode}"
            filepath = filepath.with_name(filepath.name.replace(substring, ""))

            # Filter dataset by category
            df_category = df[df["CATEGORY"] == category]
            if df_category.empty or df_category is None:
                logger.warning(f"No records found for category '{category}'. Skipping.")
                continue
            self._result.records_out = cast(int, self._result.records_out)
            self._result.records_out += len(df_category)
            # Save the category dataset
            self._io.write(data=df_category, filepath=filepath)

        # Update task result
        self._result.end_task(status=Status.SUCCESS)

        return self._result


# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


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
    mode: str = typer.Option(
        "test",
        "--mode",
        "-m",
        case_sensitive=False,
        help="Mode: valid values are 'test', 'prod', 'dev'. Defaults to 'test'.",
    ),
):
    """Main entry point for the Valuation package."""
    # Configure logging
    configure_logging()
    # Create the mode sales data generator
    config = RawDataGeneratorConfig()
    generator = RawDataGenerator(config=config)

    # Generate the mode sales data
    result = generator.run(mode=mode)

    # Log the result
    logger.info(result)


if __name__ == "__main__":
    app()
