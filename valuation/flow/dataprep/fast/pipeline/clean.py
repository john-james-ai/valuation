#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/fast/pipeline/clean.py                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Saturday October 25th 2025 04:57:02 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Optional

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import polars as pl
from tqdm import tqdm

from valuation.asset.dataset.fast.dataset import Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.stage import DatasetStage
from valuation.core.state import Status
from valuation.flow.base.pipeline import PipelineResult
from valuation.flow.dataprep.fast.base.pipeline import DataPrepPipeline, DataPrepPipelineBuilder
from valuation.flow.dataprep.fast.task.clean import CleanSalesDataTask
from valuation.flow.dataprep.fast.task.filter import FilterPartialYearsTask
from valuation.flow.dataprep.fast.task.ingest import (
    NON_NEGATIVE_COLUMNS_INGEST,
    REQUIRED_COLUMNS_INGEST,
    IngestSalesDataTask,
)
from valuation.flow.dataprep.fast.validation import Validation, ValidationBuilder
from valuation.infra.file.fastio import IOService
from valuation.infra.store.fast.dataset import DatasetStore

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class CleanSalesDataPipelineResult(PipelineResult):
    """Holds the results of a sales pipeline execution.

    Attributes:
        (inherited) DataPrepPipelineResult fields containing execution metadata and outputs.
    """

    num_datasets: int = 0
    num_errors: int = 0
    num_warnings: int = 0

    dataset: Optional[Dataset] = None


# ------------------------------------------------------------------------------------------------ #


class CleanSalesDataPipeline(DataPrepPipeline):
    """
    Pipeline implementation for preparing sales datasets using Polars.

    Args:
        dataset_store: The DatasetStore class or factory used by the pipeline.
        result: Result class used to create result instances.
    """

    _dataset_store: DatasetStore
    _result: CleanSalesDataPipelineResult

    def __init__(
        self,
        dataset_store: type[DatasetStore] = DatasetStore,
        result: type[CleanSalesDataPipelineResult] = CleanSalesDataPipelineResult,
    ) -> None:
        super().__init__(dataset_store=dataset_store)
        self._result = result(name=self.__class__.__name__)
        self._source = None
        self._target = None

    def run(self, force: bool = False) -> Optional[CleanSalesDataPipelineResult]:
        """
        Execute the configured pipeline over source categories and produce a dataset.

        Args:
            force: If True, force reprocessing even when target dataset already exists.

        Returns:
            Optional[CleanSalesDataPipelineResult]: Pipeline result object if execution completes.
        """
        self._result.start_pipeline()

        try:
            # Read category filenames mapping
            category_filenames = IOService.read(filepath=Path(self._source))[
                CONFIG_CATEGORY_INFO_KEY
            ]
            logger.debug(f"Category filenames mapping loaded: {len(category_filenames)}")

            # Get the stage and entity from the target passport
            directory = self._dataset_store.file_system.get_stage_entity_location(
                stage=DatasetStage.RAW,
            )
            logger.debug(f"Raw sales data directory: {directory}\n")

            # Create tqdm progress bar for categories
            pbar = tqdm(
                category_filenames.items(),
                total=len(category_filenames),
                desc="Processing Sales Data by Category",
                unit="category",
            )

            # Iterate through category sales files
            for _, category_info in pbar:
                # Initialize empty LazyFrame for category
                category_frames = []

                # Get filename and category
                filename = category_info["filename"]
                filepath = directory / filename
                filepath = Path(filepath)
                category = category_info["category"]
                pbar.set_description(f"Processing category: {category} from file: {filename}")

                # Set target dataset name
                if self._target is not None:
                    self._target.name = "sales_" + category.replace(" ", "_").lower()

                    # Check if dataset already exists and skip if not forcing
                    if self._dataset_store.exists(dataset_id=self._target.id) and not force:
                        logger.debug(
                            f"Dataset {self._target.label} already exists in the datastore.\n"
                            "Skipping processing."
                        )
                        continue

                # Remove existing category-level data if any exists
                if self._dataset_store.exists(dataset_id=self._target.id):  # type: ignore
                    self._dataset_store.remove(passport=self._target)

                # Load the data (lazy by default)
                df = self._load(filepath=filepath, lazy=True)

                # Process through tasks
                for task in self._tasks:
                    # Run the task (returns LazyFrame)
                    df = task.run(df=df, category=category, lazy=True)

                    # Collect for validation (validates eagerly)
                    df_validated = df.collect()

                    # Validate the result
                    task.validation.validate(df=df_validated, classname=task.__class__.__name__)

                    # Update metrics
                    self._update_metrics(validation=task.validation)

                    # Check validation status
                    if not task.validation.is_valid:
                        self._result.status_obj = Status.FAIL
                        task.validation.report()
                        break

                    # Convert back to lazy for next task
                    df = df_validated.lazy()

                # Collect final result and append to category frames
                category_frames.append(df.collect())

                # Concatenate all frames for this category
                if category_frames:
                    df_category = pl.concat(category_frames, how="vertical")
                else:
                    df_category = pl.DataFrame()

                # Create and persist the category-level data to the dataset store
                self._target.name = "sales_" + category.replace(" ", "_").lower()  # type: ignore
                self._dataset_store.remove(passport=self._target)  # type: ignore
                dataset_category = Dataset(passport=self._target, df=df_category)
                self._dataset_store.add(dataset=dataset_category, overwrite=force)

                # Update result dataset reference
                self._result.num_datasets += 1

            self._result.end_pipeline()
            logger.info(self._result)

            return self._result

        except Exception as e:
            self._result.status_obj = Status.FAIL
            logger.critical(f"Pipeline execution failed with exception: {e}")
            self._result.end_pipeline()
            raise e

    def _update_metrics(self, validation: Validation) -> None:
        """
        Update pipeline result metrics based on validation results.

        Args:
            validation: The validation object containing metrics to update.
        """
        self._result.num_errors += validation.num_errors
        self._result.num_warnings += validation.num_warnings


# ------------------------------------------------------------------------------------------------ #
class CleanSalesDataPipelineBuilder(DataPrepPipelineBuilder):
    """Builder for CleanSalesDataPipeline instances.

    Use the fluent `with_*` methods to configure source, target and tasks, then call `build()`.

    Attributes:
        _pipeline (CleanSalesDataPipeline): The pipeline instance being built.
    """

    _pipeline: CleanSalesDataPipeline

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """Reset the internal builder state and create a fresh pipeline instance.

        Returns:
            None
        """
        self._pipeline = CleanSalesDataPipeline()

    def with_source(self, source: str) -> CleanSalesDataPipelineBuilder:
        """Set the pipeline source (category mapping file path).

        Args:
            source (str): Path to the category mapping JSON file.

        Returns:
            CleanSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_source(source=source)
        return self

    def with_target(self, target: DatasetPassport) -> CleanSalesDataPipelineBuilder:
        """Set the pipeline target dataset passport.

        Args:
            target (DatasetPassport): Passport describing the pipeline's target dataset.

        Returns:
            CleanSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_target(target=target)
        return self

    def with_ingest_task(self, week_decode_table_filepath: str) -> CleanSalesDataPipelineBuilder:
        """Add the ingest task to the pipeline and configure its validator.

        Args:
            week_decode_table_filepath (str): Filepath to the week decode table used by the ingest task.

        Returns:
            CleanSalesDataPipelineBuilder: The builder instance.
        """
        # Create the validator
        validation = (
            ValidationBuilder()
            .reset()
            .with_missing_column_validator(required_columns=list(REQUIRED_COLUMNS_INGEST.keys()))
            .with_column_type_validator(column_types=REQUIRED_COLUMNS_INGEST)
            .with_non_negative_column_validator(columns=NON_NEGATIVE_COLUMNS_INGEST)
            .with_range_validator(column="profit", min_value=-100.00, max_value=100.00)
            .build()
        )
        week_decode_table = IOService.read(
            filepath=Path(week_decode_table_filepath), try_parse_dates=True
        )
        logger.debug("Week decode table loaded for ingest task.")
        logger.debug(week_decode_table.head())

        task = IngestSalesDataTask(validation=validation, week_decode_table=week_decode_table)
        self._pipeline.add_task(task=task)
        return self

    def with_clean_task(self) -> CleanSalesDataPipelineBuilder:
        """Add the clean task to the pipeline and configure its validator.

        Returns:
            CleanSalesDataPipelineBuilder: The builder instance.
        """
        task = CleanSalesDataTask()
        self._pipeline.add_task(task=task)
        return self

    def with_full_year_filter_task(self, min_weeks: int = 50) -> CleanSalesDataPipelineBuilder:
        task = FilterPartialYearsTask(min_weeks=min_weeks)
        self._pipeline.add_task(task=task)
        return self

    def build(
        self,
    ) -> CleanSalesDataPipeline:
        """Finalize and return the built CleanSalesDataPipeline.

        Returns:
            CleanSalesDataPipeline: The constructed pipeline instance.
        """
        pipeline = self._pipeline
        self.reset()
        return pipeline
