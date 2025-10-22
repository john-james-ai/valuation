#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/pipeline.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Wednesday October 22nd 2025 11:50:57 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Dict, Optional, Union

from dataclasses import dataclass
from pathlib import Path

from loguru import logger
import pandas as pd
from tqdm import tqdm

from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.entity import Entity
from valuation.core.stage import DatasetStage
from valuation.core.state import Status
from valuation.flow.dataprep.pipeline import (
    DataPrepPipeline,
    DataPrepPipelineBuilder,
    DataPrepPipelineConfig,
    DataPrepPipelineResult,
)
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
    NON_NEGATIVE_COLUMNS_INGEST,
    REQUIRED_COLUMNS_INGEST,
    IngestSalesDataTask,
)
from valuation.flow.validation import ValidationBuilder
from valuation.infra.file.io import IOService
from valuation.infra.store.dataset import DatasetStore

# ------------------------------------------------------------------------------------------------ #
CONFIG_CATEGORY_INFO_KEY = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SalesDataPrepPipelineConfig(DataPrepPipelineConfig):
    """Holds all parameters for the sales data preparation pipeline.

    Attributes:
        (inherited) DataPrepPipelineConfig fields defining pipeline-level configuration.
    """


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SalesDataPrepPipelineResult(DataPrepPipelineResult):
    """Holds the results of a sales pipeline execution.

    Attributes:
        (inherited) DataPrepPipelineResult fields containing execution metadata and outputs.
    """


# ------------------------------------------------------------------------------------------------ #
class SalesDataPrepPipeline(DataPrepPipeline):
    """Pipeline implementation for preparing sales datasets.

    Args:
        dataset_store (type[DatasetStore]): The DatasetStore class or factory used by the pipeline.
        result (type[SalesDataPrepPipelineResult]): Result class used to create result instances.
    """

    _dataset_store: DatasetStore
    _result: SalesDataPrepPipelineResult

    def __init__(
        self,
        dataset_store: type[DatasetStore] = DatasetStore,
        result: type[SalesDataPrepPipelineResult] = SalesDataPrepPipelineResult,
    ) -> None:
        super().__init__(dataset_store=dataset_store, result=result)
        self._source = None
        self._target = None

    def run(self, force: bool = False) -> Optional[SalesDataPrepPipelineResult]:
        """Execute the configured pipeline over source categories and produce a dataset.

        Args:
            force (bool): If True, force reprocessing even when target dataset already exists. Defaults to False.

        Returns:
            Optional[SalesDataPrepPipelineResult]: Pipeline result object if execution completes or is skipped.
        """
        self._result.status_obj = Status.RUNNING

        try:
            if self._target is not None:
                if self._dataset_store.exists(dataset_id=self._target.id) and not force:
                    logger.info(
                        f"Dataset {self._target.label} already exists in the datastore. \nSkipping processing."
                    )
                    dataset = self._dataset_store.get(passport=self._target)
                    self._result.dataset = dataset
                    self._result.status_obj = Status.SKIPPED
                    self._result.end_pipeline()
                    return self._result

            # Read category filenames mapping
            category_filenames = IOService.read(filepath=Path(self._source))[
                CONFIG_CATEGORY_INFO_KEY
            ]
            logger.debug(f"Category filenames mapping loaded: {len(category_filenames)}")

            # Get the stage and entity from the target passport
            directory = self._dataset_store.file_system.get_stage_entity_location(
                stage=DatasetStage.RAW, entity=Entity.SALES
            )
            logger.debug(f"Raw sales data directory: {directory}\n")

            # Create tqdm progress bar for categories
            pbar = tqdm(
                category_filenames.items(),
                total=len(category_filenames),
                desc="Processing Sales Data by Category",
                unit="category",
            )

            df_concat = pd.DataFrame()
            # Iterate through category sales files
            for _, category_info in pbar:
                filename = category_info["filename"]
                filepath = directory / filename
                filepath = Path(filepath)
                category = category_info["category"]
                pbar.set_description(f"Processing category: {category} from file: {filename}")

                # Load the data
                df = self._load(filepath=filepath)

                for task in self._tasks:

                    # Run the task
                    df = task.run(df=df, category=category)

                    # Validate the result
                    task.validate(df=df)

                    # Update metrics
                    self._update_metrics(validation=task.validation)

                    # Check validation status
                    if not task.validation.is_valid:
                        self._result.status_obj = Status.FAIL
                        task.validation.report()
                        break

                    # Append to cumulative dataframe
                    df_concat = pd.concat([df_concat, df], ignore_index=True)

            # Create final dataset
            dataset = Dataset(passport=self._config.target, df=df_concat)

            # Add dataset to result and end it
            self._result.dataset = dataset
            self._result.end_pipeline()

            # Add final dataset to the dataset store.
            self._dataset_store.add(dataset=dataset, overwrite=force)

            logger.info(self._result)

            return self._result

        except Exception as e:
            self._result.status_obj = Status.FAIL
            logger.critical(f"Pipeline execution failed with exception: {e}")
            self._result.end_pipeline()
            raise e

    def _load(self, filepath: Path, **kwargs) -> Union[pd.DataFrame, Dict[str, str]]:
        """Load data from a filepath using the IO service and enforce expected dtypes when appropriate.

        Args:
            filepath (Path): Path to the file to be loaded.
            **kwargs: Additional keyword arguments forwarded to IOService.read.

        Returns:
            Union[pd.DataFrame, Dict[str, str]]: The loaded DataFrame or other data object.
        """

        try:

            data = IOService.read(filepath=filepath, **kwargs)
            # Ensure correct data types
            if isinstance(data, pd.DataFrame):
                logger.debug(f"Applying data types to loaded DataFrame")
                data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
            else:
                logger.debug(
                    f"Loaded data is type {type(data)} and not a DataFrame. Skipping dtype application."
                )
            return data
        except Exception as e:
            logger.critical(f"Failed to load data from {filepath.name} with exception: {e}")
            raise e


# ------------------------------------------------------------------------------------------------ #
class SalesDataPrepPipelineBuilder(DataPrepPipelineBuilder):
    """Builder for SalesDataPrepPipeline instances.

    Use the fluent `with_*` methods to configure source, target and tasks, then call `build()`.

    Attributes:
        _pipeline (SalesDataPrepPipeline): The pipeline instance being built.
    """

    _pipeline: SalesDataPrepPipeline

    def __init__(self) -> None:
        super().__init__()
        self.reset

    def reset(self) -> None:
        """Reset the internal builder state and create a fresh pipeline instance.

        Returns:
            None
        """
        self._pipeline = SalesDataPrepPipeline()

    def with_source(self, source: str) -> SalesDataPrepPipelineBuilder:
        """Set the pipeline source (category mapping file path).

        Args:
            source (str): Path to the category mapping JSON file.

        Returns:
            SalesDataPrepPipelineBuilder: The builder instance.
        """
        self._pipeline.add_source(source=source)
        return self

    def with_target(self, target: DatasetPassport) -> SalesDataPrepPipelineBuilder:
        """Set the pipeline target dataset passport.

        Args:
            target (DatasetPassport): Passport describing the pipeline's target dataset.

        Returns:
            SalesDataPrepPipelineBuilder: The builder instance.
        """
        self._pipeline.add_target(target=target)
        return self

    def with_ingest_task(self, week_decode_table_filepath: str) -> SalesDataPrepPipelineBuilder:
        """Add the ingest task to the pipeline and configure its validator.

        Args:
            week_decode_table_filepath (str): Filepath to the week decode table used by the ingest task.

        Returns:
            SalesDataPrepPipelineBuilder: The builder instance.
        """
        # Create the validator
        validation = (
            ValidationBuilder()
            .with_missing_column_validator(required_columns=list(REQUIRED_COLUMNS_INGEST.keys()))
            .with_column_type_validator(column_types=REQUIRED_COLUMNS_INGEST)
            .with_non_negative_column_validator(columns=NON_NEGATIVE_COLUMNS_INGEST)
            .with_range_validator(column="profit", min_value=100.00, max_value=100.00)
            .build()
        )
        week_decode_table = IOService.read(filepath=Path(week_decode_table_filepath))
        task = IngestSalesDataTask(validation=validation, week_decode_table=week_decode_table)
        self._pipeline.add_task(task=task)
        return self

    def with_clean_task(self) -> SalesDataPrepPipelineBuilder:
        """Add the clean task to the pipeline and configure its validator.

        Returns:
            SalesDataPrepPipelineBuilder: The builder instance.
        """
        # Create the validator
        validation = (
            ValidationBuilder()
            .with_missing_column_validator(required_columns=list(REQUIRED_COLUMNS_CLEAN.keys()))
            .with_column_type_validator(column_types=REQUIRED_COLUMNS_CLEAN)
            .with_non_negative_column_validator(columns=NON_NEGATIVE_COLUMNS_CLEAN)
            .build()
        )
        task = CleanSalesDataTask(validation=validation)
        self._pipeline.add_task(task=task)
        return self

    def with_aggregate_task(self) -> SalesDataPrepPipelineBuilder:
        """Add the aggregate task to the pipeline and configure its validator.

        Returns:
            SalesDataPrepPipelineBuilder: The builder instance.
        """
        # Create the validator
        validation = (
            ValidationBuilder()
            .with_missing_column_validator(required_columns=list(REQUIRED_COLUMNS_AGGREGATE.keys()))
            .with_column_type_validator(column_types=REQUIRED_COLUMNS_AGGREGATE)
            .with_non_negative_column_validator(columns=NON_NEGATIVE_COLUMNS_AGGREGATE)
            .build()
        )
        task = AggregateSalesDataTask(validation=validation)
        self._pipeline.add_task(task=task)
        return self

    def build(
        self,
    ) -> SalesDataPrepPipeline:
        """Finalize and return the built SalesDataPrepPipeline.

        Returns:
            SalesDataPrepPipeline: The constructed pipeline instance.
        """
        pipeline = self._pipeline
        self.reset()
        return pipeline
