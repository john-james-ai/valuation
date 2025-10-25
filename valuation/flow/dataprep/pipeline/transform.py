#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/pipeline/transform.py                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Saturday October 25th 2025 11:04:43 am                                              #
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

from valuation.asset.dataset import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.pipeline import PipelineResult
from valuation.flow.dataprep.base.pipeline import DataPrepPipeline, DataPrepPipelineBuilder
from valuation.flow.dataprep.task.aggregate import AggregateSalesDataTask
from valuation.flow.dataprep.task.filter import FilterPartialYearsTask
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TransformSalesDataPipelineResult(PipelineResult):
    """Holds the results of a sales pipeline execution.

    Attributes:
        (inherited) DataPrepPipelineResult fields containing execution metadata and outputs.
    """

    dataset: Optional[Dataset] = None

    records_in: int = 0
    records_out: int = 0
    pct_change: float = 0.0

    stores: int = 0
    categories: int = 0

    first_year: Optional[int] = None
    last_year: Optional[int] = None
    num_years: int = 0
    weeks: int = 0

    def end_pipeline(self) -> None:
        super().end_pipeline()
        self.pct_change = ((self.records_in - self.records_out) / self.records_in) * 100.0


# ------------------------------------------------------------------------------------------------ #
class TransformSalesDataPipeline(DataPrepPipeline):
    """
    Pipeline implementation for transforming sales datasets using Polars.

    Args:
        dataset_store: The DatasetStore class or factory used by the pipeline.
        result: Result class used to create result instances.
    """

    _dataset_store: DatasetStore
    _result: TransformSalesDataPipelineResult

    def __init__(
        self,
        dataset_store: type[DatasetStore] = DatasetStore,
        result: type[TransformSalesDataPipelineResult] = TransformSalesDataPipelineResult,
    ) -> None:
        super().__init__(dataset_store=dataset_store)
        self._result = result(name=self.__class__.__name__)
        self._source = None
        self._target = None

    def run(self, force: bool = False) -> Optional[TransformSalesDataPipelineResult]:
        """
        Execute the transformation pipeline.

        Args:
            force: If True, force reprocessing even when target dataset exists.

        Returns:
            Optional[TransformSalesDataPipelineResult]: Pipeline result object.
        """
        self._result.start_pipeline()

        try:
            # Check if target exists and skip if not forcing
            if self._target is not None:
                if self._dataset_store.exists(dataset_id=self._target.id) and not force:
                    logger.info(
                        f"Dataset {self._target.label} already exists in the datastore.\n"
                        "Skipping processing."
                    )
                    dataset = self._dataset_store.get(passport=self._target)
                    self._result.dataset = dataset  # type: ignore
                    self._result.status_obj = Status.SKIPPED
                    self._result.end_pipeline()
                    return self._result

            # Delete the target if it exists
            self._dataset_store.remove(passport=self._target)

            # Load data using lazy evaluation
            logger.debug(f"Loading source data from {self._source}")
            df = self._load(filepath=Path(self._source), **self._target.read_kwargs)  # type: ignore

            # Initial metrics (collect only row count for efficiency)
            logger.debug("Calculating initial dataset metrics.")
            self._result.records_in = df.select(pl.len()).collect().item()

            # Process through tasks
            for task in self._tasks:
                logger.info(f"Running task: {task.__class__.__name__}")

                # Run the task (keep as LazyFrame)
                df = task.run(df=df, lazy=True)

                # Collect for validation
                df_collected = df.collect()

                # Validate the result
                task.validation.validate(df=df_collected, classname=task.__class__.__name__)

                # Update metrics
                self._update_metrics(df=df_collected)

                # Check validation status
                if not task.validation.is_valid:
                    self._result.status_obj = Status.FAIL
                    task.validation.report()
                    break

                # Convert back to lazy for next task
                df = df_collected.lazy()

            # Collect final result
            logger.debug("Collecting final Polars DataFrame.")
            df_final = df.collect()

            # Create the target dataset (store as Polars DataFrame)
            dataset = Dataset(passport=self._target, df=df_final)

            # Persist the dataset to the store
            logger.debug(f"Saving dataset {self._target.label} to the dataset store.")  # type: ignore
            self._dataset_store.add(dataset=dataset, overwrite=force)

            # Save the dataset in the result
            self._result.dataset = dataset
            self._result.status_obj = Status.SUCCESS

            # Finalize the result and log it
            self._result.end_pipeline()
            logger.info(self._result)

            return self._result

        except Exception as e:
            self._result.status_obj = Status.FAIL
            logger.critical(f"Pipeline execution failed with exception: {e}")
            self._result.end_pipeline()
            raise e

    def _update_metrics(self, df: pl.DataFrame) -> None:
        """
        Update pipeline result metrics based on the provided DataFrame.

        Args:
            df: The DataFrame to analyze for metrics.
        """
        # Get all metrics in one efficient aggregation pass
        metrics = df.select(
            [
                pl.len().alias("records_out"),
                pl.col("store").n_unique().alias("stores"),
                pl.col("category").n_unique().alias("categories"),
                pl.col("week").n_unique().alias("weeks"),
                pl.col("year").n_unique().alias("num_years"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
            ]
        )

        # Extract all values at once
        row = metrics.row(0, named=True)

        # Assign to result
        self._result.records_out = row["records_out"]
        self._result.stores = row["stores"]
        self._result.categories = row["categories"]
        self._result.weeks = row["weeks"]
        self._result.num_years = row["num_years"]

        # Handle year values (may be None if no data)
        if self._result.num_years > 0:
            self._result.first_year = row["first_year"]
            self._result.last_year = row["last_year"]
        else:
            self._result.first_year = None
            self._result.last_year = None

    def _load(self, filepath: Path, **kwargs) -> pl.LazyFrame:
        """
        Load sales data from the specified filepath into a Polars LazyFrame.

        Args:
            filepath: Path to the sales data file or directory.
            **kwargs: Additional keyword arguments for data loading.

        Returns:
            pl.LazyFrame: Loaded sales data as a Polars LazyFrame.
        """
        logger.debug(f"Loading sales data from {filepath} with Polars.")

        # Scan all parquet files in the directory
        directory_path = str(filepath)
        df = pl.scan_parquet(f"{directory_path}/*.parquet")

        # Get DataFrame columns
        df_cols = df.collect_schema().names()

        # Filter DTYPES dict to only columns present in the data
        casts_to_apply = {k: v for k, v in DTYPES.items() if k in df_cols}

        # Apply casts if any
        if casts_to_apply:
            df = df.cast(casts_to_apply)

        return df


# ------------------------------------------------------------------------------------------------ #
class TransformSalesDataPipelineBuilder(DataPrepPipelineBuilder):
    """
    Builder for TransformSalesDataPipeline instances.

    Use the fluent `with_*` methods to configure source, target and tasks, then call `build()`.

    Attributes:
        _pipeline: The pipeline instance being built.
    """

    _pipeline: TransformSalesDataPipeline

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """
        Reset the internal builder state and create a fresh pipeline instance.
        """
        self._pipeline = TransformSalesDataPipeline()

    def with_source(self, source: str) -> "TransformSalesDataPipelineBuilder":
        """
        Set the pipeline source (category mapping file path).

        Args:
            source: Path to the category mapping JSON file.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_source(source=source)
        return self

    def with_target(self, target: DatasetPassport) -> "TransformSalesDataPipelineBuilder":
        """
        Set the pipeline target dataset passport.

        Args:
            target: Passport describing the pipeline's target dataset.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_target(target=target)
        return self

    def with_aggregate_task(self) -> "TransformSalesDataPipelineBuilder":
        """
        Add the aggregate task to the pipeline and configure its validator.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        task = AggregateSalesDataTask()
        self._pipeline.add_task(task=task)
        return self

    def with_full_year_filter_task(
        self, min_weeks: int = 50
    ) -> "TransformSalesDataPipelineBuilder":
        """
        Add the filter partial years task to the pipeline and configure its validator.

        Args:
            min_weeks: Minimum number of weeks required for a year to be considered full.
                Defaults to 50.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        task = FilterPartialYearsTask(min_weeks=min_weeks)
        self._pipeline.add_task(task=task)
        return self

    def build(self) -> TransformSalesDataPipeline:
        """
        Finalize and return the built TransformSalesDataPipeline.

        Returns:
            TransformSalesDataPipeline: The constructed pipeline instance.
        """
        pipeline = self._pipeline
        self.reset()
        return pipeline
