#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/sales/pipeline/transform.py                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Wednesday October 22nd 2025 10:08:47 pm                                             #
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

from valuation.asset.dataset.base import DTYPES_PL, Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.pipeline import PipelineResult
from valuation.flow.dataprep.pipeline import DataPrepPipeline, DataPrepPipelineBuilder
from valuation.flow.dataprep.sales.task.agg_pl import AggregateSalesDataTask
from valuation.flow.dataprep.sales.task.filter_pl import FilterPartialYearsTask
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
    """Pipeline implementation for preparing sales datasets.

    Args:
        dataset_store (type[DatasetStore]): The DatasetStore class or factory used by the pipeline.
        result (type[TransformSalesDataPipelineResult]): Result class used to create result instances.
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

            # Load data using same kwargs as target
            logger.info(f"Loading source data from {self._source}")
            df = self._load(filepath=Path(self._source), **self._target.read_kwargs)  # type: ignore

            # Initial metrics
            logger.info("Calculating initial dataset metrics.")
            self._result.records_in = df.select(pl.count()).collect().item(0, 0)

            for task in self._tasks:

                logger.info(f"Running task: {task.__class__.__name__}")
                # Run the task
                df = task.run(df=df)

                # Validate the result
                task.validation.validate(df=df, classname=task.__class__.__name__)

                # Update metrics
                self._update_metrics(df=df)

                # Check validation status
                if not task.validation.is_valid:
                    self._result.status_obj = Status.FAIL
                    task.validation.report()
                    break

            # Convert the polars DataFrame to a pandas DataFrame for storage
            logger.info("Converting Polars DataFrame to Pandas DataFrame for storage.")
            df = df.to_pandas()  # type: ignore
            # Create the target dataset
            dataset = Dataset(passport=self._target, df=df)
            # Persist the dataset to the store
            logger.info(f"Saving dataset {self._target.label} to the dataset store.")  # type: ignore
            self._dataset_store.add(dataset=dataset, overwrite=force)

            # Save the dataset in the result
            self._result.dataset = dataset
            self._result.status_obj = Status.SUCCESS

            # Finalize the result and log it.
            self._result.end_pipeline()
            logger.info(self._result)

            return self._result

        except Exception as e:
            self._result.status_obj = Status.FAIL
            logger.critical(f"Pipeline execution failed with exception: {e}")
            self._result.end_pipeline()
            raise e

    def _update_metrics(self, df: pl.DataFrame) -> None:
        """Updates pipeline result metrics based on the provided DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to analyze for metrics.
        Returns:
            None
        """

        # 1. Get row count (Polars uses .height)
        self._result.records_out = df.height

        # 2. Get unique counts (Polars uses .n_unique())
        self._result.stores = df.get_column("store").n_unique()
        self._result.categories = df.get_column("category").n_unique()
        self._result.weeks = df.get_column("week").n_unique()

        # 3. Get all year stats in one efficient pass
        #    Polars aggregations (min, max, n_unique) run in parallel
        year_stats = df.select(
            [
                pl.col("year").n_unique().alias("num_years"),
                pl.col("year").min().alias("first_year"),
                pl.col("year").max().alias("last_year"),
            ]
        )

        # 4. Assign the results
        #    year_stats is a 1-row DataFrame. We extract the values.
        num_years = year_stats.item(0, "num_years")
        first_year = year_stats.item(0, "first_year")
        last_year = year_stats.item(0, "last_year")

        self._result.num_years = int(num_years)

        if self._result.num_years > 0:
            # We cast to int() here, as Polars might return None
            # if the column was all null, but int(None) fails.
            self._result.first_year = int(first_year) if first_year is not None else None
            self._result.last_year = int(last_year) if last_year is not None else None
        else:
            self._result.first_year = None
            self._result.last_year = None

    def _load(self, filepath: Path, **kwargs) -> pl.LazyFrame:
        """Loads sales data from the specified filepath into a Polars DataFrame.
        Args:
            filepath (Path): Path to the sales data file or directory.
            **kwargs: Additional keyword arguments for data loading.
        Returns:
            pl.DataFrame: Loaded sales data as a Polars DataFrame.
        """
        logger.debug(f"Loading sales data from {filepath} with Polars.")
        # Use Polars to read all parquet files in the directory
        directory_path = str(filepath)
        df = pl.scan_parquet(f"{directory_path}/*.parquet")

        # 2. Get your DataFrame's columns (using a set is faster)
        df_cols = df.collect_schema().names()  # Ensure schema is collected

        # 3. Filter the DTYPES dict, just like in your Pandas code
        casts_to_apply = {k: v for k, v in DTYPES_PL.items() if k in df_cols}

        # 4. Apply the casts
        df = df.cast(casts_to_apply)

        return df


# ------------------------------------------------------------------------------------------------ #
class TransformSalesDataPipelineBuilder(DataPrepPipelineBuilder):
    """Builder for TransformSalesDataPipeline instances.

    Use the fluent `with_*` methods to configure source, target and tasks, then call `build()`.

    Attributes:
        _pipeline (TransformSalesDataPipeline): The pipeline instance being built.
    """

    _pipeline: TransformSalesDataPipeline

    def __init__(self) -> None:
        super().__init__()
        self.reset()

    def reset(self) -> None:
        """Reset the internal builder state and create a fresh pipeline instance.

        Returns:
            None
        """
        self._pipeline = TransformSalesDataPipeline()

    def with_source(self, source: str) -> TransformSalesDataPipelineBuilder:
        """Set the pipeline source (category mapping file path).

        Args:
            source (str): Path to the category mapping JSON file.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_source(source=source)
        return self

    def with_target(self, target: DatasetPassport) -> TransformSalesDataPipelineBuilder:
        """Set the pipeline target dataset passport.

        Args:
            target (DatasetPassport): Passport describing the pipeline's target dataset.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """
        self._pipeline.add_target(target=target)
        return self

    def with_aggregate_task(self) -> TransformSalesDataPipelineBuilder:
        """Add the aggregate task to the pipeline and configure its validator.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """

        task = AggregateSalesDataTask()
        self._pipeline.add_task(task=task)
        return self

    def with_full_year_filter_task(self, min_weeks: int = 50) -> TransformSalesDataPipelineBuilder:
        """Add the filter partial years task to the pipeline and configure its validator.
        Args:
            min_weeks (int): Minimum number of weeks required for a year to be considered full.
                Defaults to 50.

        Returns:
            TransformSalesDataPipelineBuilder: The builder instance.
        """

        task = FilterPartialYearsTask(min_weeks=min_weeks)
        self._pipeline.add_task(task=task)
        return self

    def build(
        self,
    ) -> TransformSalesDataPipeline:
        """Finalize and return the built TransformSalesDataPipeline.

        Returns:
            TransformSalesDataPipeline: The constructed pipeline instance.
        """
        pipeline = self._pipeline
        self.reset()
        return pipeline
