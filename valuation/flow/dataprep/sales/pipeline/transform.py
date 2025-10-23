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
# Modified   : Wednesday October 22nd 2025 08:37:07 pm                                             #
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
import pandas as pd

from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.pipeline import PipelineResult
from valuation.flow.dataprep.pipeline import DataPrepPipeline, DataPrepPipelineBuilder
from valuation.flow.dataprep.sales.task.aggregate import AggregateSalesDataTask
from valuation.flow.dataprep.sales.task.filter import FilterPartialYearsTask
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

            self._result.records_in = df.shape[0]

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

    def _update_metrics(self, df: pd.DataFrame) -> None:
        """Updates pipeline result metrics based on the provided DataFrame.
        Args:
            df (pd.DataFrame): The DataFrame to analyze for metrics.
        Returns:
            None
        """

        self._result.records_out = df.shape[0]
        if self._result.records_in > 0:
            self._result.pct_change = (
                (self._result.records_in - self._result.records_out) / self._result.records_in
            ) * 100.0
        else:
            self._result.pct_change = 0.0

        self._result.stores = df["store"].nunique()
        self._result.categories = df["category"].nunique()

        years = df["year"].unique()
        if years.size > 0:
            self._result.first_year = int(years.min())
            self._result.last_year = int(years.max())
            self._result.num_years = int(years.size)
        else:
            self._result.first_year = None
            self._result.last_year = None
            self._result.num_years = 0

        self._result.weeks = df["week"].nunique()

    def _load(self, filepath: Path, **kwargs) -> pd.DataFrame:
        """Load and concatenate all parquet files in a directory into a single DataFrame.

        Args:
            directory (Path): Path to the directory containing parquet files.
            **kwargs: Additional keyword arguments forwarded to read_parquet.

        Returns:
            pd.DataFrame: Concatenated DataFrame containing data from all parquet files.
        """
        try:
            df = pd.read_parquet(filepath, **kwargs)
            df = df.astype({k: v for k, v in DTYPES.items() if k in df.columns})
            return df
        except Exception as e:
            logger.exception(f"Unable to read parquet directory {filepath} directly: {e}")
            raise e


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
