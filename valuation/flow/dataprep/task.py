#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/task.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Tuesday October 21st 2025 06:43:22 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from typing import Optional, cast

from abc import abstractmethod
from dataclasses import dataclass, field

from loguru import logger
import pandas as pd

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.task import Task, TaskConfig, TaskResult
from valuation.flow.validation import Validation
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepTaskConfig(TaskConfig):
    """Base configuration class for tasks."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepTaskResult(TaskResult):

    dataset_name: Optional[str] = None

    # Record counts
    records_in: Optional[int] = None
    records_out: Optional[int] = None
    pct_change: Optional[float] = None

    # Validation results
    validation: Validation = field(default_factory=Validation)

    # Contains the output data from the task
    dataset: Dataset = field(default=None)

    def end_task(self) -> None:

        super().end_task()
        if self.records_in is None or self.records_out is None or self.records_in == 0:
            self.pct_change = None
        else:
            self.pct_change = round(
                ((self.records_in - self.records_out) / self.records_in) * 100, 2
            )


# ------------------------------------------------------------------------------------------------ #
class DataPrepTask(Task):
    def __init__(self, validation: Optional[Validation] = None) -> None:
        self._validation = validation if validation else Validation()

    @abstractmethod
    def _execute(self, df: pd.DataFrame, **kwargs) -> Dataset:
        """Executes the core logic of the task.

        Subclasses must implement this method to perform the specific
        data processing or transformation that the task is responsible for.

        Args:
            dataset (Dataset): The input dataset to be processed.

        Returns:
            Dataset: The processed output dataset.
        """

    @abstractmethod
    def run(self, dataset: Dataset, force: bool = False) -> DataPrepTaskResult:
        """Executes the full task lifecycle: execution, validation, and reporting.

        This method orchestrates the task's operation within a context that
        captures timing, status, and validation results.

        Args:
            dataset (Dataset): The input dataset to be processed.
            force (bool): If True, forces re-execution even if output exists.

        Returns:
            Dataset: The processed output dataset, or None if skipped.
        """


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SISODataPrepTaskConfig(DataPrepTaskConfig):
    """Base configuration class for tasks."""

    source: DatasetPassport
    target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
class SISODataPrepTask(DataPrepTask):
    def __init__(
        self,
        config: SISODataPrepTaskConfig,
        dataset_store: DatasetStore = DatasetStore,
        validation: Optional[Validation] = None,
    ) -> None:
        super().__init__(validation=validation)
        self._config = config
        self._dataset_store = dataset_store

    @property
    def config(self) -> SISODataPrepTaskConfig:
        """Return the task configuration.

        Returns:
            SISODataPrepTaskConfig: The task configuration.
        """
        return self._config

    @abstractmethod
    def _execute(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Executes the core logic of the task.

        Subclasses must implement this method to perform the specific
        data processing or transformation that the task is responsible for.

        Args:
            dataset (Dataset): The input dataset to be processed.

        Returns:
            pd.DataFrame: The processed output DataFrame.
        """

    def run(self, dataset: Dataset, force: bool = False) -> DataPrepTaskResult:

        # Initialize the result object and start the task
        result = DataPrepTaskResult(task_name=self.task_name, config=self._config)
        result.start_task()

        # Check if output already exists to potentially skip processing.
        if self._dataset_store.exists(dataset_id=self._config.target.id) and not force:
            dataset_out = self._dataset_store.get(passport=self._config.target)
            result.status_obj = Status.SKIPPED
            result.end_task()
            logger.info(result)
            result.dataset = dataset_out
            return result
        try:
            self._dataset_store.remove(passport=self._config.target)

            # 1. Capture the size of the input dataset.
            result.records_in = cast(int, dataset.nrows)

            # 2. Execute the task
            df_out = self._execute(df=dataset.data)

            # 3. Create the output dataset object and count output records.
            result.dataset = Dataset(passport=self._config.target, df=df_out)
            result.records_out = cast(int, result.dataset.nrows)

            # 4. Validate the result
            if not self._validation.validate(
                data=result.dataset.data,
                classname=self.__class__.__name__,
            ):
                result.status_obj = Status.FAIL
                raise ValueError("Data validation failed.")

        except Exception as e:
            result.status_obj = Status.FAIL
            logger.exception(f"An error occurred during task execution: \n{e}")

        finally:
            result.end_task()
            logger.info(result)
            return result
