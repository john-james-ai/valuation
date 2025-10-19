#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/app/dataprep/task.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Sunday October 19th 2025 06:44:14 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from typing import List, Optional, cast

from abc import abstractmethod
from dataclasses import dataclass, field

from loguru import logger
import pandas as pd

from valuation.app.base.task import Task, TaskConfig, TaskResult
from valuation.app.state import Status
from valuation.app.validation import Validation
from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetID, DatasetPassport
from valuation.infra.store.dataset import DatasetStore


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
    def _validate_result(self, result: TaskResult) -> TaskResult:
        """Validates the output data and updates the TaskResult.

        Subclasses must implement specific validation logic to ensure
        the output data meets expected standards. This method should
        update the `validation` attribute of the provided `TaskResult`
        object.

        Args:
            result: The TaskResult object containing the output data
                to be validated.

        Returns:
            The updated TaskResult object with validation results.
        """
        pass

    @abstractmethod
    def run(self, dataset: Dataset, force: bool = False) -> Dataset:
        """Executes the full task lifecycle: execution, validation, and reporting.

        This method orchestrates the task's operation within a context that
        captures timing, status, and validation results.

        Args:
            dataset (Dataset): The input dataset to be processed.
            force (bool): If True, forces re-execution even if output exists.

        Returns:
            Dataset: The processed output dataset, or None if skipped.
        """

    def _validate_columns(
        self, validation: Validation, data: pd.DataFrame, required_columns: List[str]
    ) -> Validation:
        """Validates that required columns are present and of correct types.

        Args:
            validation (Validation): The current validation object to update.
            data (pd.DataFrame): The DataFrame to validate.
            required_columns (List[str]): List of required column names.

        Returns:
            Validation: The updated validation object."""
        for col in required_columns:
            if col not in data.columns:
                validation.add_message(f"Missing required column: '{col}'.")

            else:
                dtype = str(data[col].dtype)
                if not dtype == DTYPES[col]:
                    validation.add_message(
                        f"Column '{col}' of type {dtype} should be type {DTYPES[col]}."
                    )
        return validation

    def _handle_validation_failure(self, validation: Validation) -> None:
        """Handles logging and raises an exception on validation failure.

        This method centralizes the failure logic. It logs all specific
        validation messages and then raises a `RuntimeError` to halt execution.

        Args:
            validation: The Validation object containing failure messages.

        Raises:
            RuntimeError: Always raised to ensure execution is halted and
                the failure is propagated.
        """

        msg = f"{self.__class__.__name__} - Validation Failed"
        logger.error(msg)
        logger.error(f"Validation Messages:\n{validation.messages}")
        raise RuntimeError(msg)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class SISODataPrepTaskConfig(TaskConfig):
    """Base configuration class for tasks."""

    source: DatasetPassport
    target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
class SISODataPrepTask(DataPrepTask):
    def __init__(
        self,
        config: SISODataPrepTaskConfig,
        dataset_store: type[DatasetStore] = DatasetStore,
    ) -> None:
        self._config = config
        self._dataset_store = dataset_store()

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
    def _validate_result(self, result: TaskResult) -> TaskResult:
        """Validates the output data and updates the TaskResult.

        Subclasses must implement specific validation logic to ensure
        the output data meets expected standards. This method should
        update the `validation` attribute of the provided `TaskResult`
        object.

        Args:
            result: The TaskResult object containing the output data
                to be validated.

        Returns:
            The updated TaskResult object with validation results.
        """
        pass

    def run(self, dataset: Dataset, force: bool = False) -> Optional[Dataset]:

        # Initialize the result object and start the task
        result = DataPrepTaskResult(task_name=self.task_name, config=self._config)
        result.start_task()

        # Check if output already exists to potentially skip processing.
        dataset_id_out = DatasetID.from_passport(self._config.target)
        if self._dataset_store.exists(dataset_id=dataset_id_out) and not force:
            dataset_out = self._dataset_store.get(dataset_id=dataset_id_out)
            dataset_out = cast(Dataset, dataset_out)
            result.status = Status.EXISTS.value
            result.end_task()
            logger.info(result)
            return dataset_out
        try:

            # 1. Capture the size of the input dataset.
            result.records_in = cast(int, dataset.nrows)

            # 2. Execute the task
            df_out = self._execute(df=dataset.data)

            # 3. Create the output dataset object and count output records.
            result.dataset = Dataset(passport=self._config.target, df=df_out)
            result.records_out = cast(int, result.dataset.nrows)

            # 4. Validate the result
            result = self._validate_result(result=result)
            result = cast(DataPrepTaskResult, result)

            # Store data if valid otherwise handle failure
            if result.validation.is_valid:
                result.status = Status.SUCCESS.value
                self._dataset_store.add(dataset=result.dataset)
            else:
                result.status = Status.FAILURE.value
                self._handle_validation_failure(validation=result.validation)

        finally:
            result.end_task()
            logger.info(result)
            result = cast(DataPrepTaskResult, result)
            return result.dataset
