#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/workflow/task.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Saturday October 18th 2025 08:20:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from types import TracebackType
from typing import Any, Dict, List, Optional, Type, Union, cast

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import time
import traceback

from loguru import logger
import pandas as pd

from valuation.asset.dataset import Dataset
from valuation.asset.identity import Passport
from valuation.config.data import DTYPES
from valuation.core.data import DataClass
from valuation.utils.db.dataset import DatasetStore
from valuation.utils.io.service import IOService
from valuation.workflow import Status
from valuation.workflow.validation import Validation


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskConfig(DataClass):
    """Base configuration class for tasks."""

    task_name: str
    dataset_name: str
    description: str
    source: Union[Passport, Dict[str, str]]
    target: Passport


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskResult(DataClass):
    """Holds the comprehensive results and metadata of a single task execution.

    This object serves as the primary return value from a `Task.run()` method.
    It encapsulates everything about the run: the configuration used, timing
    metrics, record counts, validation outcomes, final status, and the
    actual data produced.

    Attributes:
        task_name (str): The name of the task class that was executed.
        dataset_name (str): A human-readable name for the dataset being processed.
        config (TaskConfig): The configuration object used for this task run.
        started (Optional[datetime]): Timestamp marking the start of execution.
        ended (Optional[datetime]): Timestamp marking the end of execution.
        elapsed (Optional[float]): Total execution time in seconds.
        records_in (int): The number of records in the input dataset.
        records_out (int): The number of records in the output dataset.
        pct_change (Optional[float]): The percentage change between input and
            output record counts.
        validation (Optional[Validation]): An object containing the results of
            data quality checks.
        status (Optional[str]): The final status of the task (e.g., 'Success',
            'Failure', 'Critical').
        data (Optional[Union[pd.DataFrame, Any]]): The actual data artifact
            produced by the task's execution.
    """

    task_name: str
    dataset_name: str
    config: Optional[TaskConfig] = field(default=None)

    # Timestamps
    started: Optional[datetime] = field(default=None)
    ended: Optional[datetime] = field(default=None)
    elapsed: Optional[float] = field(default=0.0)

    # Record counts
    records_in: int = 0
    records_out: int = 0
    pct_change: float = 0.0

    # Validation results
    validation: Validation = field(default_factory=Validation)

    # Status
    status: Optional[str] = field(default=Status.PENDING.value)

    # Contains the output data from the task
    dataset: Dataset = field(default=None)

    @property
    def summary(self) -> Dict[str, Any]:
        """Generates a summary dictionary of key task result metrics.

        This method provides a concise overview of the task execution,
        suitable for logging or reporting purposes. It excludes large data
        artifacts and focuses on metadata and performance indicators.

        Returns:
            A dictionary containing key metrics and status information.
        """
        return {
            "task_name": self.task_name,
            "dataset_name": self.dataset_name,
            "status": self.status,
            "started": self.started,
            "ended": self.ended,
            "elapsed_seconds": self.elapsed,
            "records_in": self.records_in,
            "records_out": self.records_out,
            "pct_change": self.pct_change,
            "num_failures": self.validation.num_failures if self.validation else None,
            "is_valid": self.validation.is_valid if self.validation else None,
        }

    def finalize(self) -> None:
        """Calculates final derived metrics after the task has run.

        This method is called by the `TaskContext` to compute values that
        depend on the final state of the result, such as the percentage
        change in record count. It modifies the instance attributes in-place.
        """
        if self.records_in is None or self.records_out is None or self.records_in == 0:
            self.pct_change = 0
        else:
            self.pct_change = round(
                ((self.records_in - self.records_out) / self.records_in) * 100, 2
            )


# ------------------------------------------------------------------------------------------------ #
class TaskContext:
    """A context manager for executing and reporting on a single task run.

    This class acts as a "flight recorder" for a task. It handles the lifecycle
    of a `TaskResult` object, ensuring that the task's execution is timed,
    its outcome (success, failure, or critical error) is recorded, and a
    final report is logged, regardless of whether an exception occurs.

    Usage:
        with TaskContext(config) as result:
            # ... perform task logic and update the result object ...
    """

    def __init__(self, config: TaskConfig):
        """Initializes the context and the underlying result object.

        Args:
            config (TaskConfig): The configuration object for the task.
        """
        self._config = config
        self._result = TaskResult(
            task_name=config.task_name,
            dataset_name=config.dataset_name,
            config=config,
            validation=Validation(),
        )
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> TaskResult:
        """Marks the start of the task execution and provides the result object.

        This method is called upon entering the `with` block. It records the
        start time and returns the `TaskResult` object, which can be populated
        by the code within the block.

        Returns:
            TaskResult: The initialized result object for the current task run.
        """
        self._result.started = datetime.now()
        self._start_time = time.perf_counter()
        return self._result

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        exc_traceback: Optional[TracebackType],
    ) -> None:
        """Finalizes and logs the task result upon exiting the `with` block.

        This method is always called when the `with` block is exited. It
        calculates the elapsed time and determines the final status based on
        whether an exception occurred and the results of the task's own
        validation logic.

        Args:
            exc_type: The type of the exception raised, if any.
            exc_value: The exception instance raised, if any.
            traceback: A traceback object, if an exception occurred.
        """
        self._result.ended = datetime.now()
        self._end_time = time.perf_counter()
        if self._start_time:
            self._result.elapsed = self._end_time - self._start_time

        if exc_type is not None:
            # A critical, unexpected exception occurred during execution.
            self._result.status = Status.CRITICAL.value
            if self._result.validation:
                self._result.validation.add_message(str(exc_value))

            # Format the full traceback into a string for detailed debugging
            traceback_details = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)  # type: ignore
            )
            logger.error(
                f"Task {self._result.task_name} failed with an exception:\n{traceback_details}"
            )
            logger.error(f"Task {self._result.task_name} failed with exception: {exc_value}")
        else:
            # The task ran without crashing; check validation status.
            is_valid = self._result.validation and self._result.validation.is_valid
            self._result.status = Status.SUCCESS.value if is_valid else Status.FAILURE.value
            logger.info(
                f"Task {self._result.task_name} completed with status: {self._result.status}"
            )

        self._result.finalize()
        logger.info(self._result)


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for data processing tasks.

    This class defines the interface and common functionality for all
    data processing tasks within the workflow. Subclasses must implement
    the `_execute` and `_validate_result` methods to provide specific
    processing logic and validation rules.

    Args:
        config (TaskConfig): The configuration object for the task.
        io (type[IOService], optional): The IO service class to use for data
            loading and saving. Defaults to `IOService`.
        dataset_store (type[DatasetStore], optional): The asset store class to use
            for managing data entities. Defaults to `DatasetStore`.
    """

    def __init__(
        self,
        config: TaskConfig,
        io: type[IOService] = IOService,
        dataset_store: DatasetStore = DatasetStore,
    ) -> None:

        self._config = config
        self._io = io()
        self._dataset_store = dataset_store
        self._task_context = TaskContext(config=config)

    @property
    def config(self) -> TaskConfig:
        """Returns the task configuration object.

        Returns:
            TaskConfig: The configuration object used by this task.
        """
        return self._config

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

    def run(self, dataset: Dataset) -> TaskResult:
        """Executes the full task lifecycle: execution, validation, and reporting.

        This method orchestrates the task's operation within a context that
        handles timing, status updates, and error logging. It ensures that a
        complete TaskResult object is returned, whether the task succeeds or fails.

        Args:
            data: Optional[pd.DataFrame]: The input data to be processed by the task.

        Returns:
            TaskResult: An object containing the final status, metrics,
                validation info, and output data of the task run.

        Raises:
            RuntimeError: If input data is missing or empty, or if the
                validation fails.
        """

        try:
            with self._task_context as result:

                # 1. Capture the size of the input dataset.
                result.records_in = cast(int, dataset.nrows)

                # 2. Check validity of target configuration
                if not isinstance(self._config.target, Passport):
                    raise RuntimeError("Target configuration must be a Passport instance.")

                # 3. Check if output already exists to potentially skip processing.
                if self._dataset_store.exists(
                    name=self._config.target.name, stage=self._config.target.stage
                ):

                    result.status = Status.EXISTS.value
                    # Get the output dataset from the asset store
                    dataset_out = self._dataset_store.get(
                        name=self._config.target.name,
                        stage=self._config.target.stage,
                    )
                    # Cast to a dataset object and assign to result
                    dataset_out = cast(Dataset, dataset_out)
                    result.records_out = cast(int, dataset_out.nrows)
                    result.dataset = dataset_out
                    return result

                # 4. Otherwise execute the task
                df_out = self._execute(df=dataset.data)

                # 2. Create the output dataset object and count output records.
                dataset_out = Dataset(passport=self._config.target, df=df_out)
                result.records_out = cast(int, result.dataset.nrows)

                # Validate the result by calling the subclass's implementation.
                result = self._validate_result(result=result)

                # Handle validation failure.
                if not result.validation.is_valid:  # type: ignore
                    self._handle_validation_failure(validation=result.validation)
        finally:
            return self._finalize(result=result, dataset=dataset_out)

    def _finalize(self, result: TaskResult, dataset: Dataset) -> TaskResult:
        """Finalizes the TaskResult with output dataset and metrics.

        This method updates the TaskResult object with the final output
        dataset and recalculates derived metrics such as record counts
        and percentage change.

        Args:
            result (TaskResult): The TaskResult object to be finalized.
            dataset_out (Dataset): The output dataset produced by the task.

        Returns:
            TaskResult: The updated TaskResult object with final metrics.
        """
        passport = dataset.passport
        passport.complete(
            created=result.started, completed=result.ended, cost=result.elapsed or 0.0
        )
        dataset.stamp_passport(passport=passport)
        result.dataset = dataset
        result.records_out = cast(int, dataset.nrows)
        result.finalize()
        self._dataset_store.add(dataset=dataset)
        return result

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
