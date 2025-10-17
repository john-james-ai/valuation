#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/workflow/pipeline.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Friday October 17th 2025 06:34:18 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from types import TracebackType
import typing
from typing import Any, Dict, List, Optional, Type, Union

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import traceback

from loguru import logger
import pandas as pd

from valuation.config.data import DTYPES
from valuation.utils.data import DataClass
from valuation.utils.io.service import IOService
from valuation.workflow import Status
from valuation.workflow.task import TaskResult

# ------------------------------------------------------------------------------------------------ #
if typing.TYPE_CHECKING:
    from valuation.workflow.pipeline import PipelineResult


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PipelineConfig(DataClass):
    """Holds all parameters for the pipeline."""

    name: str
    dataset_name: str
    description: str
    input_location: Union[Path, str]
    output_location: Union[Path, str]


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PipelineResult(DataClass):
    """Holds the results of a pipeline execution."""

    name: str
    dataset_name: str
    description: str
    config: PipelineConfig = field(default=None)  # Optional until setup
    # Timestamps
    started: Optional[datetime] = field(default=None)  # Optional until setup
    ended: Optional[datetime] = field(default=None)  # Optional until teardown
    elapsed: Optional[float] = field(default=None)  # Optional until teardown
    # Result Metrics
    num_records: Optional[int] = field(default=None)  # Only known after execute
    num_fields: Optional[int] = field(default=None)  # Only known after execute
    memory_mb: Optional[float] = field(default=None)  # Only known after execute
    filesize_mb: Optional[float] = field(default=None)  # Only known after execute

    # State
    status: str = Status.PENDING.value
    # Pipeline task results
    task_results: List[TaskResult] = field(default_factory=list)

    # Data produced by the pipeline
    data: Optional[Union[pd.DataFrame, Any]] = field(default=None)

    def add_task_result(self, result: TaskResult) -> None:
        """Adds a task result to the pipeline result.

        Args:
            result (TaskResult): The result of a task execution.
        """
        if (
            result.validation.is_valid
            and self.status != Status.FAILURE.value
            and self.status != Status.CRITICAL.value
        ):
            self.status = Status.SUCCESS.value
        else:
            self.status = (
                Status.FAILURE.value
                if self.status != Status.FAILURE.value
                else Status.CRITICAL.value
            )

        self.task_results.append(result)

    def summary(self) -> None:
        results = [result.summary for result in self.task_results]
        print(pd.DataFrame(results))


# ------------------------------------------------------------------------------------------------ #
class PipelineContext:
    """Holds the context for a pipeline execution."""

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._result = PipelineResult(
            name=config.name,
            dataset_name=config.dataset_name,
            description=config.description,
            config=config,
        )

        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None

    def __enter__(self) -> PipelineResult:
        """Marks the start of the pipeline execution and provides the result object.

        This method is called upon entering the `with` block. It records the
        start time and returns the `PipelineResult` object, which can be populated
        by the code within the block.

        Returns:
            PipelineResult: The initialized result object for the current pipeline run.
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
        """Finalizes and logs the pipeline result upon exiting the `with` block.

        This method is always called when the `with` block is exited. It
        calculates the elapsed time and determines the final status based on
        whether an exception occurred and the results of the pipeline's own
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

            # Format the full traceback into a string for detailed debugging
            traceback_details = "".join(
                traceback.format_exception(exc_type, exc_value, exc_traceback)  # type: ignore
            )
            logger.error(
                f"Pipeline {self._result.name} failed with an \
                    exception:\n{traceback_details}"
            )
            logger.error(
                f"Pipeline {self._result.name} failed with \
                exception: {exc_value}"
            )
        else:
            # The pipeline ran without crashing; check validation status.
            is_valid = all(
                task_result.status == Status.SUCCESS.value
                for task_result in self._result.task_results
            )
            if is_valid:
                self._result.status = Status.SUCCESS.value
            else:
                self._result.status = Status.FAILURE.value

        logger.info(self._result)


# ------------------------------------------------------------------------------------------------ #
class Pipeline:
    """Pipeline class for managing data processing and modeling workflows.

    This class provides a structured way to define, execute, and manage
    data processing pipelines. It supports adding multiple tasks, executing
    them in sequence, and handling input/output operations.

    Args:
        config (PipelineConfig): Configuration parameters for the pipeline.
        io (IOService): Service for handling input/output operations.
    Attributes:
        _config (PipelineConfig): Configuration parameters for the pipeline.
        _io (IOService): Service for handling input/output operations.
        _pipeline_context (PipelineContext): Context manager for pipeline execution.
        _tasks (List[Task]): List of tasks to be executed in the pipeline.
    Methods:
        _execute(data, result): Executes the pipeline logic.
        add_task(task): Adds a task to the pipeline.
        run(force): Runs all tasks in the pipeline.
    Returns:
        PipelineResult: The result of the pipeline execution.

    """

    def __init__(self, config: PipelineConfig, io: type[IOService] = IOService):
        self._config = config
        self._io = io
        self._pipeline_context = PipelineContext(config=config)
        self._tasks = []

    def _execute(
        self, data: Union[pd.DataFrame, Any], pipeline_result: PipelineResult
    ) -> PipelineResult:
        """ "Executes the pipeline logic.
        Args:
            data (pd.DataFrame): The input data for the pipeline.
            result (PipelineResult): The result object to populate during execution.

        Returns:
            PipelineResult: The updated result object after execution."""
        for task in self._tasks:
            task_result = task.run(data=data)
            pipeline_result.add_task_result(task_result)
            if not task_result.validation.is_valid:
                logger.error(f"Task {task.__class__.__name__} failed validation.")
                break
            data = task_result.data

        return pipeline_result

    def add_task(self, task) -> Pipeline:
        """Adds a task to the pipeline.

        Args:
            task: An instance of a data preparation or modeling task.

        Returns:
            Pipeline: The current instance of the pipeline (for method chaining).
        """
        self._tasks.append(task)
        return self

    def run(self, force: bool = False) -> PipelineResult:
        """Runs all tasks in the pipeline.

        Args:
            force (bool): If True, forces re-execution of all tasks.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        try:
            with self._pipeline_context as pipeline_result:
                # Check if output already exists
                if self._output_exists(force=force):
                    pipeline_result.status = Status.EXISTS.value
                    # Load existing output data
                    pipeline_result.data = self._load(filepath=Path(self._config.output_location))
                else:
                    # Obtain the input data for the pipeline
                    input_data = self._initialize()
                    # Execute the pipeline
                    pipeline_result = self._execute(
                        data=input_data, pipeline_result=pipeline_result
                    )
                    # Finalize the pipeline
                    pipeline_result = self._finalize(result=pipeline_result)
        except Exception as e:
            logger.critical(f"Pipeline {self._config.name} failed with exception: {e}")
            pipeline_result.status = Status.FAILURE.value
            raise e
        finally:
            return pipeline_result

    def _initialize(self) -> Union[pd.DataFrame, Dict[str, str]]:
        """Initializes the pipeline before execution.

        Obtains the data required for the pipeline from the input location.

        Returns:
            Union[pd.DataFrame, Dict[str,str]]: The initialized data for the pipeline.
        """
        logger.debug(f"{self.__class__.__name__} - Initializing")
        # Load input data
        data = self._load(filepath=Path(self._config.input_location))
        return data

    def _finalize(self, result: PipelineResult) -> PipelineResult:
        """Finalizes the pipeline after execution.

        Saves the output data and updates the result metrics.

        Args:
            result (PipelineResult): The result of the pipeline execution.

        Returns:
            PipelineResult: The finalized result of the pipeline execution.
        """

        logger.debug(f"{self.__class__.__name__} - Finalizing")
        if result.status == Status.SUCCESS.value and result.data is not None:
            # Save output data
            self._save(
                df=result.data,
                filepath=Path(self._config.output_location),
            )

            result.num_records = len(result.data) if isinstance(result.data, pd.DataFrame) else 0
            result.num_fields = (
                len(result.data.columns) if isinstance(result.data, pd.DataFrame) else 0
            )
            result.memory_mb = (
                result.data.memory_usage(deep=True).sum() / (1024 * 1024)
                if isinstance(result.data, pd.DataFrame)
                else 0
            )
            result.filesize_mb = Path(self._config.output_location).stat().st_size / (
                1024 * 1024
            )  # in MB
        return result

    def _load(self, filepath: Path, **kwargs) -> Union[pd.DataFrame, Dict[str, str]]:
        """Loads a DataFrame from the specified filepath using the I/O service.

        Args:
            filepath: The path to the file to be loaded.
            **kwargs: Additional keyword arguments for the I/O service.

            Returns:
            Union[pd.DataFrame, Any]: The loaded DataFrame or data object."""

        logger.debug(f"Loading data from {filepath}")

        data = self._io.read(filepath=filepath, **kwargs)
        # Ensure correct data types
        if isinstance(data, pd.DataFrame):
            logger.debug(f"Applying data types to loaded DataFrame")
            data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
        else:
            logger.debug(
                f"Loaded data is type {type(data)} and not a DataFrame. Skipping dtype application."
            )
        return data

    def _save(self, df: pd.DataFrame, filepath: Path, **kwargs) -> None:
        """
        Saves a DataFrame to the processed data directory using the I/O service.

        Args:
            df: The DataFrame to save.
            filepath: The path to the file to be saved.
        """
        logger.debug(f"Saving data to {filepath}")
        self._io.write(data=df, filepath=filepath, **kwargs)

    def _delete(self, location: Path) -> None:
        """
        Deletes a file from the specified location.

        Args:
            location: The path of the file to delete.
        """
        location.unlink(missing_ok=True)

    def _exists(self, location: Path) -> bool:
        """
        Checks if a file exists at the specified location.

        Args:
            location: The path to a file for the existence check.

        Returns:
            True if the file exists, False otherwise.
        """
        return location.exists()

    def _output_exists(self, force: bool = False) -> bool:
        """
        Determines whether the pipeline should be skipped because the output file already exists.

        If `force` is True, the existing file is deleted, and the pipeline proceeds.

        Args:
            force: If True, forces the pipeline to run by deleting existing output.

        Returns:
            True if the pipeline should be skipped (i.e., output file exists and force is False),
            False otherwise.
        """
        if force:
            self._delete(location=self._config.output_location)
            output_exists = False
        else:
            output_exists = self._exists(location=self._config.output_location) and not force

        if output_exists:
            logger.info(f"{self.__class__.__name__} - Output file already exists. Pipeline halted.")
        else:
            logger.info(f"{self.__class__.__name__}  - Starting")
        return output_exists
