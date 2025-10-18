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
# Modified   : Saturday October 18th 2025 06:11:15 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from types import TracebackType
import typing
from typing import List, Optional, Type, Union

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import time
import traceback

from loguru import logger
import pandas as pd

from valuation.utils.data import DataClass, Dataset
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
    input_filepath: Union[Path, str]
    output_filepath: Union[Path, str]


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
    dataset: Optional[Dataset] = field(default=None)

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

    def __init__(self, config: PipelineConfig):
        self._config = config
        self._pipeline_context = PipelineContext(config=config)
        self._tasks = []

    def _execute(self, pipeline_result: PipelineResult) -> PipelineResult:
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
            data = task_result.dataset.data

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
                # Execute the pipeline
                pipeline_result = self._execute(pipeline_result=pipeline_result)
                # Finalize the pipeline
                pipeline_result = self._finalize(result=pipeline_result)
        except Exception as e:
            logger.critical(f"Pipeline {self._config.name} failed with exception: {e}")
            pipeline_result.status = Status.FAILURE.value
            raise e
        finally:
            return pipeline_result

    def _finalize(self, result: PipelineResult) -> PipelineResult:
        """Finalizes the pipeline after execution.

        Saves the output data and updates the result metrics.

        Args:
            result (PipelineResult): The result of the pipeline execution.

        Returns:
            PipelineResult: The finalized result of the pipeline execution.
        """
        if not isinstance(result.dataset, Dataset) or not isinstance(
            result.dataset.data, pd.DataFrame
        ):
            raise ValueError("No valid dataset to save for the pipeline.")

        logger.debug(f"{self.__class__.__name__} - Finalizing")
        if result.status == Status.SUCCESS.value and result.dataset.data is not None:

            result.num_records = (
                len(result.dataset.data) if isinstance(result.dataset.data, pd.DataFrame) else 0
            )
            result.num_fields = (
                len(result.dataset.data.columns)
                if isinstance(result.dataset.data, pd.DataFrame)
                else 0
            )
            result.memory_mb = (
                result.dataset.data.memory_usage(deep=True).sum() / (1024 * 1024)
                if isinstance(result.dataset.data, pd.DataFrame)
                else 0
            )
            result.filesize_mb = Path(self._config.output_filepath).stat().st_size / (
                1024 * 1024
            )  # in MB
        return result
