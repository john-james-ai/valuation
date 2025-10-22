#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/base/pipeline.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Wednesday October 22nd 2025 12:02:25 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import List, Optional

from abc import abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from valuation.core.dataclass import DataClass
from valuation.core.state import Status
from valuation.flow.base.task import TaskResult


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PipelineConfig(DataClass):
    """Holds all parameters for the pipeline."""

    name: str


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PipelineResult(DataClass):
    """Holds the results of a pipeline execution."""

    name: str
    # Timestamps
    started: Optional[datetime] = field(default=None)  # Optional until setup
    ended: Optional[datetime] = field(default=None)  # Optional until teardown
    elapsed: Optional[float] = field(default=None)  # Optional until teardown

    # State
    status: Optional[str] = field(default=None)
    status_obj = Status.PENDING
    # Pipeline task results
    task_results: List[TaskResult] = field(default_factory=list)

    def add_task_result(self, result: TaskResult) -> PipelineResult:
        """Adds a task result to the pipeline result.

        Args:
            result (TaskResult): The result of a task execution.
        """
        self.task_results.append(result)
        return self

    def end_pipeline(self) -> None:
        self.status = self.status_obj.value[0]
        self.ended = datetime.now()
        if self.started:
            self.elapsed = (self.ended - self.started).total_seconds()
        else:
            self.elapsed = None


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
        self._tasks = []

    def _execute(self, pipeline_result: PipelineResult, force: bool = False) -> PipelineResult:
        """ "Executes the pipeline logic.
        Args:
            data (pd.DataFrame): The input data for the pipeline.
            result (PipelineResult): The result object to populate during execution.

        Returns:
            PipelineResult: The updated result object after execution."""
        dataset = None
        for task in self._tasks:
            task_result = task.run(dataset=dataset, force=force)
            pipeline_result.add_task_result(task_result)
            if not task_result.validation.is_valid:
                msg = "Task {task.__class__.__name__} failed validation."
                logger.error(msg)
                raise RuntimeError(msg)
            dataset = task_result.dataset

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

    @abstractmethod
    def run(self, force: bool = False) -> PipelineResult:
        """Runs all tasks in the pipeline.

        Args:
            force (bool): Whether to force reprocessing if the file already exists.
        Returns:
            PipelineResult: The result of the pipeline execution.
        """
