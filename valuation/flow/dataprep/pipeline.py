#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/pipeline.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Tuesday October 21st 2025 06:48:35 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from loguru import logger

from valuation.core.state import Status
from valuation.flow.base.pipeline import Pipeline, PipelineConfig, PipelineResult


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatePrepPipelineConfig(PipelineConfig):
    """Holds all parameters for the pipeline."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepPipelineResult(PipelineResult):
    """Holds the results of a pipeline execution."""


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipeline(Pipeline):
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

    def __init__(self, config: DatePrepPipelineConfig):
        super().__init__(config=config)

    def _execute(
        self, pipeline_result: PipelineResult, force: bool = False
    ) -> DataPrepPipelineResult:
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

        return pipeline_result  # type: ignore

    def add_task(self, task) -> DataPrepPipeline:
        """Adds a task to the pipeline.

        Args:
            task: An instance of a data preparation or modeling task.

        Returns:
            Pipeline: The current instance of the pipeline (for method chaining).
        """
        self._tasks.append(task)
        return self

    def run(self, force: bool = False) -> DataPrepPipelineResult:
        """Runs all tasks in the pipeline.

        Args:
            force (bool): If True, forces re-execution of all tasks.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        pipeline_result = DataPrepPipelineResult(name=self._config.name)
        pipeline_result.started = datetime.now()
        try:

            # Execute the pipeline
            pipeline_result = self._execute(pipeline_result=pipeline_result, force=force)
        except Exception as e:
            logger.critical(f"Pipeline {self._config.name} failed with exception: {e}")
            pipeline_result.status_obj = Status.FAIL
            raise e
        finally:
            return pipeline_result
