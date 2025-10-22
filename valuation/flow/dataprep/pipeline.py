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
# Modified   : Wednesday October 22nd 2025 04:53:22 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Dict, Optional

from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger

from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.pipeline import Pipeline, PipelineConfig, PipelineResult
from valuation.flow.validation import Validation
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatePrepPipelineConfig(PipelineConfig):
    """Holds all parameters for the pipeline."""

    target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepPipelineResult(PipelineResult):
    """Holds the results of a pipeline execution."""

    validation: Dict[str, Validation] = field(default_factory=dict)
    num_errors: int = 0
    num_warnings: int = 0


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

    _config: DatePrepPipelineConfig
    _dataset_store: DatasetStore

    def __init__(self, config: DatePrepPipelineConfig, dataset_store: DatasetStore) -> None:
        super().__init__(config=config)
        self._dataset_store = dataset_store

    def _execute(
        self, pipeline_result: DataPrepPipelineResult, force: bool = False, persist: bool = False
    ) -> Optional[DataPrepPipelineResult]:

        pipeline_result.status_obj = Status.RUNNING

        try:

            dataset = None

            for task in self._tasks:

                if self._dataset_store.exists(dataset_id=task.config.target.id) and not force:
                    logger.info(
                        f"Dataset {task.config.target.label} already exists in the store. Skipping task {task.__class__.__name__}."
                    )
                    dataset = self._dataset_store.get(passport=task.config.target)

                else:
                    # Remove the task endpoint if it exists
                    self._dataset_store.remove(passport=task.config.target)

                    # Run the task
                    task_result = task.run(dataset=dataset)

                    # Extract the validation result and add to pipeline result
                    pipeline_result.validation[task.task_name] = task.validation

                    # Add the task result to the pipeline result
                    pipeline_result.add_task_result(task_result)

                    # Extract the dataset for the next task
                    dataset = task_result.dataset

                    # Add the task validator

                    # Check validation status
                    if not task_result.validation.is_valid:
                        pipeline_result.status_obj = Status.FAIL
                        task_result.validation.report()
                        break

                    pipeline_result.num_errors += task_result.num_errors
                    pipeline_result.num_warnings += task_result.num_warnings

                    # If persist is True, save the intermediate dataset
                    if persist and task_result.validation.is_valid:
                        self._dataset_store.add(dataset=dataset, overwrite=force)

                # Add final dataset to the dataset store.
                self._dataset_store.add(dataset=dataset, overwrite=force)

            if pipeline_result.status_obj != Status.FAIL:
                pipeline_result.status_obj = Status.SUCCESS

            pipeline_result.end_pipeline()
            logger.info(pipeline_result)
            return pipeline_result
        except Exception as e:
            logger.critical(f"Pipeline {self.__class__.__name__} failed with exception: {e}")
            pipeline_result.status_obj = Status.FAIL
            logger.critical(pipeline_result)
            return pipeline_result

    def add_task(self, task) -> DataPrepPipeline:
        """Adds a task to the pipeline.

        Args:
            task: An instance of a data preparation or modeling task.

        Returns:
            Pipeline: The current instance of the pipeline (for method chaining).
        """
        self._tasks.append(task)
        return self

    def run(self, force: bool = False, persist: bool = False) -> Optional[DataPrepPipelineResult]:
        """Runs all tasks in the pipeline.

        Args:
            force (bool): If True, forces re-execution of all tasks.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        pipeline_result = DataPrepPipelineResult(name=self._config.name)
        pipeline_result.started = datetime.now()
        try:
            # Check pipeline endpoint and handle existing dataset
            if self._dataset_store.exists(dataset_id=self._config.target.id) and not force:
                msg = f"Endpoint dataset {self._config.target.label} already exists in the store. Skipping pipeline execution."
                logger.info(msg)
                pipeline_result.status_obj = Status.SKIPPED
                logger.info(pipeline_result)
                return pipeline_result

            # We're going to run the pipeline, so remove existing endpoint
            self._dataset_store.remove(passport=self._config.target)

            # Execute the pipeline
            pipeline_result = self._execute(
                pipeline_result=pipeline_result, force=force, persist=persist
            )
            return pipeline_result
        except Exception as e:
            logger.critical(f"Pipeline {self._config.name} failed with exception: {e}")
            pipeline_result.status_obj = Status.FAIL
            logger.error(pipeline_result)
