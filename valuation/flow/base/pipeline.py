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
# Modified   : Saturday October 25th 2025 06:10:16 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Dict, Optional, Union

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from distributed import Status
from loguru import logger
import pandas as pd

from valuation.asset.dataset import DTYPES
from valuation.core.dataclass import DataClass
from valuation.core.state import Status
from valuation.infra.file.io.fast import IOService


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

    started: Optional[datetime] = None
    completed: Optional[datetime] = None
    duration: Optional[float] = None

    status: str = None
    status_obj: Status = Status.NOTSTARTED

    def start_pipeline(self) -> None:
        """Marks the start of the pipeline execution."""
        self.started = datetime.now()
        self.status_obj = Status.RUNNING
        self.status = self.status_obj.value[0]

    def end_pipeline(self) -> None:
        """Marks the completion of the pipeline and calculates duration."""
        self.completed = datetime.now()
        if self.started:
            self.duration = (self.completed - self.started).total_seconds()

        # Set status based on status_obj.
        if self.status_obj == Status.RUNNING:
            self.status_obj = Status.SUCCESS
        self.status = self.status_obj.value[0]


# ------------------------------------------------------------------------------------------------ #
class Pipeline(ABC):
    """Abstract base class for pipelines.

    Methods:
        run(force=False): Runs all tasks in the pipeline.
        add_task(task): Adds a task to the pipeline.
    """

    def __init__(self) -> None:
        self._tasks = []

    @abstractmethod
    def run(self, force: bool = False) -> Optional[PipelineResult]:
        """Runs all tasks in the pipeline.

        Args:
            force (bool): If True, forces re-execution of all tasks.

        Returns:
            PipelineResult: The result of the pipeline execution.
        """
        pass

    def add_task(self, task) -> Pipeline:
        """Adds a task to the pipeline.

        Args:
            task: An instance of a data preparation or modeling task.

        Returns:
            Pipeline: The current instance of the pipeline (for method chaining).
        """
        self._tasks.append(task)
        return self

    def _load(self, filepath: Path, **kwargs) -> Union[pd.DataFrame, Dict[str, str]]:
        """Loads a DataFrame from the specified filepath using the I/O service.

        Args:
            filepath: The path to the file to be loaded.
            **kwargs: Additional keyword arguments for the I/O service.

            Returns:
            Union[pd.DataFrame, Any]: The loaded DataFrame or data object."""

        try:

            data = IOService.read(filepath=filepath, **kwargs)
            # Ensure correct data types
            if isinstance(data, pd.DataFrame):
                data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})

            return data
        except Exception as e:
            logger.critical(f"Failed to load data from {filepath.name} with exception: {e}")
            raise e


# ------------------------------------------------------------------------------------------------ #
class PipelineBuilder(ABC):
    """Abstract base class for building pipelines."""

    @abstractmethod
    def reset(self) -> None:
        """Resets the builder to its initial state."""
        pass

    @abstractmethod
    def build(self) -> Pipeline:
        """Builds and returns the pipeline instance.

        Returns:
            Pipeline: The constructed pipeline instance.
        """
        pass
