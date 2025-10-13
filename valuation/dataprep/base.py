#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/base.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Monday October 13th 2025 06:11:52 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
import pandas as pd

from valuation.config.data import DTYPES
from valuation.utils.data import DataClass
from valuation.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskConfig:
    """Base configuration class for tasks."""

    force: bool
    input_location: Union[Path, Dict[str, Path]]
    output_location: Union[Path, Dict[str, Path]]


# ------------------------------------------------------------------------------------------------ #
class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    SUCCESS = (0, "Success")
    FAILURE = (1, "Failure")
    WARNING = (2, "Warning")
    SKIPPED = (3, "Existing File - Skipped")

    @classmethod
    def __new__(cls, code: int, result: str) -> TaskStatus:
        obj = object.__new__(cls)
        obj._value_ = code
        obj.display = result  # type: ignore
        return obj


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskReport(DataClass):
    """Holds the results of a task execution."""

    task_name: str
    config: TaskConfig = field(default=None)  # Optional until setup
    started: Optional[datetime] = field(default=None)  # Optional until setup
    # The Engine manages this state
    ended: Optional[datetime] = field(default=None)  # Optional until teardown
    elapsed: Optional[float] = field(default=None)  # Optional until teardown
    records_in: Optional[int] = field(default=None)  # Only known after initial load
    records_out: Optional[int] = field(default=None)  # Only known after execute
    status: TaskStatus = TaskStatus.WARNING  # Set an initial state


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    def __init__(self, config: TaskConfig, io: IOService = IOService) -> None:
        self._config = config
        self._io = io
        self._task_result: Optional[Union[pd.DataFrame, Any]] = None
        self._task_report = TaskReport(task_name=self.__name__, config=self._config)

    @property
    def config(self) -> TaskConfig:
        """Gets the task configuration."""
        return self._config

    @property
    def result(self) -> TaskReport:
        """Gets the task result."""
        return self._task_report

    @property
    def report(self) -> TaskReport:
        """Gets the task report."""
        return self._task_report

    def run(self, force: bool = False) -> None:
        """Runs the SISO task."""
        self._setup()

        if self._output_exists(force=force):
            self._task_report.status = TaskStatus.SKIPPED
            self._task_report.records_out = None
            self._teardown()
            return

        # Load input data
        input_data = self._load(filepath=self._config.input_location)  # type: ignore
        self._task_report.records_in = len(input_data)

        # Execute the task
        self._task_result = self._execute(data=input_data)

        # Validate the output
        if not self._validate(data=self._task_result):
            self._task_report.status = TaskStatus.FAILURE
            self._task_report.records_out = 0
        else:
            self._task_report.status = TaskStatus.SUCCESS
            self._task_report.records_out = len(self._task_result)
            # Save the output data
            self._save(df=output_data, filepath=self._config.output_location)  # type: ignore

        self._teardown()

    @abstractmethod
    def _execute(self, *args, **kwargs) -> Union[pd.DataFrame, Any]:
        """Executes the core logic of the task."""
        pass

    @abstractmethod
    def _validate(self, data: Union[pd.DataFrame, Any]) -> bool:
        """Validates the output data."""
        pass

    def _setup(self) -> None:
        """Sets up the task environment."""
        self._task_report.started = datetime.now()

    def _teardown(self) -> None:
        """Cleans up the task environment."""
        self._task_report.ended = datetime.now()
        self._task_report.elapsed = (
            self._task_report.ended - self._task_report.started
        ).total_seconds()  # type: ignore
        logger.info(self._task_report)

    def _load(self, filepath: Path) -> pd.DataFrame:
        """Loads a single data file from the raw data directory.

        Args:
            location(Path): The path to the file to be loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """

        df = self._io.read(filepath=filepath)
        # Ensure correct data types
        return df.astype({k: v for k, v in DTYPES.items() if k in df.columns})

    def _save(self, df: pd.DataFrame, filepath: Path) -> None:
        """Saves a DataFrame to the processed data directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            location (Path): The path to the file to be saved.
        """
        self._io.write(data=df, filepath=filepath)

    def _delete(self, location: Path) -> None:
        """Deletes a file from the processed data directory.

        Args:
            location (Path): The path of the file to delete
        """
        location.unlink(missing_ok=True)

    def _exists(self, location: Path) -> bool:
        """Checks if a file exists in the processed data directory.

        Args:
            location (Path): The path to a file for the existence check

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return location.exists()

    def _output_exists(self, force: bool = False) -> bool:
        """Determines whether to use cached data based on file existence and force flag.

        Args:
            config (DataPrepSISOConfig): Configuration object containing core settings.
        Returns:
            bool: True if cached data should be used, False otherwise.

        """
        if force:
            self._delete(location=self._config.output_location)
            use_cache = False
        else:
            use_cache = self._exists(location=self._config.output_location) and not force

        if use_cache:
            logger.info(
                f"{self.__class__.__name__} - Output file already exists. Using cached data."
            )
        else:
            logger.info(f"{self.__class__.__name__}  - Starting")
        return use_cache
