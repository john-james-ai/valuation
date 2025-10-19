#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/app/base/task.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Sunday October 19th 2025 09:32:28 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from typing import Optional

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from valuation.app.state import Status
from valuation.asset.base import Asset
from valuation.core.structure import DataClass
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskConfig(DataClass):
    """Base configuration class for tasks.

    Attributes:
        (inherited from DataClass) provide task-specific configuration fields.
    """


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskResult(DataClass, ABC):
    """Holds task execution metadata."""

    task_name: str
    config: Optional[TaskConfig] = field(default=None)

    # Timestamps
    started: Optional[datetime] = field(default=None)
    ended: Optional[datetime] = field(default=None)
    elapsed: Optional[float] = field(default=0.0)

    # Status
    status: Optional[str] = field(default=Status.PENDING.value)

    def start_task(self) -> None:
        """Mark the task as started and record the start time."""
        self.started = datetime.now()
        self.status = Status.RUNNING.value

    def end_task(self) -> None:
        """Mark the task as ended, compute elapsed time, and set final status.

        Args:
            status (Status): Final status enum value for the task.

        Returns:
            None
        """
        self.ended = datetime.now()
        if self.started:
            self.elapsed = (self.ended - self.started).total_seconds()


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """Abstract base class for tasks.

    Args:
        config (TaskConfig): Configuration for the task.
        io (type[IOService]): IO service type used by the task (an instance is created internally).

    Properties:
        config (TaskConfig): The task configuration.
        task_name (str): The task's class name.
    """

    def __init__(
        self,
        config: TaskConfig,
        io: type[IOService] = IOService,
    ) -> None:

        self._config = config
        self._io = io()

    @property
    def config(self) -> TaskConfig:
        """Return the TaskConfig associated with this task.

        Returns:
            TaskConfig: The task configuration.
        """
        return self._config

    @property
    def task_name(self) -> str:
        """Return the task's class name.

        Returns:
            str: The name of the task class.
        """
        return self.__class__.__name__

    @abstractmethod
    def run(self, asset: Asset) -> TaskResult:
        """Execute the task against the provided asset.

        Args:
            asset (Asset): The asset to process.

        Returns:
            TaskResult: Result object containing timestamps, status, and any output data.
        """
        pass
