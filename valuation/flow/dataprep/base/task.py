#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/base/task.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Saturday October 25th 2025 03:46:49 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from typing import Optional

from abc import abstractmethod

import pandas as pd

from valuation.flow.base.task import Task
from valuation.flow.dataprep.validation import Validation


# ------------------------------------------------------------------------------------------------ #
class DataPrepTask(Task):
    """Abstract base class for executable tasks.

    Args:
        config (TaskConfig): Configuration object for the task instance.

    Properties:
        config (TaskConfig): The task configuration.
    """

    def __init__(self, validation: Optional[Validation] = None) -> None:
        super().__init__()
        self._validation = validation or Validation()

    @property
    def validation(self) -> Validation:
        """Gets the validation configuration."""
        return self._validation

    @abstractmethod
    def run(self, df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
        """Execute the task using the supplied DataFrame and produce a transformed DataFrame.

        Args:
            df (pd.DataFrame): Input DataFrame to be processed by the task.
            force (bool): If True, force reprocessing even if outputs already exist. Defaults to False.

        Returns:
            pd.DataFrame: The resulting DataFrame after task execution.
        """
        pass
