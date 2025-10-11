#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/data_prep.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 11th 2025 01:05:04 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

from pydantic.dataclasses import dataclass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepCoreConfig:
    """Core configuration for tasks."""

    task_name: str
    force: bool


# ------------------------------------------------------------------------------------------------ #


@dataclass
class DataPrepBaseConfig:
    """Base configuration class for tasks."""

    core_config: DataPrepCoreConfig


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepSingleOutputConfig(DataPrepBaseConfig):
    """Single Output configuration."""

    output_filepath: Path


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepSISOConfig(DataPrepBaseConfig):
    """Single Input Single Output configuration."""

    input_filepath: Path
    output_filepath: Path
