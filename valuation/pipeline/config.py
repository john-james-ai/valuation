#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/pipeline/config.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Sunday October 12th 2025 05:56:10 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Configuration settings for pipeline module."""
from pathlib import Path

from pydantic.dataclasses import dataclass

# ------------------------------------------------------------------------------------------------ #


@dataclass
class DataPrepBaseConfig:
    """Base configuration class for tasks."""

    task_name: str
    force: bool


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
