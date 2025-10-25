#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/analytics/base.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 25th 2025 10:54:28 am                                              #
# Modified   : Saturday October 25th 2025 11:03:27 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from pathlib import Path

import polars as pl

from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Analytics(ABC):
    """Base class for all analytics types."""

    @staticmethod
    @abstractmethod
    def analyze(df: pl.DataFrame | pl.LazyFrame, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        pass

    @staticmethod
    @abstractmethod
    def visualize(df: pl.DataFrame | pl.LazyFrame, **kwargs) -> None:
        pass

    @staticmethod
    def save(filepath: Path | str, df: pl.DataFrame | pl.LazyFrame, **kwargs) -> None:
        IOService.write(filepath=filepath, data=df)

    @staticmethod
    def load(filepath: Path | str, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        return IOService.read(filepath=filepath)
