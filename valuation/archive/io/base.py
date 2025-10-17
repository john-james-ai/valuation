#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/base.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 07:28:28 pm                                             #
# Modified   : Thursday October 16th 2025 10:20:52 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from typing import Any, Dict

from abc import ABC, abstractmethod
from dataclasses import dataclass

from valuation.utils.data import DataClass


# ------------------------------------------------------------------------------------------------ #
class IO(ABC):  # pragma: no cover

    @classmethod
    @abstractmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        pass

    @classmethod
    @abstractmethod
    def write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                       READ/WRITE KWARGS                                          #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ReadKwargs(DataClass):

    @property
    @abstractmethod
    def kwargs(self) -> Dict[str, Any]:
        pass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class WriteKwargs(DataClass):

    @property
    @abstractmethod
    def kwargs(self) -> Dict[str, Any]:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                    IO SERVICE BASE CLASS                                         #
# ------------------------------------------------------------------------------------------------ #
class IOService:  # pragma: no cover
    """Base class for IO services."""

    @classmethod
    @abstractmethod
    def read(cls, filepath: str, **kwargs):
        """Reads data from a file."""

    @classmethod
    @abstractmethod
    def write(cls, filepath: str, **kwargs) -> None:
        """Writes data to a file."""

    @classmethod
    @abstractmethod
    def _get_io_handler(cls, filepath: str, **kwargs):
        """Gets the appropriate IO handler based on file extension."""
