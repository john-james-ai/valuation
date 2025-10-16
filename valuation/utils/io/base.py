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
# Modified   : Wednesday October 15th 2025 09:42:23 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

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
    pass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class WriteKwargs(DataClass):
    pass
