#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/parquet/pandas.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 10:09:20 pm                                             #
# Modified   : Wednesday October 15th 2025 11:53:39 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional

from valuation.utils.io.base import ReadKwargs, WriteKwargs


# ------------------------------------------------------------------------------------------------ #
#                                       PANDAS PARQUET KWARGS                                      #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasReadParquetKwargs(ReadKwargs):
    engine: str = "pyarrow"
    dtype_backend: str = "pyarrow"
    filesystem: Any = None

    @property
    def read_kwargs(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteParquetKwargs(WriteKwargs):
    engine: str = "pyarrow"
    index: bool = False
    partition_cols: Optional[List[str]] = None
    compression: str = "zstd"

    @property
    def write_kwargs(self) -> dict[str, Any]:
        return asdict(self)
