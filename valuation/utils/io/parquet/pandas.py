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
# Modified   : Thursday October 16th 2025 01:14:20 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, List, Optional

import pandas as pd

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
    def kwargs(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteParquetKwargs(WriteKwargs):
    engine: str = "pyarrow"
    index: bool = False
    partition_cols: Optional[List[str]] = None
    compression: str = "zstd"

    @property
    def kwargs(self) -> dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
class PandasParquetIO:  # pragma: no cover

    __read_kwargs_class__ = PandasReadParquetKwargs
    __write_kwargs_class__ = PandasWriteParquetKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> pd.DataFrame:

        read_kwargs = cls.__read_kwargs_class__(**kwargs).kwargs
        return pd.read_parquet(filepath, **read_kwargs)

    @classmethod
    def write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        **kwargs,
    ) -> None:

        write_kwargs = cls.__write_kwargs_class__(**kwargs).kwargs
        data.to_parquet(filepath, **write_kwargs)
