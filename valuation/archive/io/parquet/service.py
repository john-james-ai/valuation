#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/parquet/service.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 11:57:38 pm                                             #
# Modified   : Thursday October 16th 2025 02:31:17 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides a service layer for IO operations."""
from typing import Union

import os

import dask.dataframe as dd
import pandas as pd

from valuation.utils.io.base import IOService
from valuation.utils.io.parquet.dask import DaskParquetIO
from valuation.utils.io.parquet.pandas import PandasParquetIO

# ------------------------------------------------------------------------------------------------ #
IO_HANDLERS = {
    "dask": DaskParquetIO,
    "pandas": PandasParquetIO,
}


# ------------------------------------------------------------------------------------------------ #
class ParquetIOService(IOService):  # pragma: no cover

    @classmethod
    def read(
        cls, filepath: str, engine: str = "pandas", **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        handler = cls._get_io_handler(engine=engine)
        return handler.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data, engine: str = "pandas", **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        handler = cls._get_io_handler(engine=engine)
        handler.write(filepath, data, **kwargs)

    @classmethod
    def _get_io_handler(cls, engine: str = "pandas"):

        try:
            handler = IO_HANDLERS[engine]
            return handler
        except KeyError:
            raise ValueError(f"{engine} is not a supported engine for csv extension")
