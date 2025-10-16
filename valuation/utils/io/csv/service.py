#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/csv/service.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 11:57:38 pm                                             #
# Modified   : Thursday October 16th 2025 12:07:40 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides a service layer for IO operations."""
import os

from valuation.utils.io.base import IOService
from valuation.utils.io.csv.dask import DaskCSVIO
from valuation.utils.io.csv.pandas import PandasCSVIO

# ------------------------------------------------------------------------------------------------ #
IO_HANDLERS = {
    "dask": DaskCSVIO,
    "pandas": PandasCSVIO,
}


# ------------------------------------------------------------------------------------------------ #
class CSVIOService(IOService):  # pragma: no cover

    @classmethod
    def read(cls, filepath: str, engine: str = "pandas", **kwargs):
        handler = cls._get_io_handler(filepath=filepath, engine=engine)
        filepath = cls._format_filepath(filepath, engine)
        return handler.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data, engine: str = "pandas", **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        handler = cls._get_io_handler(filepath=filepath, engine=engine)
        filepath = cls._format_filepath(filepath, engine)
        return handler.write(filepath, data, **kwargs)

    @classmethod
    def _get_io_handler(cls, filepath: str, engine: str = "pandas"):
        ext = filepath.split(".")[-1]

        try:
            handler = IO_HANDLERS[engine]
            return handler
        except KeyError:
            raise ValueError(f"{engine} is not a supported engine for csv extension")

    @classmethod
    def _format_filepath(cls, filepath: str, engine: str) -> str:
        if os.path.isdir(filepath) and engine == "dask":
            return f"{filepath}/*"
        return filepath
