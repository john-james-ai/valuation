#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/service.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 11:57:38 pm                                             #
# Modified   : Thursday October 16th 2025 12:47:11 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides a service layer for IO operations."""
import os

from valuation.utils.io.csv.dask import DaskCSVIO
from valuation.utils.io.csv.pandas import PandasCSVIO
from valuation.utils.io.json import JsonIO
from valuation.utils.io.parquet.dask import DaskParquetIO
from valuation.utils.io.parquet.pandas import PandasParquetIO
from valuation.utils.io.yaml import YamlIO
from valuation.utils.io.zipfile.dask import ZipFileDaskIO
from valuation.utils.io.zipfile.pandas import ZipFilePandasIO

# ------------------------------------------------------------------------------------------------ #
IO_HANDLERS = {
    "csv": {
        "dask": DaskCSVIO,
        "pandas": PandasCSVIO,
    },
    "parquet": {
        "dask": DaskParquetIO,
        "pandas": PandasParquetIO,
    },
    "zip": {
        "dask": ZipFileDaskIO,
        "pandas": ZipFilePandasIO,
    },
    "yaml": YamlIO,
    "json": JsonIO,
}


# ------------------------------------------------------------------------------------------------ #
class IOService:  # pragma: no cover

    @classmethod
    def read(cls, filepath: str, engine: str = "pandas", **kwargs):
        handler = cls._get_io_handler(filepath=filepath, engine=engine)
        return handler.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data, engine: str = "pandas", **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        handler = cls._get_io_handler(filepath=filepath, engine=engine)
        return handler.write(filepath, data, **kwargs)

    @classmethod
    def _get_io_handler(cls, filepath: str, engine: str = "pandas"):
        ext = filepath.split(".")[-1]
        # Confirm the extension is supported
        if ext not in IO_HANDLERS:
            raise ValueError(f"Unsupported file extension: {ext}")
        handler = IO_HANDLERS[ext]
        if isinstance(handler, dict):
            if engine not in handler:
                raise ValueError(f"Unsupported engine '{engine}' for extension '{ext}'")
            return handler[engine]
        return handler
