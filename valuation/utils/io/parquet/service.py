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
# Modified   : Thursday October 16th 2025 11:40:30 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides a service layer for IO operations."""
from typing import Union

from abc import abstractmethod
import os

import dask.dataframe as dd
import pandas as pd

from valuation.utils.io.base import IO, IOService
from valuation.utils.io.csv.dask import DaskCSVIO
from valuation.utils.io.csv.pandas import PandasCSVIO
from valuation.utils.io.json import JsonIO
from valuation.utils.io.parquet.dask import DaskParquetIO
from valuation.utils.io.parquet.pandas import PandasParquetIO
from valuation.utils.io.yaml import YamlIO
from valuation.utils.io.zipfile.dask import ZipFileDaskIO
from valuation.utils.io.zipfile.pandas import ZipFilePandasIO

# ------------------------------------------------------------------------------------------------ #
IO_HANDLERS_TABULAR = {
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
}
IO_HANDLERS_OBJECT = {
    "yaml": YamlIO,
    "json": JsonIO,
}


# ------------------------------------------------------------------------------------------------ #
class TabularIOService(IOService):  # pragma: no cover

    @classmethod
    @abstractmethod
    def read(
        cls, filepath: str, engine: str = "pandas", **kwargs
    ) -> Union[pd.DataFrame, dd.DataFrame]:
        """Reads tabular data from a file.

        Args:
            filepath (str): The path to the file.
            engine (str): The engine to use for reading. Options are 'pandas' or 'dask'.
            **kwargs: Additional keyword arguments to pass to the read method of the selected
                engine.

        Returns:
            Union[pd.DataFrame, dd.DataFrame]: The data read from the file as a Pandas or Dask
            DataFrame.
        """

    @classmethod
    @abstractmethod
    def write(cls, filepath: str, data, engine: str = "pandas", **kwargs) -> None:
        """Writes tabular data to a file.

        Args:
            filepath (str): The path to the file.
            data (Union[pd.DataFrame, dd.DataFrame]): The data to write to the file.
            engine (str): The engine to use for writing. Options are 'pandas' or 'dask'.
            **kwargs: Additional keyword arguments to pass to the write method of the selected
                engine.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

    @classmethod
    @abstractmethod
    def _get_io_handler(cls, filepath: str, engine: str = "pandas") -> IO:
        """Gets the appropriate IO handler based on the file extension and engine.
        Args:
            filepath (str): The path to the file.
            engine (str): The engine to use. Options are 'pandas' or 'dask'.

        Returns:
            IO: The IO handler class for the specified file extension and engine.
        """

    @classmethod
    def _format_filepath(cls, filepath: str, engine: str) -> str:
        if os.path.isdir(filepath) and engine == "dask":
            return f"{filepath}/*"
        return filepath


# ------------------------------------------------------------------------------------------------ #
class ObjectIOService(IOService):  # pragma: no cover

    @classmethod
    def read(cls, filepath: str, **kwargs):
        handler = cls._get_io_handler(filepath=filepath)
        return handler.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data, engine: str = "pandas", **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        handler = cls._get_io_handler(filepath=filepath)
        return handler.write(filepath, data, **kwargs)

    @classmethod
    def _get_io_handler(cls, filepath: str, engine: str = "pandas"):
        ext = filepath.split(".")[-1]

        try:
            handler = IO_HANDLERS_OBJECT[ext]
            return handler
        except KeyError:
            raise ValueError(f"Unsupported file extension: {ext}")
