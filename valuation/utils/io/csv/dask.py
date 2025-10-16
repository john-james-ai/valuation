#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/csv/dask.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 08:21:32 pm                                             #
# Modified   : Thursday October 16th 2025 01:00:34 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from typing import Any, Dict, Optional, Union

from dataclasses import asdict, dataclass

import dask.dataframe as dd

from valuation.config.data import DTYPES
from valuation.utils.io.base import IO, ReadKwargs, WriteKwargs
from valuation.utils.io.csv.base import CompressionType

DASK_COMPRESSION_TYPES = {
    CompressionType.GZIP,
    CompressionType.BZ2,
    CompressionType.XZ,
}


# ------------------------------------------------------------------------------------------------ #
#                                         PANDAS CSV KWARGS                                        #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskReadCSVKwargs(ReadKwargs):
    blocksize: Optional[Union[str, int]] = "128MB"
    assume_missing: bool = False
    dtype: Optional[Dict[str, Any]] = None  # Default is None, which infers dtypes.

    @property
    def kwargs(self) -> Dict[str, Any]:
        self.dtype = DTYPES
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskWriteCSVKwargs(WriteKwargs):
    single_file: bool = False
    header_first_partition_only: bool = True  # Prevents headers in every partition file.
    compression: Optional[str] = None
    compute: bool = True
    mode: str = "w"
    encoding: str = "utf-8"
    index: bool = False

    @property
    def kwargs(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
class DaskCSVIO(IO):
    """Provides an I/O interface for reading and writing CSV files using pandas.

    This class uses a factory pattern with classmethods to handle I/O without
    requiring an instance. It leverages class-level attributes to define
    configuration dataclasses (__read_kwargs_class__ and __write_kwargs_class__),
    which validate and manage default arguments for the underlying pandas functions.
    """

    __read_kwargs_class__ = DaskReadCSVKwargs
    __write_kwargs_class__ = DaskWriteCSVKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> dd.DataFrame:
        """Reads a CSV file into a pandas DataFrame.

        The provided keyword arguments are used to override the defaults specified
        in the __read_kwargs_class__ dataclass. This provides both flexibility
        and validation, as unknown arguments will raise a TypeError.

        Args:
            filepath (str): The absolute or relative path to the CSV file.
            **kwargs: Keyword arguments to override the default read settings.
                These are passed directly to `pandas.read_csv`.

        Returns:
            pd.DataFrame: The data from the CSV file as a pandas DataFrame.

        Raises:
            TypeError: If an unsupported keyword argument is provided in `**kwargs`.
        """
        read_kwargs = cls.__read_kwargs_class__(**kwargs).kwargs
        return dd.read_csv(filepath, **read_kwargs)

    @classmethod
    def write(cls, filepath: str, data: dd.DataFrame, **kwargs) -> None:
        """Writes a pandas DataFrame to a CSV file.

        The provided keyword arguments are used to override the defaults specified
        in the __write_kwargs_class__ dataclass. This provides both flexibility
        and validation, as unknown arguments will raise a TypeError.

        Args:
            filepath (str): The absolute or relative path for the output file.
            data (pd.DataFrame): The DataFrame to write to the file.
            **kwargs: Keyword arguments to override the default write settings.
                These are passed directly to `pandas.DataFrame.to_csv`.

        Raises:
            TypeError: If an unsupported keyword argument is provided in `**kwargs`.
        """
        write_kwargs = cls.__write_kwargs_class__(**kwargs).kwargs
        dd.to_csv(df=data, filename=filepath, **write_kwargs)
