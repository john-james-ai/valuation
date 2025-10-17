#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/parquet/dask.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 10:08:23 pm                                             #
# Modified   : Thursday October 16th 2025 03:10:31 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dataclasses import asdict, dataclass
import os

import dask.dataframe as dd

from valuation.utils.io.base import IO, ReadKwargs, WriteKwargs


# ------------------------------------------------------------------------------------------------ #
#                                        DASK PARQUET KWARGS                                       #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskReadParquetKwargs(ReadKwargs):
    dtype_backend: str = "pyarrow"  # Performant choice.
    calculate_divisions: bool = False  # NOTE: Correct, as True can be very slow.
    ignore_metadata_file: bool = False
    split_row_groups: Union[bool, int, str] = "infer"
    # blocksize: Union[int, str] = "256MB" # NOTE: Not supported when using pyarrow engine.
    aggregate_files: Optional[bool] = None  # NOTE: Let Dask handle this by default.
    filesystem: str = (
        "arrow"  # Specifying filesystem="arrow" leverages a complete reimplementation of the Parquet reader that is solely based on PyArrow. It is significantly faster than the legacy implementation, but doesn’t yet support all features.
    )

    @property
    def kwargs(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskWriteParquetKwargs(WriteKwargs):
    compression: str = "zstd"
    write_index: bool = False
    append: bool = False
    overwrite: bool = False
    partition_on: Optional[List[str]] = None
    write_metadata_file: bool = True
    compute: bool = True
    schema: Union[str, Dict[str, Any]] = "infer"

    @property
    def kwargs(self) -> Dict[str, Any]:
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
class DaskParquetIO(IO):
    """Provides an I/O interface for reading and writing Parquet files using pandas.

    This class uses a factory pattern with classmethods to handle I/O without
    requiring an instance. It leverages class-level attributes to define
    configuration dataclasses (__read_kwargs_class__ and __write_kwargs_class__),
    which validate and manage default arguments for the underlying pandas functions.
    """

    __read_kwargs_class__ = DaskReadParquetKwargs
    __write_kwargs_class__ = DaskWriteParquetKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> dd.DataFrame:
        """Reads a Parquet file into a pandas DataFrame.

        The provided keyword arguments are used to override the defaults specified
        in the __read_kwargs_class__ dataclass. This provides both flexibility
        and validation, as unknown arguments will raise a TypeError.

        Args:
            filepath (str): The absolute or relative path to the Parquet file.
            **kwargs: Keyword arguments to override the default read settings.
                These are passed directly to `pandas.read_csv`.

        Returns:
            pd.DataFrame: The data from the Parquet file as a pandas DataFrame.

        Raises:
            TypeError: If an unsupported keyword argument is provided in `**kwargs`.
        """
        read_kwargs = cls.__read_kwargs_class__(**kwargs).kwargs
        return dd.read_parquet(os.path.abspath(filepath), **read_kwargs)

    @classmethod
    def write(cls, filepath: str, data: dd.DataFrame, **kwargs) -> None:
        """Writes a pandas DataFrame to a Parquet file.

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
        dd.to_parquet(df=data, path=filepath, **write_kwargs)
