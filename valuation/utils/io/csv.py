#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/csv.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 08:21:32 pm                                             #
# Modified   : Wednesday October 15th 2025 08:39:10 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from valuation.utils.io.base import IO, ReadKwargs, WriteKwargs

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                         PANDAS CSV KWARGS                                        #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasReadCSVKwargs(ReadKwargs):
    sep: str = ","
    header: Union[int, str] = "infer"
    names: List[str] = field(default_factory=list)
    index_col: Optional[Union[bool, int, str]] = None  # NOTE: None is the pandas default.
    usecols: Optional[List[str]] = None
    mangle_dupe_cols: bool = True
    dtype: Optional[Dict[str, Any]] = None
    engine: str = "c"
    na_values: Any = None
    keep_default_na: bool = True
    na_filter: bool = True
    verbose: bool = False
    skip_blank_lines: bool = True
    parse_dates: bool = False
    infer_datetime_format: bool = False
    keep_date_col: bool = False
    day_first: bool = False
    cache_dates: bool = True
    compression: Union[str, None] = "infer"
    thousands: Optional[str] = None
    lineterminator: Optional[str] = "\n"
    low_memory: bool = False
    encoding: str = "utf-8"
    on_bad_lines: str = "warn"
    delim_whitespace: bool = False


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteCSVKwargs(WriteKwargs):
    sep: str = ","
    float_format: Optional[str] = None
    columns: Optional[List[str]] = None
    header: bool = True
    index: bool = False
    index_label: Optional[str] = None
    mode: str = "w"
    encoding: str = "utf-8"
    compression: Union[str, Dict[str, str], None] = "infer"
    line_terminator: str = "\n"
    chunk_size: Optional[int] = None
    date_format: Optional[str] = None
    errors: str = "strict"


# ------------------------------------------------------------------------------------------------ #
class PandasCSVIO(IO):
    """Provides an I/O interface for reading and writing CSV files using pandas.

    This class uses a factory pattern with classmethods to handle I/O without
    requiring an instance. It leverages class-level attributes to define
    configuration dataclasses (__read_kwargs_class__ and __write_kwargs_class__),
    which validate and manage default arguments for the underlying pandas functions.
    """

    __read_kwargs_class__ = PandasReadCSVKwargs
    __write_kwargs_class__ = PandasWriteCSVKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> pd.DataFrame:
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
        read_kwargs = cls.__read_kwargs_class__(**kwargs).as_dict()
        return pd.read_csv(filepath, **read_kwargs)

    @classmethod
    def write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
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
        write_kwargs = cls.__write_kwargs_class__(**kwargs).as_dict()
        data.to_csv(filepath, **write_kwargs)
