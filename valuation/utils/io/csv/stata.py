#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/csv/stata.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 10:25:13 pm                                             #
# Modified   : Wednesday October 15th 2025 10:37:41 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from valuation.utils.io.base import IO, ReadKwargs, WriteKwargs

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                         PANDAS Stata KWARGS                                        #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasReadStataKwargs(ReadKwargs):
    convert_dates: bool = True
    convert_categoricals: bool = False
    convert_missing: bool = False
    index_col: Optional[Union[bool, int, str]] = None  # NOTE: None is the pandas default.
    preserve_dtypes: bool = True
    columns: List[str] = field(default_factory=list)
    order_categoricals: bool = False

    @property
    def read_kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(self)
        if not self.columns:
            kwargs.pop("columns")  # Remove empty list to use pandas default of None
        return kwargs


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteStataKwargs(WriteKwargs):
    convert_dates: bool = True
    write_index: bool = False
    time_stamp: Optional[datetime] = None
    data_label: Optional[str] = None
    value_labels: List[Dict[Any, str]] = field(default_factory=list)
    variable_labels: Dict[str, str] = field(default_factory=dict)
    version: int = 114
    compression: str = "infer"

    @property
    def write_kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(self)
        if not self.value_labels:
            kwargs.pop("value_labels")  # Remove empty list to use pandas default of None
        if not self.variable_labels:
            kwargs.pop("variable_labels")  # Remove empty dict to use pandas default of None
        return kwargs


# ------------------------------------------------------------------------------------------------ #
class PandasStataIO(IO):
    """Provides an I/O interface for reading and writing Stata files using pandas.

    This class uses a factory pattern with classmethods to handle I/O without
    requiring an instance. It leverages class-level attributes to define
    configuration dataclasses (__read_kwargs_class__ and __write_kwargs_class__),
    which validate and manage default arguments for the underlying pandas functions.
    """

    __read_kwargs_class__ = PandasReadStataKwargs
    __write_kwargs_class__ = PandasWriteStataKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> pd.DataFrame:
        """Reads a Stata file into a pandas DataFrame.

        The provided keyword arguments are used to override the defaults specified
        in the __read_kwargs_class__ dataclass. This provides both flexibility
        and validation, as unknown arguments will raise a TypeError.

        Args:
            filepath (str): The absolute or relative path to the Stata file.
            **kwargs: Keyword arguments to override the default read settings.
                These are passed directly to `pandas.read_csv`.

        Returns:
            pd.DataFrame: The data from the Stata file as a pandas DataFrame.

        Raises:
            TypeError: If an unsupported keyword argument is provided in `**kwargs`.
        """
        read_kwargs = cls.__read_kwargs_class__(**kwargs).read_kwargs
        return pd.read_stata(filepath, **read_kwargs)

    @classmethod
    def write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Writes a pandas DataFrame to a Stata file.

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
        write_kwargs = cls.__write_kwargs_class__(**kwargs).write_kwargs
        data.to_stata(filepath, **write_kwargs)
