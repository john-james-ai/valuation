#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/config.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 02:03:56 am                                                #
# Modified   : Saturday October 18th 2025 08:20:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""I/O Configuration Module"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from dataclasses import asdict, dataclass, field

from valuation.core.data import DataClass

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, unused-argument


# ------------------------------------------------------------------------------------------------ #
#                                       READ/WRITE KWARGS                                          #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class ReadKwargs(DataClass):
    pass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class WriteKwargs(DataClass):
    pass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IOPackageKwargs(DataClass):
    read: ReadKwargs = field(default_factory=ReadKwargs)
    write: WriteKwargs = field(default_factory=WriteKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                       PANDAS STATA KWARGS                                        #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasReadStataKwargs(ReadKwargs):
    convert_dates: bool = True
    convert_categoricals: bool = False
    convert_missing: bool = False
    preserve_dtypes: bool = True
    index_col: Optional[str] = None
    columns: Optional[List[str]] = field(default_factory=list)
    convert_strl: bool = False
    chunksize: Optional[int] = None
    iterator: bool = False
    compression: Union[str, Dict[str, str]] = "infer"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteStataKwargs(WriteKwargs):
    write_index: bool = False
    version: int = 114
    convert_dates: bool = True
    compression: Union[str, Dict[str, str]] = "infer"
    variable_labels: Optional[Dict[str, str]] = None
    value_labels: Optional[Dict[str, Dict[Any, str]]] = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasStataKwargs(IOPackageKwargs):
    read: PandasReadStataKwargs = field(default_factory=PandasReadStataKwargs)
    write: PandasWriteStataKwargs = field(default_factory=PandasWriteStataKwargs)


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
    thousands: Optional[str] = None  # Defaulting to ',' is not safe for international data.
    lineterminator: Optional[str] = "\n"
    low_memory: bool = False
    encoding: str = "utf-8"
    on_bad_lines: str = "warn"  # Replaces deprecated error_bad_lines and warn_bad_lines.
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
@dataclass
class PandasCSVKwargs(IOPackageKwargs):
    read: PandasReadCSVKwargs = field(default_factory=PandasReadCSVKwargs)
    write: PandasWriteCSVKwargs = field(default_factory=PandasWriteCSVKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                       PANDAS PARQUET KWARGS                                      #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasReadParquetKwargs(ReadKwargs):
    engine: str = "pyarrow"
    dtype_backend: str = "pyarrow"
    filesystem: Any = None


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasWriteParquetKwargs(WriteKwargs):
    engine: str = "pyarrow"
    index: bool = False
    partition_cols: Optional[List[str]] = None
    compression: str = "zstd"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class PandasParquetKwargs(IOPackageKwargs):
    read: PandasReadParquetKwargs = field(default_factory=PandasReadParquetKwargs)
    write: PandasWriteParquetKwargs = field(default_factory=PandasWriteParquetKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                          DASK CSV KWARGS                                         #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskReadCSVKwargs(ReadKwargs):
    blocksize: Optional[Union[str, int]] = "64MB"
    assume_missing: bool = False
    _pandas_read_kwargs: PandasReadCSVKwargs = field(default_factory=PandasReadCSVKwargs)

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(self._pandas_read_kwargs)
        kwargs.update(asdict(self))  # Merge with pandas kwargs
        return kwargs


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskWriteCSVKwargs(WriteKwargs):
    single_file: bool = False
    header_first_partition_only: bool = True  # Prevents headers in every partition file.
    compression: Optional[str] = (
        "gzip"  # : gzip is the standard for compressing text files like CSV.
    )
    compute: bool = True
    mode: str = "w"
    encoding: str = "utf-8"
    _pandas_write_kwargs: PandasWriteCSVKwargs = field(default_factory=PandasWriteCSVKwargs)

    @property
    def kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(self._pandas_write_kwargs)
        kwargs.update(asdict(self))  # Merge with pandas kwargs
        return kwargs


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskCSVKwargs(IOPackageKwargs):
    read: DaskReadCSVKwargs = field(default_factory=DaskReadCSVKwargs)
    write: DaskWriteCSVKwargs = field(default_factory=DaskWriteCSVKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                        DASK PARQUET KWARGS                                       #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskReadParquetKwargs(ReadKwargs):
    index: Union[str, bool, None] = False  # CORRECT: Avoid loading a meaningless index.
    categories: Optional[Union[List[str], Dict[str, str]]] = None
    dtype_backend: str = "pyarrow"  # CORRECT: Performant choice.
    calculate_divisions: bool = False  # NOTE: Correct, as True can be very slow.
    ignore_metadata_file: bool = (
        False  # CHANGED: You WANT to use the metadata file if it exists; it's much faster.
    )
    split_row_groups: Union[bool, int, str] = "infer"
    blocksize: Union[int, str] = "256MB"
    aggregate_files: Optional[bool] = None  # NOTE: Let Dask handle this by default.


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskWriteParquetKwargs(WriteKwargs):
    compression: str = "zstd"  # CHANGED: Consistent with pandas and modern best practice.
    write_index: bool = False  # CORRECT: Do not write a meaningless index.
    append: bool = False
    overwrite: bool = (
        False  # NOTE: Consider a single 'write_mode' parameter instead of append/overwrite flags.
    )
    partition_on: Optional[List[str]] = None
    write_metadata_file: bool = (
        True  # CHANGED: Explicitly creating this makes subsequent reads much faster.
    )
    compute: bool = True
    schema: Union[str, Dict[str, Any]] = "infer"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskParquetKwargs(IOPackageKwargs):
    read: DaskReadParquetKwargs = field(default_factory=DaskReadParquetKwargs)
    write: DaskWriteParquetKwargs = field(default_factory=DaskWriteParquetKwargs)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class IOKwargs(DataClass):
    pandas_csv: PandasCSVKwargs = field(default_factory=PandasCSVKwargs)
    pandas_parquet: PandasParquetKwargs = field(default_factory=PandasParquetKwargs)
    pandas_stata: PandasStataKwargs = field(default_factory=PandasStataKwargs)
    dask_csv: DaskCSVKwargs = field(default_factory=DaskCSVKwargs)
    dask_parquet: DaskParquetKwargs = field(default_factory=DaskParquetKwargs)
