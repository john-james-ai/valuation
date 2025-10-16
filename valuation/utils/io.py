#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 04:41:21 pm                                              #
# Modified   : Wednesday October 15th 2025 08:58:44 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module providing I/O services for various file formats."""
from __future__ import annotations

from abc import ABC, abstractmethod
import codecs
import csv
from dataclasses import asdict, dataclass, field
import io
import json
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast
import zipfile

import dask.dataframe as dd
from loguru import logger
import pandas as pd
import yaml

from valuation.utils.data import DataClass

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
class IOKwargs(DataClass):
    read: ReadKwargs = field(default_factory=ReadKwargs)
    write: WriteKwargs = field(default_factory=WriteKwargs)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TabularIOKwargs(IOKwargs):
    engine: str = "pandas"


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
class PandasStataKwargs(IOKwargs):
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
    thousands: Optional[str] = (
        None  # CHANGED: Defaulting to ',' is not safe for international data.
    )
    lineterminator: Optional[str] = "\n"
    low_memory: bool = False
    encoding: str = "utf-8"
    on_bad_lines: str = "warn"  # CHANGED: Replaces deprecated error_bad_lines and warn_bad_lines.
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
class PandasCSVKwargs(IOKwargs):
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
class PandasParquetKwargs(IOKwargs):
    read: PandasReadParquetKwargs = field(default_factory=PandasReadParquetKwargs)
    write: PandasWriteParquetKwargs = field(default_factory=PandasWriteParquetKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                          DASK CSV KWARGS                                         #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskReadCSVKwargs(ReadKwargs):
    blocksize: Optional[Union[str, int]] = "64MB"
    assume_missing: bool = False

    @property
    def read_kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(PandasReadCSVKwargs())
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

    @property
    def write_kwargs(self) -> Dict[str, Any]:
        kwargs = asdict(PandasWriteCSVKwargs())
        kwargs.update(asdict(self))  # Merge with pandas kwargs
        return kwargs


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DaskCSVKwargs(IOKwargs):
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
class DaskParquetKwargs(IOKwargs):
    read: DaskReadParquetKwargs = field(default_factory=DaskReadParquetKwargs)
    write: DaskWriteParquetKwargs = field(default_factory=DaskWriteParquetKwargs)


# ------------------------------------------------------------------------------------------------ #
#                                           IO                                                     #
# ------------------------------------------------------------------------------------------------ #


class IO(ABC):  # pragma: no cover

    @classmethod
    def read(cls, filepath: str, *args, **kwargs) -> Any:
        data = cls._read(filepath, **kwargs)
        return data

    @classmethod
    @abstractmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        pass

    @classmethod
    def write(cls, filepath: str, data: Any, *args, **kwargs) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        cls._write(filepath, data, **kwargs)

    @classmethod
    @abstractmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        pass


# ------------------------------------------------------------------------------------------------ #
#                                      ZIPFILE IO                                                  #
# ------------------------------------------------------------------------------------------------ #
class ZipFileIO(IO):  # pragma: no cover

    # Valid kwargs for read and write methods of the ZipFileIO class for pandas as of version 2.3.3
    VALID_KWARGS: Dict[str, Dict[str, Set[str]]] = {
        "csv": {
            # Valid arguments for pandas.read_csv (expanded for performance/data selection)
            "read": {
                "sep",
                "header",
                "names",
                "index_col",
                "dtype",
                "parse_dates",
                "encoding",
                "usecols",
                "skiprows",
                "nrows",
                "na_values",
                "engine",
                "on_bad_lines",  # Added crucial performance/data quality parameters
            },
            # Valid arguments for pandas.to_csv (expanded for formatting control)
            "write": {
                "sep",
                "header",
                "index",
                "encoding",
                "compression",
                "date_format",
                "float_format",
                "index_label",
                "columns",  # Added crucial formatting parameters
            },
        },
        "dta": {
            # Valid arguments for pandas.read_stata (expanded for categorical data)
            "read": {
                "index_col",
                "convert_dates",
                "convert_missing",
                "preserve_dtypes",
                "convert_categoricals",
                "columns",
                "chunksize",  # Added key Stata parameters
            },
            # Valid arguments for pandas.to_stata (expanded for metadata and types)
            "write": {
                "time_stamp",
                "write_index",
                "data_label",
                "version",
                "convert_dates",
                "variable_labels",
                "value_labels",  # Added key Stata metadata/type parameters
            },
        },
    }

    # Mapping extensions to their corresponding pandas reader functions
    READERS: Dict[str, Callable] = {
        ".csv": pd.read_csv,
        ".dta": pd.read_stata,
    }

    # Mapping extensions to their corresponding pandas DataFrame writer method names
    WRITERS: Dict[str, str] = {
        ".csv": "to_csv",
        ".dta": "to_stata",
    }

    @classmethod
    def _read(
        cls,
        filepath: str,
        **kwargs,
    ) -> pd.DataFrame:
        output = []
        unsupported_files = []

        with zipfile.ZipFile(filepath, "r") as zip_ref:

            # Iterate through the files in the zip reading from files with supported filetypes
            for internal_filepath in zip_ref.namelist():
                # Ensure filetype is supported
                ext = os.path.splitext(internal_filepath)[1].lower()
                if ext not in cls.READERS:
                    unsupported_files.append(internal_filepath)
                else:
                    # Obtain kwargs for the specific file extension reader
                    read_kwargs = cls._build_kwargs(
                        all_kwargs=kwargs, file_extension=ext, operation="read"
                    )

                    # Obtain the appropriate reader function
                    reader = cls.READERS[ext]
                    # Open the file from the archive as a file-like object
                    with zip_ref.open(internal_filepath) as internal_file:
                        df = reader(internal_file, **read_kwargs)
                        output.append(df)
        if len(unsupported_files) > 0:
            unsupported_files_list = "\n".join(unsupported_files)
            msg = f"The following {len(unsupported_files)} unsupported files were skipped:\n {unsupported_files_list}"
            logger.warning(msg)
        return pd.concat(output, ignore_index=True)

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        internal_filepath: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> None:

        # Set default internal_filepath if not provided
        internal_filepath = str(internal_filepath) if internal_filepath else None
        internal_filepath = (
            internal_filepath or f"{os.path.splitext(os.path.basename(filepath))[0]}.csv"
        )

        # Confirm the internal_filepath has a supported extension
        internal_ext = os.path.splitext(internal_filepath)[1].lower()
        if internal_ext not in cls.WRITERS.keys():
            msg = f"File extension {internal_ext} is not supported for writing in ZipFileIO."
            logger.error(msg)
            raise ValueError(msg)

        # Get the appropriate write kwargs for the file extension
        write_kwargs = cls._build_kwargs(
            all_kwargs=kwargs, file_extension=internal_ext, operation="write"
        )

        # Choose the appropriate in-memory buffer
        is_binary = internal_ext not in (".csv", ".json", ".txt")
        buffer_class = io.BytesIO if is_binary else io.StringIO

        writer_method_name = cls.WRITERS[internal_ext]
        writer_method = getattr(data, writer_method_name)

        # Write the DataFrame to the in-memory buffer
        try:
            with buffer_class() as buffer:
                writer_method(buffer, **write_kwargs)
                buffer_content = buffer.getvalue()

                # Ensure content is bytes for zip.writestr
                if not is_binary and isinstance(buffer_content, str):
                    buffer_content = buffer_content.encode("utf-8")

                # Write the buffer content to the ZIP archive in **APPEND** mode
                # mode='a' enables appending to existing zipfiles
                with zipfile.ZipFile(filepath, "a", compression=zipfile.ZIP_DEFLATED) as zip_ref:
                    zip_ref.writestr(internal_filepath, buffer_content)
        except Exception as e:
            logger.error(f"An error occurred while writing to zip file {filepath}: {e}")
            raise

    @classmethod
    def _build_kwargs(
        cls, all_kwargs: Dict[str, Any], file_extension: str, operation: str = "read"
    ) -> Dict[str, Any]:
        """
        Filters a dictionary of all passed keyword arguments down to only
        those supported by the target pandas reader/writer function for a given
        extension and operation ('read' or 'write').
        """
        # 1. Get the dictionary for the specific file_extension (e.g., {'read': {...}, 'write': {...}})
        op_kwargs = cls.VALID_KWARGS.get(file_extension, {})

        # 2. Get the set of valid keys for the specific operation (e.g., 'read')
        valid_keys = op_kwargs.get(operation, set())

        # Use dictionary comprehension to select only the valid keys
        filtered_kwargs = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        if file_extension == ".csv" and operation == "read":
            # Special handling for 'on_bad_lines' to ensure performance
            if "on_bad_lines" not in filtered_kwargs:
                filtered_kwargs["on_bad_lines"] = "skip"  # Default to 'skip' for performance
            if "engine" not in filtered_kwargs:
                filtered_kwargs["engine"] = "c"  # Default to 'c' engine for performance
            if "low_memory" not in filtered_kwargs:
                filtered_kwargs["low_memory"] = False  # Default to False for mixed types

        return filtered_kwargs


# ------------------------------------------------------------------------------------------------ #
#                                         STATA IO                                                 #
# ------------------------------------------------------------------------------------------------ #


class StataIO(IO):  # pragma: no cover
    """A class for handling I/O operations for Stata files."""

    @classmethod
    def _read(
        cls,
        filepath: Union[Path, str],
        columns: List[str] = None,
        index_col: str = None,
        convert_dates: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """Reads a Stata file into a pandas DataFrame."""
        data = pd.read_stata(
            filepath_or_buffer=filepath,
            columns=columns,
            index_col=index_col,
            convert_dates=convert_dates,
            **kwargs,
        )

        return cast(pd.DataFrame, data)

    @classmethod
    def _write(
        cls,
        filepath: Union[Path, str],
        data: pd.DataFrame,
        write_index: bool = False,
        version: int = 114,
        **kwargs,
    ) -> None:
        """Writes a pandas DataFrame to a Stata file."""
        data.to_stata(
            filepath_or_buffer=filepath,  # type: ignore
            write_index=write_index,
            version=version,
            **kwargs,
        )


# ------------------------------------------------------------------------------------------------ #
#                                         EXCEL IO                                                 #
# ------------------------------------------------------------------------------------------------ #


class ExcelIO(IO):  # pragma: no cover
    @classmethod
    def _read(
        cls,
        filepath: str,
        sheet_name: Union[str, int, list, None] = 0,
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        data = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            header=header,
            index_col=index_col,
            usecols=usecols,
            **kwargs,
        )

        return cast(pd.DataFrame, data)

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sheet_name: str = "Sheet1",
        columns: Union[str, list] = None,
        header: Union[bool, list] = True,
        index: bool = False,
        **kwargs,
    ) -> None:
        data.to_excel(
            excel_writer=filepath,
            sheet_name=sheet_name,
            columns=columns,
            header=header,
            index=index,
            **kwargs,
        )


# ------------------------------------------------------------------------------------------------ #
#                                        CSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class CSVIO(IO):  # pragma: no cover
    @classmethod
    def _read(
        cls,
        filepath: str,
        **kwargs,
    ) -> pd.DataFrame:
        return pd.read_csv(filepath, **kwargs)

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        **kwargs,
    ) -> None:
        data.to_csv(
            filepath,
            **kwargs,
        )


# ------------------------------------------------------------------------------------------------ #
#                                        TSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #


class TSVIO(IO):  # pragma: no cover
    @classmethod
    def _read(
        cls,
        filepath: str,
        sep: str = "\t",
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        low_memory: bool = False,
        encoding: str = "utf-8",
        **kwargs,
    ) -> pd.DataFrame:
        return pd.read_csv(
            filepath,
            sep=sep,
            header=header,
            index_col=index_col,
            usecols=usecols,
            low_memory=low_memory,
            encoding=encoding,
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sep: str = "\t",
        index: bool = False,
        index_label: str = None,
        encoding: str = "utf-8",
        **kwargs,
    ) -> None:
        data.to_csv(
            filepath,
            sep=sep,
            index=index,
            index_label=index_label,
            encoding=encoding,
            escapechar="\\",
            quoting=csv.QUOTE_NONE,
        )


# ------------------------------------------------------------------------------------------------ #
#                                        YAML IO                                                   #
# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> dict:
        with open(filepath, "r", encoding="utf-8") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:  # pragma: no cover
                logger.exception(e)
                raise IOError(e) from e
            finally:
                f.close()

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            try:
                yaml.dump(data, f)
            except yaml.YAMLError as e:  # pragma: no cover
                logger.exception(e)
                raise IOError(e) from e
            finally:
                f.close()


# ------------------------------------------------------------------------------------------------ #
#                                         PICKLE                                                   #
# ------------------------------------------------------------------------------------------------ #


class PickleIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        with open(filepath, "rb") as f:
            try:
                return pickle.load(f)
            except pickle.PickleError as e:  # pragma: no cover
                logger.exception(e)
                raise IOError(e) from e
            finally:
                f.close()

    @classmethod
    def _write(cls, filepath: str, data: Any, write_mode: str = "wb", **kwargs) -> None:
        # Note, "a+" write_mode for append. If <TypeError: write() argument must be str, not bytes>
        # use "ab+"
        with open(filepath, write_mode) as f:
            try:
                pickle.dump(data, f)
            except pickle.PickleError as e:  # pragma: no cover
                logger.exception(e)
                raise (e)
            finally:
                f.close()


# ------------------------------------------------------------------------------------------------ #
#                                         PARQUET                                                  #
# ------------------------------------------------------------------------------------------------ #


class ParquetIO(IO):
    """Handles reading and writing Parquet files using pandas or Dask.

    This class provides a unified interface to read Parquet files into either
    an in-memory pandas DataFrame for smaller datasets or a lazy Dask
    DataFrame for larger-than-memory datasets.
    """

    @classmethod
    def _read(cls, filepath: str, **kwargs) -> pd.DataFrame:
        """Reads a Parquet file into a pandas or Dask DataFrame.

        Args:
            filepath: The path to the Parquet file or directory.
            **kwargs: Additional keyword arguments passed to the underlying
                      read_parquet function.

        Returns:
            A pandas DataFrame.
        """
        return pd.read_parquet(path=filepath, **kwargs)

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Writes a pandas or Dask DataFrame to a Parquet file.

        The underlying method handles both types correctly. For Dask, this
        will trigger the computation graph and write the results to disk.

        Args:
            filepath: The path to the output Parquet file or directory.
            data: The DataFrame (pandas or Dask) to write.
            **kwargs: Additional keyword arguments passed to the underlying
                      to_parquet method.
        """
        data.to_parquet(path=filepath, **kwargs)


# ------------------------------------------------------------------------------------------------ #
#                                           HTML                                                   #
# ------------------------------------------------------------------------------------------------ #


class HtmlIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Read the raw html."""
        file = codecs.open(filename=filepath, encoding="utf-8")
        return file.read()

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Converts Pandas DataFrame to a pyarrow table, then persists."""
        raise NotImplementedError


# ------------------------------------------------------------------------------------------------ #
#                                          JSON                                                    #
# ------------------------------------------------------------------------------------------------ #


class JsonIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Read the parsed dictionary from a json file."""
        with open(filepath, encoding="utf-8") as json_file:
            return json.load(json_file)

    @classmethod
    def _write(cls, filepath: str, data: dict, **kwargs) -> None:
        """Writes a dictionary to a json file."""
        with open(filepath, "w", encoding="utf-8") as json_file:
            if isinstance(data, list):
                for datum in data:
                    if isinstance(datum, dict):
                        json.dump(datum, json_file, indent=2)
                    else:
                        msg = "JsonIO supports dictionaries and lists of dictionaries only."
                        logger.exception(msg)
                        raise ValueError(msg)
            else:
                try:
                    json.dump(data, json_file, indent=2)
                except json.JSONDecodeError as e:
                    logger.exception(f"Exception of type {type(e)} occurred.\n{e}")
                    raise


# ------------------------------------------------------------------------------------------------ #
#                                       IO SERVICE                                                 #
# ------------------------------------------------------------------------------------------------ #
class IOService:  # pragma: no cover
    __io = {
        "html": HtmlIO,
        "dat": CSVIO,
        "csv": CSVIO,
        "tsv": TSVIO,
        "yaml": YamlIO,
        "yml": YamlIO,
        "json": JsonIO,
        "pkl": PickleIO,
        "pickle": PickleIO,
        "xlsx": ExcelIO,
        "xls": ExcelIO,
        "parquet": ParquetIO,
        "zip": ZipFileIO,
    }

    @classmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        io = cls._get_io(filepath)
        return io.read(filepath, **kwargs)

    @classmethod
    def write(cls, filepath: str, data: Any, **kwargs) -> None:
        io = cls._get_io(filepath)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        io.write(filepath=filepath, data=data, **kwargs)

    @classmethod
    def _get_io(cls, filepath: str) -> IO:
        file_format = os.path.splitext(filepath)[-1].replace(".", "")
        try:
            return IOService.__io[file_format]
        except TypeError as exc:
            if filepath is None:
                msg = "Filepath is None"
                logger.exception(msg)
                raise ValueError(msg) from exc
            raise
        except KeyError as exc:
            msg = "File type {} is not supported.".format(file_format)
            logger.exception(msg)
            raise ValueError(msg) from exc


# ------------------------------------------------------------------------------------------------ #
#                                    DaskIOService                                                 #
# ------------------------------------------------------------------------------------------------ #
class DaskIOService(IOService):  # pragma: no cover
    __io = {
        "html": HtmlIO,
        "dat": CSVIO,
        "csv": CSVIO,
        "tsv": TSVIO,
        "yaml": YamlIO,
        "yml": YamlIO,
        "json": JsonIO,
        "pkl": PickleIO,
        "pickle": PickleIO,
        "xlsx": ExcelIO,
        "xls": ExcelIO,
        "parquet": ParquetIO,
        "zip": ZipFileIO,
    }

    @classmethod
    def read(cls, filepath: str, **kwargs) -> Any:
        io = cls._get_io(filepath)
        if io == ParquetIO:
            return dd.read_parquet(filepath, **kwargs)
        return io.read(filepath, **kwargs)
