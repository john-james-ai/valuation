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
# Modified   : Tuesday October 14th 2025 10:08:50 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import codecs
import csv
import io
import json
import os
from pathlib import Path
import pickle
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast
import zipfile

from loguru import logger
import pandas as pd
import yaml

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, unused-argument
# ------------------------------------------------------------------------------------------------ #

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
        sep: str = ",",
        header: Union[int, None] = 0,
        index_col: Union[int, str] = None,
        usecols: List[str] = None,
        escapechar: str = None,
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
            escapechar=escapechar,
            low_memory=low_memory,
            encoding=encoding,
        )

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: pd.DataFrame,
        sep: str = ",",
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


class ParquetIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> Any:
        """Reads using pandas API."""
        return pd.read_parquet(path=filepath)

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Writes a parquet file using pandas API."""
        data.to_parquet(path=filepath)


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
