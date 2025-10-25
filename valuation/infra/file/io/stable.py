#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/io.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 05:59:08 pm                                              #
# Modified   : Tuesday October 21st 2025 08:46:37 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Callable, Dict, List, Optional, Set, Union, cast

from abc import ABC, abstractmethod
import codecs
import csv
import io
import json
import os
from pathlib import Path
import pickle
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
        mode: str = "w",
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

                with zipfile.ZipFile(str(filepath), mode=mode, compression=zipfile.ZIP_DEFLATED) as zip_ref:  # type: ignore
                    zip_ref.writestr(internal_filepath, buffer_content)
        except Exception as e:
            logger.error(f"An error occurred while writing to zip file {filepath}: {e}")
            raise

    @classmethod
    def _build_kwargs(
        cls, all_kwargs: Dict[str, Any], file_extension: str, operation: str = "read"
    ) -> Dict[str, Any]:
        """
        Filter to only supported kwargs for the target pandas reader/writer and
        inject safe, performant defaults per (ext, op).
        """
        ext = file_extension.lower()

        # 1) Allow only supported keys
        op_map = cls.VALID_KWARGS.get(ext, {})
        valid_keys = op_map.get(operation, set())
        filtered = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        # 2) Inject format-specific defaults
        if ext == ".csv":
            if operation == "read":
                # fast & robust CSV reads
                filtered.setdefault("on_bad_lines", "skip")
                filtered.setdefault("engine", "c")
                filtered.setdefault("low_memory", False)
                # Optional modern dtypes (pandas >=2.0)
                filtered.setdefault("dtype_backend", "pyarrow")  # or "numpy_nullable"
            else:  # write
                filtered.setdefault("index", False)  # never persist index to CSV
                # Optional: deterministic quoting
                # filtered.setdefault("quoting", csv.QUOTE_MINIMAL)
        elif ext == ".json":
            if operation == "read":
                # Optional modern dtypes
                filtered.setdefault("dtype_backend", "pyarrow")
                # If you expect records-orient a lot:
                # filtered.setdefault("orient", "records")
                pass
            else:  # write
                filtered.setdefault("index", False)
                # filtered.setdefault("orient", "records")  # if that’s your house style
        elif ext == ".parquet":
            # Prefer pyarrow across read/write; most stable with BytesIO
            filtered.setdefault("engine", "pyarrow")
            if operation == "write":
                filtered.setdefault("index", False)
                # Optional: compression defaults (pyarrow handles inside the payload, not zip)
                # filtered.setdefault("compression", "zstd")  # good balance; needs pyarrow built with it
        elif ext in (".feather", ".ipc"):  # Arrow IPC/Feather
            filtered.setdefault("compression", "zstd")  # if supported in your build
            # Feather doesn’t store index; nothing to add for write
        elif ext == ".txt":
            if operation == "write":
                filtered.setdefault("index", False)
        else:
            # Unknown ext: leave filtered as-is; caller has already validated support upstream
            pass

        return filtered


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
        df = pd.read_parquet(filepath, **kwargs)

        # Drop old index artifacts (__index_level_0__, Unnamed, etc.)
        df = df.loc[:, ~df.columns.str.match(r"^(Unnamed:|__index_level__)")]

        # Ensure no accidental MultiIndex columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = ["_".join(map(str, col)).strip() for col in df.columns.values]

        # Trim whitespace from column names (sometimes sneaks in)
        df.columns = df.columns.str.strip()
        return df

    @classmethod
    def _write(cls, filepath: str, data: pd.DataFrame, **kwargs) -> None:
        """Writes a parquet file using pandas API."""
        if data.index.name or not data.index.equals(pd.RangeIndex(len(data))):
            data = data.reset_index(drop=True)
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
