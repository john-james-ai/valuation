#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/fastio.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 05:59:08 pm                                              #
# Modified   : Saturday October 25th 2025 04:50:27 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Callable, Dict, List, Optional, Set, Union

from abc import ABC, abstractmethod
import io
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile

from loguru import logger
import polars as pl
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

    # Valid kwargs for scan and write methods of Polars
    VALID_KWARGS: Dict[str, Dict[str, Set[str]]] = {
        "csv": {
            "read": {
                "separator",
                "has_header",
                "columns",
                "new_columns",
                "dtypes",
                "skip_rows",
                "n_rows",
                "encoding",
                "null_values",
                "missing_utf8_is_empty_string",
                "ignore_errors",
                "try_parse_dates",
                "infer_schema_length",
                "batch_size",
                "rechunk",
                "low_memory",
            },
            "write": {
                "separator",
                "include_header",
                "date_format",
                "datetime_format",
                "float_precision",
                "null_value",
                "quote_char",
                "quote_style",
                "line_terminator",
            },
        },
        "parquet": {
            "read": {
                "columns",
                "n_rows",
                "parallel",
                "row_count_name",
                "row_count_offset",
                "low_memory",
                "rechunk",
            },
            "write": {
                "compression",
                "compression_level",
                "statistics",
                "row_group_size",
                "data_page_size",
            },
        },
        "json": {
            "read": {
                "infer_schema_length",
                "batch_size",
                "ignore_errors",
            },
            "write": {
                "pretty",
                "row_oriented",
            },
        },
        "ndjson": {
            "read": {
                "infer_schema_length",
                "batch_size",
                "ignore_errors",
                "low_memory",
            },
            "write": {},
        },
        "ipc": {
            "read": {
                "columns",
                "n_rows",
                "memory_map",
                "rechunk",
            },
            "write": {
                "compression",
            },
        },
    }

    # Mapping extensions to their corresponding Polars scan functions
    LAZY_READERS: Dict[str, Callable] = {
        ".csv": "scan_csv",
        ".parquet": "scan_parquet",
        ".json": "scan_ndjson",
        ".ndjson": "scan_ndjson",
        ".ipc": "scan_ipc",
    }

    # Mapping extensions to their corresponding Polars write method names
    WRITERS: Dict[str, Callable] = {
        ".csv": "write_csv",
        ".parquet": "write_parquet",
        ".json": "write_json",
        ".ndjson": "write_ndjson",
        ".ipc": "write_ipc",
    }

    @classmethod
    def _read(
        cls,
        filepath: str,
        lazy: bool = True,
        **kwargs,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Read files from a zip archive and return a LazyFrame or DataFrame.

        Args:
            filepath: Path to the zip file

            lazy: If True, return LazyFrame; if False, collect to DataFrame

            **kwargs: Additional arguments passed to Polars readers

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: Combined data from all supported files
        """
        lazy_frames = []
        temp_files: List[Path] = []
        unsupported_files = []
        logger.debug(f"Reading fastio zip file: {filepath}")

        # Create a unique temporary directory for this extraction
        temp_dir = Path(tempfile.mkdtemp(prefix="fastio_zip_"))

        try:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                for internal_filepath in zip_ref.namelist():
                    ext = os.path.splitext(internal_filepath)[1].lower()

                    if ext not in cls.LAZY_READERS:
                        unsupported_files.append(internal_filepath)
                        continue

                    # Extract file temporarily for lazy reading
                    temp_file = temp_dir / Path(internal_filepath).name
                    temp_file.parent.mkdir(parents=True, exist_ok=True)
                    with zip_ref.open(internal_filepath) as source:
                        temp_file.write_bytes(source.read())

                    # Keep track of temp files for later cleanup (if needed)
                    temp_files.append(temp_file)

                    # Build kwargs for the specific reader
                    read_kwargs = cls._build_kwargs(
                        all_kwargs=kwargs, file_extension=ext, operation="read"
                    )

                    # Get the appropriate lazy reader
                    reader_name = cls.LAZY_READERS[ext]
                    reader = getattr(pl, reader_name)

                    try:
                        lf = reader(str(temp_file), **read_kwargs)
                        lazy_frames.append(lf)
                    except Exception as e:
                        logger.error(f"Error reading {internal_filepath}: {e}")
                        # Continue processing remaining files

            if unsupported_files:
                unsupported_list = "\n".join(unsupported_files)
                logger.warning(
                    f"Skipped {len(unsupported_files)} unsupported files:\n{unsupported_list}"
                )

            if not lazy_frames:
                # No frames parsed — clean up extracted files and directory
                for tf in temp_files:
                    try:
                        tf.unlink(missing_ok=True)
                    except Exception:
                        pass
                try:
                    shutil.rmtree(temp_dir)
                except Exception:
                    pass
                return pl.LazyFrame() if lazy else pl.DataFrame()

            # Concatenate all lazy frames
            combined = pl.concat(lazy_frames, how="vertical_relaxed")

            if not lazy:
                # Eager: collect now and then clean up the temp files
                try:
                    result = combined.collect()
                finally:
                    for tf in temp_files:
                        try:
                            tf.unlink(missing_ok=True)
                        except Exception:
                            pass
                    try:
                        shutil.rmtree(temp_dir)
                    except Exception:
                        pass
                return result

            # lazy == True: Keep temp files available until the caller collects.
            logger.debug(f"Returning LazyFrame referencing temporary files at: {temp_dir}")
            return combined

        except Exception as e:
            # On unexpected error, attempt cleanup then re-raise
            for tf in temp_files:
                try:
                    tf.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass
            raise e

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: Union[pl.DataFrame, pl.LazyFrame],
        mode: str = "w",
        internal_filepath: Optional[Union[Path, str]] = None,
        **kwargs,
    ) -> None:
        """
        Write a Polars DataFrame or LazyFrame to a zip archive.

        Args:
            filepath: Path to the zip file
            data: Polars DataFrame or LazyFrame to write
            mode: Zip file mode ('w' or 'a')
            internal_filepath: Name of file inside zip (with extension)
            **kwargs: Additional arguments passed to Polars writers
        """
        logger.debug(f"Writing fastio zip file: {filepath}")
        # Collect LazyFrame if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Set default internal filepath
        internal_filepath = str(internal_filepath) if internal_filepath else None
        internal_filepath = (
            internal_filepath or f"{os.path.splitext(os.path.basename(filepath))[0]}.csv"
        )

        # Validate extension
        internal_ext = os.path.splitext(internal_filepath)[1].lower()
        if internal_ext not in cls.WRITERS:
            raise ValueError(
                f"File extension {internal_ext} is not supported. "
                f"Supported: {list(cls.WRITERS.keys())}"
            )

        # Build write kwargs
        write_kwargs = cls._build_kwargs(
            all_kwargs=kwargs, file_extension=internal_ext, operation="write"
        )

        # Get the writer method
        writer_method_name = cls.WRITERS[internal_ext]
        writer_method = getattr(data, writer_method_name)

        # Write to in-memory buffer
        try:
            buffer = io.BytesIO()
            writer_method(buffer, **write_kwargs)
            buffer_content = buffer.getvalue()

            # Use positional arguments to satisfy type-checkers that expect this overload
            with zipfile.ZipFile(filepath, mode, zipfile.ZIP_DEFLATED) as zip_ref:  # type: ignore
                zip_ref.writestr(internal_filepath, buffer_content)

        except Exception as e:
            raise RuntimeError(f"Error writing to zip file {filepath}: {e}") from e

    @classmethod
    def _convert_dtype_value(cls, v: Any) -> Any:
        """Normalize a single dtype value to a Polars dtype if expressed as string.

        Supports common names like 'string', 'float64', 'int64', 'datetime64[ns]', 'bool', etc.
        Leaves values unchanged if already a Polars DataType or unrecognized (caller may handle).
        """
        if v is None:
            return v
        # Already a Polars dtype
        if isinstance(v, pl.DataType):
            return v
        # Strings mapping
        if isinstance(v, str):
            s = v.strip().lower()
            if s in ("string", "str", "utf8", "utf-8"):
                return pl.Utf8
            if s in ("float64", "float", "double"):
                return pl.Float64
            if s in ("int64", "int", "int_64", "i64"):
                return pl.Int64
            if s in ("int32", "i32"):
                return pl.Int32
            if s in ("bool", "boolean"):
                return pl.Boolean
            if s in ("datetime64[ns]", "datetime", "datetime64"):
                return pl.Datetime("ns")
            # Fallback: return original string (Polars may accept some names)
            return v
        # If it's numpy/pandas dtype object, try to map its name
        try:
            dtype_name = getattr(v, "name", None)
            if isinstance(dtype_name, str):
                return cls._convert_dtype_value(dtype_name)
        except Exception:
            pass
        return v

    @classmethod
    def _convert_dtypes_dict(cls, dtypes: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a dict of column->dtype values to use Polars dtypes where possible."""
        if not isinstance(dtypes, dict):
            return dtypes
        return {col: cls._convert_dtype_value(dt) for col, dt in dtypes.items()}

    @classmethod
    def _build_kwargs(
        cls,
        all_kwargs: Dict[str, Any],
        file_extension: str,
        operation: str = "read",
    ) -> Dict[str, Any]:
        """
        Filter kwargs to only those supported by the target Polars reader/writer
        and inject safe, performant defaults.

        Args:
            all_kwargs: All provided kwargs
            file_extension: File extension (e.g., '.csv')
            operation: Either 'read' or 'write'

        Returns:
            Dict[str, Any]: Filtered and enhanced kwargs
        """
        ext = file_extension.lower()

        # Normalize extension for lookup
        format_key = ext[1:] if ext.startswith(".") else ext
        if ext in (".json", ".ndjson"):
            format_key = "json" if ext == ".json" else "ndjson"

        # Get valid keys for this format and operation
        op_map = cls.VALID_KWARGS.get(format_key, {})
        valid_keys = op_map.get(operation, set())
        filtered = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        # If the caller provided a dtypes mapping, normalize values to Polars dtypes
        if "dtypes" in filtered and isinstance(filtered["dtypes"], dict):
            filtered["dtypes"] = cls._convert_dtypes_dict(filtered["dtypes"])

        # Inject format-specific defaults
        if ext == ".csv":
            if operation == "read":
                filtered.setdefault("ignore_errors", True)
                filtered.setdefault("try_parse_dates", True)
                filtered.setdefault("rechunk", True)
            else:  # write
                filtered.setdefault("include_header", True)

        elif ext == ".parquet":
            if operation == "read":
                filtered.setdefault("parallel", "auto")
                filtered.setdefault("rechunk", True)
            else:  # write
                filtered.setdefault("compression", "zstd")
                filtered.setdefault("statistics", True)

        elif ext in (".json", ".ndjson"):
            if operation == "read":
                filtered.setdefault("ignore_errors", True)
                filtered.setdefault("infer_schema_length", 10000)

        elif ext == ".ipc":
            if operation == "read":
                filtered.setdefault("rechunk", True)
            else:  # write
                filtered.setdefault("compression", "zstd")

        return filtered


# ------------------------------------------------------------------------------------------------ #
#                                        CSV IO                                                    #
# ------------------------------------------------------------------------------------------------ #
class CSVIO(IO):  # pragma: no cover
    """
    CSV IO handler using Polars with lazy loading support.
    """

    @classmethod
    def _read(
        cls,
        filepath: str,
        separator: str = ",",
        has_header: bool = True,
        columns: Optional[List[str]] = None,
        skip_rows: int = 0,
        n_rows: Optional[int] = None,
        encoding: str = "utf8",
        null_values: Optional[Union[str, List[str]]] = None,
        ignore_errors: bool = False,
        try_parse_dates: bool = True,
        infer_schema_length: int = 100,
        rechunk: bool = True,
        low_memory: bool = False,
        lazy: bool = True,
        **kwargs,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Read CSV file using Polars.

        Args:
            filepath: Path to CSV file
            separator: Field delimiter (default: ",")
            has_header: Whether first row contains column names
            columns: Columns to read (None = all columns)
            skip_rows: Number of rows to skip at start
            n_rows: Maximum number of rows to read
            encoding: File encoding (default: "utf8")
            null_values: String(s) to interpret as null
            ignore_errors: Continue on parse errors
            try_parse_dates: Attempt to parse dates automatically
            infer_schema_length: Rows to use for schema inference (0 = all rows)
            rechunk: Rechunk to contiguous memory after reading
            low_memory: Reduce memory usage at cost of performance
            lazy: If True, return LazyFrame; if False, return DataFrame
            **kwargs: Additional arguments passed to scan_csv/read_csv

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: Loaded data
        """
        read_kwargs = {
            "separator": separator,
            "has_header": has_header,
            "skip_rows": skip_rows,
            "encoding": encoding,
            "ignore_errors": ignore_errors,
            "try_parse_dates": try_parse_dates,
            "rechunk": rechunk,
            "low_memory": low_memory,
        }
        logger.debug(f"Reading fastio csv file: {filepath}")

        # Add optional parameters
        if columns is not None:
            read_kwargs["columns"] = columns
        if n_rows is not None:
            read_kwargs["n_rows"] = n_rows
        if null_values is not None:
            read_kwargs["null_values"] = null_values
        if infer_schema_length != 100:  # Only set if non-default
            read_kwargs["infer_schema_length"] = infer_schema_length

        # Merge any additional kwargs
        read_kwargs.update(kwargs)

        if lazy:
            # Use lazy reading for query optimization
            return pl.scan_csv(filepath, **read_kwargs)
        else:
            # Eager reading
            return pl.read_csv(filepath, **read_kwargs)

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: Union[pl.DataFrame, pl.LazyFrame],
        separator: str = ",",
        include_header: bool = True,
        quote_char: str = '"',
        quote_style: str = "necessary",
        null_value: str = "",
        datetime_format: Optional[str] = None,
        date_format: Optional[str] = None,
        float_precision: Optional[int] = None,
        line_terminator: str = "\n",
        **kwargs,
    ) -> None:
        """
        Write DataFrame to CSV file using Polars.

        Args:
            filepath: Path to output CSV file
            data: Polars DataFrame or LazyFrame to write
            separator: Field delimiter (default: ",")
            include_header: Whether to write column names as first row
            quote_char: Character to use for quoting
            quote_style: When to quote fields ("necessary", "always", "non_numeric", "never")
            null_value: String to represent null values
            datetime_format: Format string for datetime columns
            date_format: Format string for date columns
            float_precision: Number of decimal places for floats
            line_terminator: Line ending character(s)
            **kwargs: Additional arguments passed to write_csv
        """
        logger.debug(f"Writing fastio csv file: {filepath}")
        # Collect LazyFrame if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        write_kwargs = {
            "separator": separator,
            "include_header": include_header,
            "quote_char": quote_char,
            "quote_style": quote_style,
            "null_value": null_value,
            "line_terminator": line_terminator,
        }

        # Add optional parameters
        if datetime_format is not None:
            write_kwargs["datetime_format"] = datetime_format
        if date_format is not None:
            write_kwargs["date_format"] = date_format
        if float_precision is not None:
            write_kwargs["float_precision"] = float_precision

        # Merge any additional kwargs
        write_kwargs.update(kwargs)

        data.write_csv(filepath, **write_kwargs)

    @classmethod
    def read_lazy(cls, filepath: str, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        """
        Convenience method to explicitly read as LazyFrame.

        Args:
            filepath: Path to CSV file
            **kwargs: Arguments passed to _read

        Returns:
            pl.LazyFrame: Lazy DataFrame for query optimization
        """
        kwargs["lazy"] = True
        return cls._read(filepath, **kwargs)

    @classmethod
    def read_eager(cls, filepath: str, **kwargs) -> pl.DataFrame | pl.LazyFrame:
        """
        Convenience method to explicitly read as DataFrame.

        Args:
            filepath: Path to CSV file
            **kwargs: Arguments passed to _read

        Returns:
            pl.DataFrame: Materialized DataFrame
        """
        kwargs["lazy"] = False
        return cls._read(filepath, **kwargs)


# ------------------------------------------------------------------------------------------------ #
#                                         PARQUET                                                  #
# ------------------------------------------------------------------------------------------------ #
class ParquetIO(IO):  # pragma: no cover
    """
    Parquet IO handler using Polars with lazy loading support.
    """

    @classmethod
    def _read(
        cls,
        filepath: str,
        lazy: bool = True,
        clean_columns: bool = True,
        **kwargs,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Read Parquet file using Polars.

        Args:
            filepath: Path to Parquet file (supports wildcards for multiple files)
            lazy: If True, return LazyFrame; if False, return DataFrame
            clean_columns: Remove index artifacts and clean column names
            **kwargs: Additional arguments (columns, n_rows, parallel, rechunk, etc.)

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: Loaded data with cleaned columns
        """
        logger.debug(f"Reading fastio parquet file: {filepath}")

        # Build kwargs with defaults
        read_kwargs = cls._build_kwargs(kwargs, operation="read")

        # Read the parquet file
        if lazy:
            data = pl.scan_parquet(filepath, **read_kwargs)
        else:
            data = pl.read_parquet(filepath, **read_kwargs)

        # Clean columns if requested
        if clean_columns:
            data = cls._clean_columns(data, lazy=lazy)

        return data

    @classmethod
    def _write(
        cls,
        filepath: str,
        data: Union[pl.DataFrame, pl.LazyFrame],
        **kwargs,
    ) -> None:
        """
        Write DataFrame to Parquet file using Polars.

        Args:
            filepath: Path to output Parquet file
            data: Polars DataFrame or LazyFrame to write
            **kwargs: Additional arguments (compression, compression_level, statistics, etc.)
        """
        logger.debug(f"Writing fastio parquet file: {filepath}")

        # Collect LazyFrame if needed
        if isinstance(data, pl.LazyFrame):
            data = data.collect()

        # Build kwargs with defaults
        write_kwargs = cls._build_kwargs(kwargs, operation="write")

        data.write_parquet(filepath, **write_kwargs)

    @classmethod
    def _build_kwargs(cls, all_kwargs: Dict[str, Any], operation: str = "read") -> Dict[str, Any]:
        """
        Filter to only supported kwargs for Polars parquet operations and
        inject safe, performant defaults.

        Args:
            all_kwargs: User-provided kwargs that may include unsupported options
            operation: Either "read" or "write"

        Returns:
            Filtered kwargs with sensible defaults
        """
        # Define valid kwargs for each operation
        valid_read_keys = {
            "columns",
            "n_rows",
            "parallel",
            "rechunk",
            "low_memory",
            "use_statistics",
            "hive_partitioning",
            "use_pyarrow",
            "memory_map",
            "storage_options",
            "row_index_name",
            "row_index_offset",
        }
        valid_write_keys = {
            "compression",
            "compression_level",
            "statistics",
            "row_group_size",
            "data_page_size",
            "use_pyarrow",
            "pyarrow_options",
        }

        # Filter to valid keys
        valid_keys = valid_read_keys if operation == "read" else valid_write_keys
        filtered = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        # Inject defaults
        if operation == "read":
            filtered.setdefault("rechunk", True)  # Contiguous memory
            filtered.setdefault("low_memory", False)  # Better performance
            filtered.setdefault("use_statistics", True)  # Predicate pushdown
            filtered.setdefault("hive_partitioning", True)  # Support partitioned data
            filtered.setdefault("parallel", "auto")  # Auto parallelization
        else:  # write
            filtered.setdefault("compression", "zstd")  # Good balance of speed/size
            filtered.setdefault("statistics", True)  # Enable column statistics

        return filtered

    @classmethod
    def _clean_columns(
        cls,
        data: Union[pl.LazyFrame, pl.DataFrame],
        lazy: bool = True,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Clean column names by removing index artifacts and trimming whitespace.

        Removes columns matching patterns:
        - __index_level_N__
        - Unnamed: N

        Also strips whitespace from column names.

        Args:
            data: DataFrame or LazyFrame to clean
            lazy: Whether data is a LazyFrame

        Returns:
            Cleaned DataFrame or LazyFrame
        """
        if lazy and isinstance(data, pl.LazyFrame):
            # For LazyFrame, we need to collect schema to inspect columns
            columns = data.collect_schema().names()
        else:
            columns = data.columns

        # Pattern to match index artifacts
        index_pattern = re.compile(r"^(Unnamed:|__index_level__)")

        # Filter out index artifact columns and strip whitespace
        cleaned_columns = [col.strip() for col in columns if not index_pattern.match(col)]

        # Select only the cleaned columns
        if lazy and isinstance(data, pl.LazyFrame):
            return data.select(cleaned_columns)
        else:
            return data.select(cleaned_columns)

    @classmethod
    def read_lazy(cls, filepath: str, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        """
        Convenience method to explicitly read as LazyFrame.

        Args:
            filepath: Path to Parquet file
            **kwargs: Arguments passed to _read

        Returns:
            pl.LazyFrame: Lazy DataFrame for query optimization
        """
        kwargs["lazy"] = True
        return cls._read(filepath, **kwargs)

    @classmethod
    def read_eager(cls, filepath: str, **kwargs) -> pl.LazyFrame | pl.DataFrame:
        """
        Convenience method to explicitly read as DataFrame.

        Args:
            filepath: Path to Parquet file
            **kwargs: Arguments passed to _read

        Returns:
            pl.DataFrame: Materialized DataFrame
        """
        kwargs["lazy"] = False
        return cls._read(filepath, **kwargs)

    @classmethod
    def read_partitioned(
        cls,
        directory: str,
        lazy: bool = True,
        **kwargs,
    ) -> Union[pl.LazyFrame, pl.DataFrame]:
        """
        Read Hive-partitioned Parquet files from a directory.

        Args:
            directory: Path to directory containing partitioned Parquet files
            lazy: If True, return LazyFrame; if False, return DataFrame
            **kwargs: Arguments passed to _read

        Returns:
            Union[pl.LazyFrame, pl.DataFrame]: Combined data from all partitions
        """
        # Use wildcard pattern to read all parquet files
        pattern = f"{directory}/**/*.parquet"
        kwargs["hive_partitioning"] = True
        kwargs["lazy"] = lazy
        return cls._read(pattern, **kwargs)


# ------------------------------------------------------------------------------------------------ #
#                                        YAML IO                                                   #
# ------------------------------------------------------------------------------------------------ #


class YamlIO(IO):  # pragma: no cover
    @classmethod
    def _read(cls, filepath: str, **kwargs) -> dict:
        logger.debug(f"Reading fastio yaml file: {filepath}")
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
        logger.debug(f"Writing fastio yaml file: {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            try:
                yaml.dump(data, f)
            except yaml.YAMLError as e:  # pragma: no cover
                logger.exception(e)
                raise IOError(e) from e
            finally:
                f.close()


from typing import Any, Dict, Union

from pathlib import Path

import polars as pl


class JsonIO(IO):  # pragma: no cover
    """
    JSON IO handler using Polars for DataFrame operations.
    Supports both structured JSON (for DataFrames) and raw JSON (for dicts/lists).
    """

    @classmethod
    def _read(
        cls, filepath: str, as_dataframe: bool = True, **kwargs
    ) -> Union[pl.DataFrame, dict, list]:
        """
        Read JSON file using Polars or standard json library.

        Args:
            filepath: Path to JSON file
            as_dataframe: If True, read as DataFrame; if False, read as dict/list
            **kwargs: Additional arguments passed to pl.read_json or _build_kwargs

        Returns:
            Union[pl.DataFrame, dict, list]: Parsed JSON data
        """
        logger.debug(f"Reading JSON file: {filepath}")

        if as_dataframe:
            # Use Polars to read structured JSON as DataFrame
            read_kwargs = cls._build_kwargs(kwargs, operation="read")
            return pl.read_json(filepath, **read_kwargs)
        else:
            # Use standard library for raw JSON (dicts/lists)
            import json

            with open(filepath, encoding="utf-8") as json_file:
                return json.load(json_file)

    @classmethod
    def _write(
        cls, filepath: str, data: Union[pl.DataFrame, pl.LazyFrame, dict, list], **kwargs
    ) -> None:
        """
        Write data to JSON file using Polars or standard json library.

        Args:
            filepath: Path to output JSON file
            data: Data to write (DataFrame, LazyFrame, dict, or list)
            **kwargs: Additional arguments passed to write_json or _build_kwargs
        """
        logger.debug(f"Writing JSON file: {filepath}")

        if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
            # Write DataFrame/LazyFrame using Polars
            if isinstance(data, pl.LazyFrame):
                data = data.collect()

            write_kwargs = cls._build_kwargs(kwargs, operation="write")
            data.write_json(filepath, **write_kwargs)

        elif isinstance(data, (dict, list)):
            # Write raw JSON using standard library
            import json

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
        else:
            raise TypeError(
                f"Unsupported data type: {type(data)}. Expected DataFrame, dict, or list."
            )

    @classmethod
    def _build_kwargs(cls, all_kwargs: Dict[str, Any], operation: str = "read") -> Dict[str, Any]:
        """
        Filter to only supported kwargs for Polars JSON operations and
        inject safe, performant defaults.

        Args:
            all_kwargs: User-provided kwargs that may include unsupported options
            operation: Either "read" or "write"

        Returns:
            Filtered kwargs with sensible defaults
        """
        # Define valid kwargs for each operation
        valid_read_keys = {
            "schema",
            "schema_overrides",
            "infer_schema_length",
        }
        valid_write_keys = {
            "pretty",
            "row_oriented",
        }

        # Filter to valid keys
        valid_keys = valid_read_keys if operation == "read" else valid_write_keys
        filtered = {k: v for k, v in all_kwargs.items() if k in valid_keys}

        # Inject defaults
        if operation == "read":
            filtered.setdefault("infer_schema_length", 10000)  # Better type inference
        else:  # write
            filtered.setdefault("pretty", False)  # Compact JSON by default
            filtered.setdefault("row_oriented", True)  # Records format (array of objects)

        return filtered

    @classmethod
    def read_dataframe(
        cls, filepath: str, **kwargs
    ) -> Union[dict, list] | pl.LazyFrame | pl.DataFrame:
        """
        Convenience method to explicitly read JSON as DataFrame.

        Args:
            filepath: Path to JSON file
            **kwargs: Arguments passed to _read

        Returns:
            pl.DataFrame: Parsed JSON as DataFrame
        """
        kwargs["as_dataframe"] = True
        return cls._read(filepath, **kwargs)

    @classmethod
    def read_dict(cls, filepath: str, **kwargs) -> Union[dict, list] | pl.LazyFrame | pl.DataFrame:
        """
        Convenience method to explicitly read JSON as dict/list.

        Args:
            filepath: Path to JSON file
            **kwargs: Arguments passed to _read

        Returns:
            Union[dict, list]: Raw JSON data
        """
        kwargs["as_dataframe"] = False
        return cls._read(filepath, **kwargs)


# ------------------------------------------------------------------------------------------------ #
#                                       IO SERVICE                                                 #
# ------------------------------------------------------------------------------------------------ #
class IOService:  # pragma: no cover
    __io = {
        "csv": CSVIO,
        "parquet": ParquetIO,
        "zip": ZipFileIO,
        "yaml": YamlIO,
        "yml": YamlIO,
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
