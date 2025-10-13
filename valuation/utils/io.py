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
# Modified   : Monday October 13th 2025 02:15:11 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
import codecs
import csv
import json
import logging
import os
from pathlib import Path
import pickle
from typing import Any, List, Union, cast
import zipfile

import pandas as pd
import yaml

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, unused-argument
# ------------------------------------------------------------------------------------------------ #

# ------------------------------------------------------------------------------------------------ #
#                                           IO                                                     #
# ------------------------------------------------------------------------------------------------ #


class IO(ABC):  # pragma: no cover
    _logger = logging.getLogger(
        f"{__module__}.{__name__}",
    )

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
#                                      ZIPFILE   IO                                                #
# ------------------------------------------------------------------------------------------------ #
class ZipFileCSVIO(IO):  # pragma: no cover

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
        """
        Reads a single CSV file from a ZIP archive into a Pandas DataFrame.

        This function opens a ZIP file in memory, finds the first file with a
        .csv extension, and reads it without extracting any files to disk.

        Args:
            zip_path: The file path to the .zip archive.

        Returns:
            A Pandas DataFrame containing the data from the CSV file.

        Raises:
            FileNotFoundError: If no .csv file is found inside the ZIP archive.
        """
        try:
            with zipfile.ZipFile(filepath, "r") as zip_ref:
                # Find the first file in the zip that ends with .csv
                csv_filepath = next(
                    (name for name in zip_ref.namelist() if name.endswith(".csv")), None
                )

                if csv_filepath is None:
                    raise FileNotFoundError(f"No CSV file found in {filepath}")

                # Open the CSV file from the archive as a file-like object
                with zip_ref.open(csv_filepath) as csv_file:
                    # Read the file-like object directly into pandas
                    return pd.read_csv(
                        csv_file,
                        sep=sep,
                        header=header,
                        index_col=index_col,
                        usecols=usecols,
                        escapechar=escapechar,
                        low_memory=low_memory,
                        encoding=encoding,
                    )
        except FileNotFoundError as e:
            print(e)
            raise
        except Exception as e:
            print(f"An error occurred while processing {filepath}: {e}")
            raise

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
        """
        Writes a DataFrame to a CSV and places it inside a structured ZIP archive.

        The ZIP archive will contain a directory named after the zip file (sans
        extension), and inside that directory will be a CSV file of the same name.

        Args:
            filepath: The destination path for the .zip archive.
            data: The Pandas DataFrame to save.
            sep: The separator to use for the CSV file.
            index: Whether to write the DataFrame index to the CSV.
            index_label: Column label for index column(s) if desired.
            encoding: The encoding to use for the output CSV file.
        """
        # Get the base name of the zip file, without its .zip extension
        basename_no_ext = os.path.splitext(os.path.basename(filepath))[0]

        # Construct the full path for the CSV file inside the zip archive
        # e.g., "my_data/my_data.csv"
        internal_csv_path = f"{basename_no_ext}/{basename_no_ext}.csv"

        # Open the zip file in write mode with compression
        with zipfile.ZipFile(filepath, "w", compression=zipfile.ZIP_DEFLATED) as zip_ref:
            # Convert the DataFrame to a CSV string in memory
            csv_buffer = data.to_csv(
                sep=sep, index=index, index_label=index_label, encoding=encoding
            )

            # Write the CSV string to the specified path within the zip archive
            zip_ref.writestr(internal_csv_path, csv_buffer)


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
                cls._logger.exception(e)
                raise IOError(e) from e
            finally:
                f.close()

    @classmethod
    def _write(cls, filepath: str, data: Any, **kwargs) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            try:
                yaml.dump(data, f)
            except yaml.YAMLError as e:  # pragma: no cover
                cls._logger.exception(e)
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
                cls._logger.exception(e)
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
                cls._logger.exception(e)
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
                        cls._logger.exception(msg)
                        raise ValueError(msg)
            else:
                try:
                    json.dump(data, json_file, indent=2)
                except json.JSONDecodeError as e:
                    cls._logger.exception(f"Exception of type {type(e)} occurred.\n{e}")
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
        "zip": ZipFileCSVIO,
        "dta": StataIO,
    }
    _logger = logging.getLogger(
        f"{__module__}.{__name__}",
    )

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
                cls._logger.exception(msg)
                raise ValueError(msg) from exc
            raise
        except KeyError as exc:
            msg = "File type {} is not supported.".format(file_format)
            cls._logger.exception(msg)
            raise ValueError(msg) from exc
