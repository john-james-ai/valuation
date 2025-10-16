#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/zipfile/pandas.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 10:23:48 pm                                             #
# Modified   : Thursday October 16th 2025 01:13:44 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides functionality to read from and write to ZIP files containing CSV and Stata files using pandas."""

import io
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union
import zipfile

from loguru import logger
import pandas as pd

from valuation.utils.io.base import IO
from valuation.utils.io.csv.pandas import PandasReadCSVKwargs, PandasWriteCSVKwargs
from valuation.utils.io.stata import PandasReadStataKwargs, PandasWriteStataKwargs


# ------------------------------------------------------------------------------------------------ #
class ZipFilePandasIO(IO):  # pragma: no cover

    __read_csv_kwargs_class__ = PandasReadCSVKwargs
    __write_csv_kwargs_class__ = PandasWriteCSVKwargs
    __read_stata_kwargs_class__ = PandasReadStataKwargs
    __write_stata_kwargs_class__ = PandasWriteStataKwargs

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
    def read(
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
    def write(
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
        if operation == "read":
            if file_extension == ".csv":
                kwargs_class = cls.__read_csv_kwargs_class__
            elif file_extension == ".dta":
                kwargs_class = cls.__read_stata_kwargs_class__
            else:
                raise ValueError(f"Unsupported file extension for reading: {file_extension}")
            return kwargs_class(**all_kwargs).kwargs
        elif operation == "write":
            if file_extension == ".csv":
                kwargs_class = cls.__write_csv_kwargs_class__
            elif file_extension == ".dta":
                kwargs_class = cls.__write_stata_kwargs_class__
            else:
                raise ValueError(f"Unsupported file extension for writing: {file_extension}")
            return kwargs_class(**all_kwargs).kwargs
        else:
            raise ValueError("Operation must be either 'read' or 'write'.")
