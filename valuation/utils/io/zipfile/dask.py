#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/zipfile/dask.py                                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 15th 2025 10:23:48 pm                                             #
# Modified   : Wednesday October 15th 2025 11:43:05 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides functionality to read from and write to ZIP files containing CSV and Stata files using dask."""

import os
import shutil
import tempfile

import dask.dataframe as dd

from valuation.utils.io.base import IO
from valuation.utils.io.csv.dask import DaskReadCSVKwargs, DaskWriteCSVKwargs


# ------------------------------------------------------------------------------------------------ #
class ZipFileDaskIO(IO):  # pragma: no cover

    __read_csv_kwargs_class__ = DaskReadCSVKwargs
    __write_csv_kwargs_class__ = DaskWriteCSVKwargs

    @classmethod
    def read(cls, filepath: str, **kwargs) -> dd.DataFrame:
        """
        Reads all CSV files within a ZIP archive into a single Dask DataFrame.
        """
        # Build the final, flattened kwargs for the read operation
        read_kwargs = cls.__read_csv_kwargs_class__(**kwargs).read_kwargs

        # Dask can read directly from a zip archive using the `zip://` protocol.
        # The `*.csv` glob finds all CSV files inside.
        uri = f"zip://*.csv::{filepath}"
        return dd.read_csv(uri, **read_kwargs)

    @classmethod
    def write(
        cls,
        filepath: str,
        data: dd.DataFrame,
        internal_prefix: str = "data",
        **kwargs,
    ) -> None:
        """
        Writes a Dask DataFrame to a ZIP archive in a memory-safe way.

        This works by first writing the partitioned CSV files to a temporary
        directory and then zipping the contents of that directory.
        """
        # Build the final, flattened kwargs for the write operation
        write_kwargs = cls.__write_csv_kwargs_class__(**kwargs).write_kwargs

        # Ensure the final output directory exists
        output_dir = os.path.dirname(filepath)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Let Dask write partitioned CSVs to the temp directory.
            #    This is parallel and memory-efficient.
            dask_output_path = os.path.join(temp_dir, f"{internal_prefix}-*.csv")
            data.to_csv(dask_output_path, **write_kwargs)

            # 2. Zip the contents of the temp directory. `shutil.make_archive`
            #    is a convenient, standard library way to do this.
            #    We give it the final name but without the .zip extension.
            archive_name = os.path.splitext(filepath)[0]
            shutil.make_archive(base_name=archive_name, format="zip", root_dir=temp_dir)
