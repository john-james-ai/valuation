#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_io/test_csv.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 12:31:54 am                                              #
# Modified   : Thursday October 16th 2025 12:14:29 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect
import os
import shutil

import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
from loguru import logger
import pandas as pd
import pytest

from valuation.utils.io.csv.service import CSVIOService

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"
# ------------------------------------------------------------------------------------------------ #
CSV_PANDAS_IN_FILEPATH = "tests/data/wbat.csv"
CSV_PANDAS_OUT_FILEPATH = "tests/data/test_csv/pandas_out.csv"
CSV_DASK_IN_FILEPATH = f"tests/data/test_csv/dask_in.csv"
CSV_DASK_OUT_FILEPATH = "tests/data/test_csv/dask_out.csv"


@pytest.mark.csv
class TestCSV:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        os.remove(CSV_PANDAS_OUT_FILEPATH) if os.path.exists(CSV_PANDAS_OUT_FILEPATH) else None
        shutil.rmtree(CSV_DASK_OUT_FILEPATH) if os.path.exists(CSV_DASK_OUT_FILEPATH) else None
        assert not os.path.exists(CSV_PANDAS_OUT_FILEPATH)
        assert not os.path.exists(CSV_DASK_OUT_FILEPATH)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_pandas_csv(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Read CSV with Pandas
        df = CSVIOService.read(filepath=CSV_PANDAS_IN_FILEPATH, engine="pandas")
        assert isinstance(df, pd.DataFrame)
        assert df.shape[1] == 11
        assert df.shape[0] > 1e4

        # Write CSV with Pandas
        CSVIOService.write(
            filepath=CSV_PANDAS_OUT_FILEPATH,
            data=df,
            engine="pandas",
        )
        df_out = CSVIOService.read(filepath=CSV_PANDAS_OUT_FILEPATH, engine="pandas")
        assert df.equals(df_out)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_dask_csv(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Read CSV with Pandas
        ddf = CSVIOService.read(filepath=CSV_PANDAS_OUT_FILEPATH, engine="dask")

        # Write the Dask DataFrame to CSV files for Dask input
        CSVIOService.write(
            filepath=CSV_DASK_OUT_FILEPATH,
            data=ddf,
            engine="dask",
        )
        ddf2 = CSVIOService.read(filepath=CSV_DASK_OUT_FILEPATH, engine="dask")

        assert isinstance(ddf, dd.DataFrame)
        assert len(ddf.columns) == 11
        assert len(ddf) > 1e4
        assert_eq(ddf, ddf2)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
