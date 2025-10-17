#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_io/test_parquet.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 12:31:54 am                                              #
# Modified   : Thursday October 16th 2025 06:01:42 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect
import os
import shutil

import dask.dataframe as dd
from loguru import logger
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from valuation.utils.io.csv.service import CSVIOService
from valuation.utils.io.parquet.service import ParquetIOService

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"
# ------------------------------------------------------------------------------------------------ #
CSV_PANDAS_IN_FILEPATH = "tests/data/wbat.csv"
PARQUET_PANDAS_FILEPATH = "tests/data/test_parquet/pandas.parquet"
PARQUET_DASK_FILEPATH = f"tests/data/test_parquet/dask.parquet"


@pytest.mark.parquet
class TestParquet:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #

        os.remove(PARQUET_PANDAS_FILEPATH) if os.path.exists(PARQUET_PANDAS_FILEPATH) else None
        shutil.rmtree(PARQUET_DASK_FILEPATH) if os.path.exists(PARQUET_DASK_FILEPATH) else None
        assert not os.path.exists(PARQUET_PANDAS_FILEPATH)
        assert not os.path.exists(PARQUET_DASK_FILEPATH)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_pandas_parquet(self, caplog) -> None:
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
        ParquetIOService.write(
            filepath=PARQUET_PANDAS_FILEPATH,
            data=df,
            engine="pandas",
        )
        df_out = ParquetIOService.read(filepath=PARQUET_PANDAS_FILEPATH, engine="pandas")
        assert_frame_equal(df, df_out, check_dtype=False)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_dask_parquet(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Read the CSV into a DataFrame
        df = ParquetIOService.read(filepath=PARQUET_PANDAS_FILEPATH, engine="pandas")
        ddf = dd.from_pandas(df, npartitions=4)  # type: ignore
        assert isinstance(ddf, dd.DataFrame)
        assert ddf.shape[1] == 11
        # assert ddf.shape[0].compute() == df.shape[0]

        # Write the dataframe using Dask profile
        ParquetIOService.write(
            filepath=PARQUET_DASK_FILEPATH,
            data=ddf,
            engine="dask",
        )
        ddf2 = ParquetIOService.read(filepath=PARQUET_DASK_FILEPATH, engine="dask")

        assert isinstance(ddf, dd.DataFrame)
        assert isinstance(ddf2, dd.DataFrame)
        assert len(ddf.columns) == len(ddf2.columns)

        df1 = ddf.compute()
        df2 = ddf2.compute()

        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)
        assert df1.shape == df2.shape

        assert_frame_equal(df1, df2, check_dtype=False)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
