#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_io/test_csv_simple.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 02:19:33 am                                              #
# Modified   : Thursday October 16th 2025 02:20:31 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import dask.dataframe as dd
import pytest

CSV_PANDAS_IN_FILEPATH = "tests/data/wbat.csv"


@pytest.mark.csv2
def test_dask_barebones(caplog) -> None:
    """Tests the most basic dask.read_csv call."""

    # Call dask.read_csv with no extra arguments
    df_plan = dd.read_csv(CSV_PANDAS_IN_FILEPATH)

    # Check the result
    print(f"Columns found by Dask: {df_plan.columns}")
    assert len(df_plan.columns) == 11
