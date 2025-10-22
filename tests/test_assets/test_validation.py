#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_assets/test_validation.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 21st 2025 04:51:55 pm                                               #
# Modified   : Tuesday October 21st 2025 11:26:59 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect

from loguru import logger
import pandas as pd
import pytest

from valuation.flow.dataprep.sales.ingest import (
    NON_NEGATIVE_COLUMNS_INGEST,
    REQUIRED_COLUMNS_INGEST,
)
from valuation.flow.validation import (
    ColumnTypeValidator,
    MissingColumnValidator,
    NonNegativeValidator,
)

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"

CONFIG_FILEPATH = "config.yaml"
WEEK_DECODE_TABLE_FILEPATH = "reference/week_decode_table.csv"


@pytest.mark.validation
class TestValidation:  # pragma: no cover
    # ============================================================================================ #
    def test_missing_columns(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        df = pd.DataFrame(
            {
                "STORE": [1, 2, 3],
                "WEEK": [1, 2, 3],
                "YEAR": [2020, 2020, 2020],
                "REVENUE": [100.0, 200.0, 300.0],
            }
        )
        validator = MissingColumnValidator(
            required_columns=["STORE", "WEEK", "YEAR", "REVENUE", "GROSS_PROFIT"]
        )
        assert not validator.validate(data=df, classname=self.__class__.__name__)
        logger.info(validator.messages)
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_column_types(self, sales_df, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        sales_df["STORE"] = sales_df["STORE"].astype("string")  # Incorrect type
        validator = ColumnTypeValidator(column_types=REQUIRED_COLUMNS_INGEST)
        assert validator.validate(data=sales_df, classname=self.__class__.__name__) is False
        logger.info(validator.messages)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_non_negative(self, sales_df, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        sales_df["PRICE"] = sales_df["PRICE"] * -1  # Introduce negative value
        validator = NonNegativeValidator(columns=NON_NEGATIVE_COLUMNS_INGEST)
        assert validator.validate(data=sales_df, classname=self.__class__.__name__) is False
        logger.info(validator.messages)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
