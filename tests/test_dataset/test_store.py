#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_dataset/test_store.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 09:10:58 am                                                #
# Modified   : Saturday October 18th 2025 11:15:33 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect

from loguru import logger
import pandas as pd
import pytest

from valuation.analysis.store import StoreDataset

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
# --------------------------------------------------------
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.store
class TestStore:  # pragma: no cover
    # ============================================================================================ #
    def test_store_kpis(self, sales, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        store_dataset = StoreDataset(sales=sales, min_weeks=50)
        store_kpis = store_dataset.store_kpis
        assert isinstance(store_kpis, pd.DataFrame)
        assert not store_kpis.empty
        assert "store" in list(store_kpis.columns)
        assert "revenue" in list(store_kpis.columns)
        assert "gross_profit" in list(store_kpis.columns)
        assert "gross_margin_pct" in list(store_kpis.columns)
        assert store_kpis["store"].nunique() > 80

        logger.info(store_kpis.head(3).to_string())
        logger.info(f"Store KPIs shape: {store_kpis.shape}")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_sales_growth(self, sales, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        store_dataset = StoreDataset(sales=sales, min_weeks=50)
        sales_growth = store_dataset.sales_growth
        assert isinstance(sales_growth, pd.DataFrame)
        assert not sales_growth.empty
        assert "store" in list(sales_growth.columns)
        assert "revenue_prev" in list(sales_growth.columns)
        assert "revenue_curr" in list(sales_growth.columns)
        assert "sales_growth_rate" in list(sales_growth.columns)
        assert sales_growth.shape[0] > 50

        logger.info(f"\n{sales_growth.head(3)}")
        logger.info(f"Sales Growth shape: {sales_growth.shape}")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
