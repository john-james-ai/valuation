#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_dataset/test_category.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 09:10:58 am                                                #
# Modified   : Sunday October 12th 2025 10:35:16 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from datetime import datetime
import inspect

from loguru import logger
import pandas as pd
import pytest

from valuation.dataset.category import CategoryDataset

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
# --------------------------------------------------------
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.category
class TestCategory:  # pragma: no cover
    # ============================================================================================ #
    def test_category_kpis(self, sales, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        category_dataset = CategoryDataset(sales=sales, min_weeks=50)
        category_kpis = category_dataset.category_kpis
        assert isinstance(category_kpis, pd.DataFrame)
        assert not category_kpis.empty
        assert "category" in list(category_kpis.columns)
        assert "revenue" in list(category_kpis.columns)
        assert "gross_profit" in list(category_kpis.columns)
        assert "gross_margin_pct" in list(category_kpis.columns)
        assert category_kpis["category"].nunique() == 28

        logger.info(category_kpis.head(3).to_string())
        logger.info(f"Category KPIs shape: {category_kpis.shape}")

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
        category_dataset = CategoryDataset(sales=sales, min_weeks=50)
        sales_growth = category_dataset.sales_growth
        assert isinstance(sales_growth, pd.DataFrame)
        assert not sales_growth.empty
        assert "category" in list(sales_growth.columns)
        assert "revenue_prev" in list(sales_growth.columns)
        assert "revenue_curr" in list(sales_growth.columns)
        assert "sales_growth_rate" in list(sales_growth.columns)
        assert sales_growth.shape[0] == 28

        logger.info(f"\n{sales_growth.head(3)}")
        logger.info(f"Sales Growth shape: {sales_growth.shape}")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
