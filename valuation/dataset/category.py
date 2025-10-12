#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/category.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 03:17:59 am                                                #
# Modified   : Sunday October 12th 2025 10:23:32 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import pandas as pd

from valuation.dataset.base import DataAggregator, Dataset

# ------------------------------------------------------------------------------------------------ #


class CategoryDataset(Dataset):
    """Class for storing category information.
    Computes category level KPIs and sales growth.

    Args:
        sales (pd.DataFrame): Sales data.
        min_weeks (int, optional): Minimum number of weeks a category must be open to be included.
            Defaults to 50.
        aggregator_cls (type[DataAggregator], optional): Aggregator class to use for data
            aggregation. Defaults to DataAggregator.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        min_weeks: int = 50,
        aggregator_cls: type[DataAggregator] = DataAggregator,
    ) -> None:
        super().__init__(sales=sales, min_weeks=min_weeks)
        self._aggregator = aggregator_cls()
        self._category_kpis = None
        self._sales_growth = None

    @property
    def category_kpis(self) -> pd.DataFrame:
        """Gets the category level KPIs."""
        self._category_kpis = self._compute_category_kpis()
        return self._category_kpis

    def _compute_category_kpis(self) -> pd.DataFrame:
        """Computes category level KPIs."""
        if self._category_kpis is not None:
            return self._category_kpis

        # Aggregate data to category level
        dataset = self.dataset
        return self._aggregator.aggregate(data=dataset.data.copy(), groupby=["category"])
