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
# Modified   : Sunday October 12th 2025 10:33:38 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import numpy as np
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

    @property
    def sales_growth(self) -> pd.DataFrame:
        """Gets the store level sales growth."""
        self._sales_growth = self._compute_sales_growth()
        return self._sales_growth

    def _compute_category_kpis(self) -> pd.DataFrame:
        """Computes category level KPIs."""
        if self._category_kpis is not None:
            return self._category_kpis

        # Aggregate data to category level
        dataset = self.dataset
        return self._aggregator.aggregate(data=dataset.data.copy(), groupby=["category"])

    def _compute_sales_growth(self) -> pd.DataFrame:
        """Returns a DataFrame containing sales growth by category."""
        if self._sales_growth is not None:
            return self._sales_growth

        # Obtain the dataset from the base class
        dataset = self.dataset

        aggregated = self._aggregator.aggregate(
            data=dataset.data.copy(), groupby=["category", "year"]
        )

        # Identify the previous and current years
        previous_year = dataset.years[-2] if len(dataset.years) > 1 else None
        current_year = dataset.years[-1] if len(dataset.years) > 0 else None

        # Identify categorys that were open in both years
        categorys_previous = set(
            aggregated[aggregated["year"] == previous_year]["category"].unique()
        )
        categorys_current = set(
            aggregated[aggregated["year"] == current_year]["category"].unique()
        )
        comp_categorys = categorys_previous.intersection(categorys_current)

        # Filter for only the comparable categorys and the two relevant years
        comp_categorys_data = aggregated[
            aggregated["category"].isin(comp_categorys)
            & aggregated["year"].isin([previous_year, current_year])
        ]

        # Calculate sales growth for each category
        revenue_prev = comp_categorys_data[comp_categorys_data["year"] == previous_year]
        revenue_curr = comp_categorys_data[comp_categorys_data["year"] == current_year]

        #  4. Merge the two DataFrames on 'category' to align previous and current year revenues
        comp_categorys_data = pd.merge(
            revenue_prev[["category", "revenue"]],
            revenue_curr[["category", "revenue"]],
            on="category",
            suffixes=("_prev", "_curr"),
        )
        # 5. Calculate the sales growth rate
        comp_categorys_data["sales_growth_rate"] = np.where(
            comp_categorys_data["revenue_prev"] > 0,
            ((comp_categorys_data["revenue_curr"] / comp_categorys_data["revenue_prev"]) - 1)
            * 100,
            0,
        )

        return comp_categorys_data
