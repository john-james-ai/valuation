#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/store.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 03:17:59 am                                                #
# Modified   : Sunday October 12th 2025 10:18:15 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from loguru import logger
import numpy as np
import pandas as pd

from valuation.dataset.base import DataAggregator, Dataset

# ------------------------------------------------------------------------------------------------ #


class StoreDataset(Dataset):
    """Class for storing store information.
    Computes store level KPIs and sales growth.

    Args:
        sales (pd.DataFrame): Sales data.
        min_weeks (int, optional): Minimum number of weeks a store must be open to be included. Defaults to 50.
        aggregator_cls (type[DataAggregator], optional): Aggregator class to use for data aggregation. Defaults to DataAggregator.
    """

    def __init__(
        self,
        sales: pd.DataFrame,
        min_weeks: int = 50,
        aggregator_cls: type[DataAggregator] = DataAggregator,
    ) -> None:
        super().__init__(sales=sales, min_weeks=min_weeks)
        self._aggregator = aggregator_cls()
        self._store_kpis = None
        self._sales_growth = None

    @property
    def store_kpis(self) -> pd.DataFrame:
        """Gets the store level KPIs."""
        self._store_kpis = self._compute_store_kpis()
        return self._store_kpis

    @property
    def sales_growth(self) -> pd.DataFrame:
        """Gets the store level sales growth."""
        self._sales_growth = self._compute_sales_growth()
        return self._sales_growth

    def _compute_store_kpis(self) -> pd.DataFrame:
        """Computes store level KPIs."""
        if self._store_kpis is not None:
            return self._store_kpis

        # Aggregate data to store level
        dataset = self.dataset
        return self._aggregator.aggregate(data=dataset.data.copy(), groupby=["store"])

    def _compute_sales_growth(self) -> pd.DataFrame:
        """Returns a DataFrame containing sales growth by store."""
        if self._sales_growth is not None:
            return self._sales_growth

        # Obtain the dataset from the base class
        dataset = self.dataset

        aggregated = self._aggregator.aggregate(
            data=dataset.data.copy(), groupby=["store", "year"]
        )

        # Identify the previous and current years
        previous_year = dataset.years[-2] if len(dataset.years) > 1 else None
        current_year = dataset.years[-1] if len(dataset.years) > 0 else None

        # Identify stores that were open in both years
        stores_previous = set(aggregated[aggregated["year"] == previous_year]["store"].unique())
        stores_current = set(aggregated[aggregated["year"] == current_year]["store"].unique())
        comp_stores = stores_previous.intersection(stores_current)

        # Filter for only the comparable stores and the two relevant years
        comp_stores_data = aggregated[
            aggregated["store"].isin(comp_stores)
            & aggregated["year"].isin([previous_year, current_year])
        ]

        # Calculate sales growth for each store
        revenue_prev = comp_stores_data[comp_stores_data["year"] == previous_year]
        revenue_curr = comp_stores_data[comp_stores_data["year"] == current_year]
        logger.debug(
            f"Revenue Previous Shape: {revenue_prev.shape}, Revenue Current Shape: {revenue_curr.shape}"
        )

        #  4. Merge the two DataFrames on 'store' to align previous and current year revenues
        comp_stores_data = pd.merge(
            revenue_prev[["store", "revenue"]],
            revenue_curr[["store", "revenue"]],
            on="store",
            suffixes=("_prev", "_curr"),
        )
        # 5. Calculate the sales growth rate
        comp_stores_data["sales_growth_rate"] = np.where(
            comp_stores_data["revenue_prev"] > 0,
            ((comp_stores_data["revenue_curr"] / comp_stores_data["revenue_prev"]) - 1) * 100,
            0,
        )

        return comp_stores_data
