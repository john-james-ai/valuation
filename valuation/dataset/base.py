#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 05:20:43 pm                                              #
# Modified   : Sunday October 12th 2025 09:56:07 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for base operational analysis class."""
from abc import ABC
from dataclasses import dataclass, field
from typing import List, Union

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetContainer:

    data: pd.DataFrame
    years: List[int] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------ #
class Dataset(ABC):
    def __init__(self, sales: pd.DataFrame, min_weeks: int = 50) -> None:
        self._sales = sales
        self._min_weeks = min_weeks
        self._data = None

    @property
    def dataset(self) -> DatasetContainer:
        """Gets the data and its metadata in a container."""
        self._data = self._filter_partial_years()
        return self._data

    def _filter_partial_years(self) -> DatasetContainer:
        """Returns a DataFrame containing only full years of data."""
        if self._data is not None and isinstance(self._data, DatasetContainer):
            return self._data
        # 1. Group by year and count the number of unique weeks in each group
        weeks_per_year = self._sales.groupby("year")["week"].nunique()

        # 2. Filter to get a sorted list of years that have more than min_weeks of data
        years = weeks_per_year[weeks_per_year >= self._min_weeks].index.tolist()

        # 3. Filter the original DataFrame to include only rows from the full years
        data = self._sales[self._sales["year"].isin(years)].copy()

        # 4. Load the weeks in the dataset into the container
        self._data = DatasetContainer(years=years, data=data)
        return self._data


# ------------------------------------------------------------------------------------------------ #
class DataAggregator:

    def aggregate(self, data: pd.DataFrame, groupby: Union[str, List[str]]) -> pd.DataFrame:
        """Gets the data for only full years."""

        aggregated = (
            data.groupby(groupby)
            .agg(
                revenue=("revenue", "sum"),
                gross_profit=("gross_profit", "sum"),
            )
            .reset_index()
        )
        aggregated["gross_margin_pct"] = aggregated["gross_profit"] / aggregated["revenue"]
        return aggregated
