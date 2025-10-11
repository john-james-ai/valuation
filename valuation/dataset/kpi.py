#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Company Valuation                                                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/kpi.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:04 am                                                #
# Modified   : Saturday October 11th 2025 11:41:45 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Store Dataset Preparation"""

from pydantic.dataclasses import dataclass

from valuation.config.data_prep import DataPrepSISOConfig
from valuation.dataset.base import DataPrepSingleOutput


# ------------------------------------------------------------------------------------------------ #
@dataclass
class KPIDataPrepConfig(DataPrepSISOConfig):
    """Configuration for Store Data Preparation."""

    groupby: str


class KPIDataPrep(DataPrepSingleOutput):
    """Computes Store level KPIs for profitability analysis"""

    def prepare(self, config: KPIDataPrepConfig) -> None:
        if self._use_cache(config=config):
            return

        df = self.load(filepath=config.input_filepath)

        store_kpis = (
            df.groupby(config.groupby)
            .agg(revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"))
            .reset_index()
        )

        store_kpis["gross_margin_pct"] = store_kpis["gross_profit"] / store_kpis["revenue"]

        self.save(df=store_kpis, filepath=config.output_filepath)
