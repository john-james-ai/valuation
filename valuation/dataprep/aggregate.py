#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/aggregate.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:04 am                                                #
# Modified   : Monday October 13th 2025 01:23:48 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Store Dataset Preparation"""

from dataclasses import dataclass

from valuation.dataprep.base import DataPrepSingleOutput
from valuation.dataprep.config import DataPrepSISOConfig


# ------------------------------------------------------------------------------------------------ #
@dataclass
class KPIDataPrepConfig(DataPrepSISOConfig):
    """Configuration for Store Data Preparation."""

    groupby: str


class KPIDataPrep(DataPrepSingleOutput):
    """Computes Store level KPIs for profitability analysis"""

    def prepare(self, config: KPIDataPrepConfig) -> None:
        if self._output_exists(config=config):
            return

        df = self.load(filepath=config.input_location)

        store_kpis = (
            df.groupby(config.groupby)
            .agg(revenue=("revenue", "sum"), gross_profit=("gross_profit", "sum"))
            .reset_index()
        )

        store_kpis["gross_margin_pct"] = store_kpis["gross_profit"] / store_kpis["revenue"]

        self.save(df=store_kpis, filepath=config.output_location)
