#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/model/mlforecast.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 04:30:54 pm                                              #
# Modified   : Friday October 24th 2025 09:49:07 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Any, Optional

from dataclasses import asdict

from mlforecast import MLForecast

from valuation.asset.identity.model import ModelPassport
from valuation.asset.model.base import Model
from valuation.flow.modeling.model_selection.base import ModelParams
from valuation.flow.modeling.model_selection.performance import PerformanceMetrics
from valuation.infra.file.model import ModelFileSystem


# ------------------------------------------------------------------------------------------------ #
class MLForecastModel(Model):
    def __init__(
        self,
        passport: ModelPassport,
        params: Optional[ModelParams] = None,
        performance: Optional[PerformanceMetrics] | None = None,
        model: Optional[Any] = None,
    ) -> None:
        super().__init__(passport=passport, params=params, model=model, performance=performance)
        self._asset_filepath = ModelFileSystem().get_asset_filepath(passport=passport)

    def as_dict(self) -> dict:
        """Returns the MLForecast model as a dictionary.

        Returns:
            dict: The MLForecast model as a dictionary.
        """
        return {
            "passport": self.passport.to_dict(),
            "params": {
                "light_gbm": asdict(self.params.light_gbm) if self.params else None,
                "mlforecast": asdict(self.params.mlforecast) if self.params else None,
            },
            "performance": asdict(self.performance) if self.performance else None,
        }

    def load(self) -> None:
        """Loads the MLForecast model from disk.

        Returns:
            MLForecast: The loaded MLForecast model.
        """
        self._model = MLForecast.load(self._asset_filepath)  # type: ignore
