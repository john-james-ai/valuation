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
# Modified   : Thursday October 23rd 2025 04:44:43 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from mlforecast import MLForecast

from valuation.asset.identity.model import ModelPassport
from valuation.asset.model.base import Model
from valuation.infra.file.model import ModelFileSystem


# ------------------------------------------------------------------------------------------------ #
class MLForecastModel(Model):
    def __init__(self, passport: ModelPassport, model: MLForecast | None = None) -> None:
        super().__init__(passport, model)
        self._asset_filepath = ModelFileSystem().get_asset_filepath(passport=passport)

    def load(self) -> None:
        """Loads the MLForecast model from disk.

        Returns:
            MLForecast: The loaded MLForecast model.
        """
        self._model = MLForecast.load(self._asset_filepath)  # type: ignore
