#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/base.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 12:14:37 am                                                #
# Modified   : Friday October 24th 2025 12:18:05 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass

from valuation.core.dataclass import DataClassRecursive
from valuation.flow.modeling.model_selection.cv import CrossValidationHP
from valuation.flow.modeling.model_selection.light_gbm import LightGBMHP
from valuation.flow.modeling.model_selection.mlforecast import MLForecastHP

# ------------------------------------------------------------------------------------------------ #


@dataclass
class ModelParams(DataClassRecursive):
    """Base class for model hyperparameters."""

    light_gbm: LightGBMHP
    mlforecast: MLForecastHP
    cross_validation: CrossValidationHP
