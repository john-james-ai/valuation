#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/store.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 08:00:53 pm                                              #
# Modified   : Saturday October 18th 2025 08:20:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the AssetStore abstract base class."""
from abc import ABC, abstractmethod

from valuation.asset.base import Asset

# ------------------------------------------------------------------------------------------------ #



class AssetStore(ABC):

    @abstractmethod
    def add(self, asset: Asset, overwrite: bool = False) -> None:
    @abstractmethod
    def get(self, name: str, stage: Stage) -> Optional[Asset]:

    @abstractmethod
    def remove(self, name: str, stage: Stage) -> None:
    @abstractmethod
    def exists(self, name: str, stage: Stage) -> bool: