#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Sunday October 19th 2025 03:53:32 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from abc import ABC, abstractmethod

from valuation.asset.identity.base import Passport


# ------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """
    An abstract base class for any object with a persistent state and a unique idasset.
    """

    def __init__(self, passport: Passport) -> None:
        self._passport = passport

    @property
    def name(self) -> str:
        """The asset's name."""
        return self._passport.name

    @property
    def passport(self) -> Passport:
        """The asset's unique and immutable passport."""
        return self._passport

    @abstractmethod
    def load(self) -> None:
        """Loads the asset's data from the filepath specified in its passport."""
        pass

    @abstractmethod
    def save(self) -> None:
        """Saves the asset's data to the filepath specified in its passport."""
        pass
