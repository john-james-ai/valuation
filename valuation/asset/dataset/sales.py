#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/dataset/sales.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 05:27:10 am                                                #
# Modified   : Sunday October 19th 2025 05:31:20 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the SalesDataset class."""

from valuation.asset.dataset.base import Dataset
from valuation.asset.entity import Entity

# ------------------------------------------------------------------------------------------------ #


class SalesDataset(Dataset):
    """Manages sales datasets."""

    @property
    def entity(self) -> Entity:
        return Entity.SALES
