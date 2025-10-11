#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/reader.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Friday October 10th 2025 11:38:15 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Configuration Reader"""
from typing import Any, Dict

from valuation.config.filepaths import CONFIG_FILEPATH
from valuation.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class ConfigReader:
    """Reads configuration settings from a YAML file."""

    def __init__(self, io: IOService = IOService()) -> None:
        self.config = io.read(filepath=CONFIG_FILEPATH)

    def read(self, key: str, default=None) -> Dict[str, Any]:
        return self.config.get(key, default)
