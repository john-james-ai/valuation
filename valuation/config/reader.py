#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/reader.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Monday October 13th 2025 06:24:10 am                                                #
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
