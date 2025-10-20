#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/file.py                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday October 20th 2025 01:49:48 am                                                #
# Modified   : Monday October 20th 2025 01:58:11 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Core file structures."""
from enum import StrEnum

# ------------------------------------------------------------------------------------------------ #


class FileFormat(StrEnum):
    """Enumeration of supported file formats."""

    PARQUET = "parquet"
    CSV = "csv"
    JSON = "json"
