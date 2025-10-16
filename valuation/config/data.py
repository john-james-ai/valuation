#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/data.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Thursday October 16th 2025 11:15:25 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Centralized data configuration settings"""


# ------------------------------------------------------------------------------------------------ #
DTYPES = {}
DTYPES = {
    "CATEGORY": "string",
    "STORE": "Int64",
    "DATE": "datetime64[ns]",
    "UPC": "Int64",
    "WEEK": "Int64",
    "QTY": "Int64",
    "MOVE": "Int64",
    "OK": "Int64",
    "SALE": "string",
    "PRICE": "float64",
    "REVENUE": "float64",
    "PROFIT": "float64",
    "YEAR": "Int64",
    "START": "datetime64[ns]",
    "END": "datetime64[ns]",
    "GROSS_MARGIN_PCT": "float64",
    "GROSS_MARGIN": "float64",
    "GROSS_PROFIT": "float64",
    "OK": "Int64",
}
DTYPES_CAPITAL = {k.capitalize(): v for k, v in DTYPES.items()}
DTYPES_LOWER = {k.lower(): v for k, v in DTYPES.items()}
DTYPES.update(DTYPES_CAPITAL)
DTYPES.update(DTYPES_LOWER)

NUMERIC_COLUMNS = [k for k, v in DTYPES.items() if v in ("Int64", "float64")]
DATETIME_COLUMNS = [k for k, v in DTYPES.items() if v == "datetime64[ns]"]
STRING_COLUMNS = [k for k, v in DTYPES.items() if v == "str"]

NUMERIC_PLACEHOLDER = -1  # Placeholder for missing numeric values
STRING_PLACEHOLDER = "Unknown"  # Placeholder for missing string values
