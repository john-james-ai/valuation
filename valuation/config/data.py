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
# Modified   : Monday October 13th 2025 09:34:46 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Centralized data configuration settings"""


# ------------------------------------------------------------------------------------------------ #
DTYPES = {
    # Original column names
    "STORE": "Int64",
    "UPC": "Int64",
    "WEEK": "Int64",
    "QTY": "Int64",
    "MOVE": "Int64",
    "OK": "Int64",
    "PRICE": "float64",
    "PROFIT": "float64",
    "YEAR": "Int64",
    "START": "datetime64[ns]",
    "END": "datetime64[ns]",
    "GROSS_MARGIN_PCT": "float64",
    # New column names
    "store": "Int64",
    "upc": "Int64",
    "week": "Int64",
    "year": "Int64",
    "qty": "Int64",
    "move": "Int64",
    "ok": "Int64",
    "price": "float64",
    "profit": "float64",
    "start": "datetime64[ns]",
    "end": "datetime64[ns]",
    "gross_profit": "float64",
    "gross_margin_pct": "float64",
    "gross_margin": "float64",
    "revenue": "float64",
    "category": "str",
}

NUMERIC_COLUMNS = [k for k, v in DTYPES.items() if v in ("Int64", "float64")]
DATETIME_COLUMNS = [k for k, v in DTYPES.items() if v == "datetime64[ns]"]
STRING_COLUMNS = [k for k, v in DTYPES.items() if v == "str"]

NUMERIC_PLACEHOLDER = -1  # Placeholder for missing numeric values
STRING_PLACEHOLDER = "Unknown"  # Placeholder for missing string values
