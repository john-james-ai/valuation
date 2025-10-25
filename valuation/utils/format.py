#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/format.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 25th 2025 12:07:46 pm                                              #
# Modified   : Saturday October 25th 2025 01:31:30 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

"""Utilities for formatting DataFrame columns for display.

Provides formatter implementations for both Polars and Pandas DataFrames.
"""

from typing import Any, Dict

from abc import ABC, abstractmethod

import pandas as pd
import polars as pl

# ------------------------------------------------------------------------------------------------ #
FORMAT_INT = "{:,.0f}"
FORMAT_INT_NO_COMMA = "{:.0f}"
FORMAT_FLOAT = "{:,.2f}"
FORMAT_PCT = "{:.2%}"
FORMAT_DOLLAR = "${:,.2f}"
FORMAT_DOLLAR_INT = "${:,.0f}"
FORMAT_DOLLAR_NO_CENTS = "${:,.0f}"
FORMAT_SCIENTIFIC = "{:.2e}"
FORMAT_SCIENTIFIC_INT = "{:.0e}"
FORMAT_SCIENTIFIC_PCT = "{:.2e}%"
FORMAT_SCIENTIFIC_DOLLAR = "${:.2e}"
FORMAT_SCIENTIFIC_DOLLAR_INT = "${:.0e}"
FORMAT_SCIENTIFIC_DOLLAR_PCT = "${:.2e}%"
FORMAT_PERCENTAGE = "{:.2%}"
FORMAT_PERCENTAGE_INT = "{:.0%}"
FORMAT_PERCENTAGE_SCIENTIFIC = "{:.2e}%"
FORMAT_PERCENTAGE_SCIENTIFIC_INT = "{:.0e}%"
FORMAT_STRING = "{}"


# ------------------------------------------------------------------------------------------------ #
class Formatter(ABC):
    """Base utility class for formatting DataFrame columns.

    Implementations should provide a format_dataframe method that applies the
    provided schema of format strings to the given DataFrame-like object.
    """

    @classmethod
    @abstractmethod
    def format_dataframe(cls, df: Any, schema: Dict[str, str]) -> Any:
        """Format DataFrame columns according to a schema mapping.

        Args:
            df (Any): DataFrame-like object (Polars or Pandas) to format.
            schema (Dict[str, str]): Mapping of column names to Python format strings.

        Returns:
            Any: The formatted DataFrame-like object.
        """
        pass


# ------------------------------------------------------------------------------------------------ #
class PolarsFormatter(Formatter):
    """Formatter for Polars DataFrames."""

    @classmethod
    def format_dataframe(cls, df: pl.DataFrame, schema: Dict[str, str]) -> pl.DataFrame:
        """Format columns of a Polars DataFrame/LazyFrame using format strings.

        Args:
            df (pl.DataFrame | pl.LazyFrame): Polars DataFrame or LazyFrame to format.
            schema (Dict[str, str]): Mapping from column name to Python format string (e.g. "{:,.2f}").

        Returns:
            pl.DataFrame | pl.LazyFrame: Formatted DataFrame or LazyFrame with specified columns formatted as strings.
        """
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        # Convert to Pandas for formatting
        df = df.to_pandas()
        schema = {col: fmt for col, fmt in schema.items() if col in df.columns}
        df = df.style.format(schema)
        return df


# ------------------------------------------------------------------------------------------------ #
class PandasFormatter(Formatter):
    """Formatter for Pandas DataFrames."""

    @classmethod
    def format_dataframe(cls, df: pd.DataFrame, schema: Dict[str, str]) -> pd.DataFrame:
        """Format columns of a Pandas DataFrame using format strings.

        Args:
            df (pd.DataFrame): Pandas DataFrame to format.
            schema (Dict[str, str]): Mapping from column name to Python format string (e.g. "{:,.2f}").

        Returns:
            pd.DataFrame: The formatted Pandas DataFrame with specified columns formatted as strings.
        """

        for col, fmt in schema.items():
            if col in df.columns:
                df[col] = df[col].apply(lambda x: fmt.format(x))
        return df
