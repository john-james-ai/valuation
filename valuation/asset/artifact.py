#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/artifact.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 07:11:18 pm                                               #
# Modified   : Saturday October 25th 2025 10:25:22 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides data utilities."""
from __future__ import annotations

from typing import Any

import polars as pl

from valuation.asset.base import Asset, Passport
from valuation.core.types import AssetType
from valuation.infra.file.artifact import ArtifactFileSystem

# ------------------------------------------------------------------------------------------------ #
DTYPES = {}
DTYPES = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "date": pl.Datetime,
    "upc": pl.Int64,
    "week": pl.Int64,
    "qty": pl.Int64,
    "move": pl.Int64,
    "ok": pl.Int64,
    "price": pl.Float64,
    "revenue": pl.Float64,
    "profit": pl.Float64,
    "year": pl.Int64,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "gross_margin_pct": pl.Float64,
    "gross_margin": pl.Float64,
    "gross_profit": pl.Float64,
    "price_hex": pl.Utf8,
    "profit_hex": pl.Utf8,
}

DTYPES_CAPITAL = {k.capitalize(): v for k, v in DTYPES.items()}
DTYPES_UPPER = {k.upper(): v for k, v in DTYPES.items()}
DTYPES.update(DTYPES_CAPITAL)
DTYPES.update(DTYPES_UPPER)

# Derive helper column lists from polars DTYPES mapping
NUMERIC_TYPES = (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
NUMERIC_COLUMNS = [k for k, v in DTYPES.items() if v in NUMERIC_TYPES]
DATETIME_COLUMNS = [k for k, v in DTYPES.items() if "Datetime" in str(v)]
STRING_COLUMNS = [k for k, v in DTYPES.items() if v == pl.Utf8]

NUMERIC_PLACEHOLDER = -1  # Placeholder for missing numeric values
STRING_PLACEHOLDER = "Unknown"  # Placeholder for missing string values
# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
#                                           ARTIFACT                                               #
# ------------------------------------------------------------------------------------------------ #


class Artifact(Asset):

    def __init__(
        self,
        passport: Passport,
        file_system: ArtifactFileSystem,
        asset: Any,
    ) -> None:
        super().__init__(
            passport=passport,
            file_system=file_system,
            asset=asset,
        )

    @property
    def artifact(self) -> Any:
        """The artifact's content."""
        super().asset

    @property
    def asset_type(self) -> AssetType:
        """The type of asset."""
        return AssetType.ARTIFACT


# ------------------------------------------------------------------------------------------------ #


class Table(Artifact):

    def __init__(
        self,
        passport: Passport,
        file_system: ArtifactFileSystem,
        asset: Any,
    ) -> None:
        super().__init__(
            passport=passport,
            file_system=file_system,
            asset=asset,
        )

    @property
    def asset_type(self) -> AssetType:
        """The type of asset."""
        return AssetType.TABLE


# ------------------------------------------------------------------------------------------------ #


class Plot(Artifact):

    def __init__(
        self,
        passport: Passport,
        file_system: ArtifactFileSystem,
        asset: Any,
    ) -> None:
        super().__init__(
            passport=passport,
            file_system=file_system,
            asset=asset,
        )

    @property
    def asset_type(self) -> AssetType:
        """The type of asset."""
        return AssetType.PLOT
