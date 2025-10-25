#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/base.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 06:47:58 pm                                              #
# Modified   : Saturday October 25th 2025 10:04:02 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Filesystem path utilities for asset data and passports."""

from abc import ABC, abstractmethod
import os
from pathlib import Path

from dotenv import load_dotenv

from valuation.asset.identity.base import ID, Passport
from valuation.core.stage import Stage
from valuation.core.types import AssetType

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
PROJ_ROOT = Path(__file__).resolve().parents[3]
MODE = os.getenv("MODE", "dev")


# ------------------------------------------------------------------------------------------------ #
class FileSystem(ABC):
    """Utility providing filesystem path construction for assets and passports.

    This class encapsulates the logic for building consistent filepaths for asset data
    and passport JSON files on a local filesystem.

    Args:
        asset_type (AssetType): The asset type used to determine store and asset base locations.

    Methods:
        get_asset_filepath: Build the full path for an asset data file and ensure the stage directory exists.
        get_passport_filepath: Build the passport JSON filepath for an asset.
    """

    __ASSET_DIR_MAP = {
        "data": PROJ_ROOT / "data" / MODE,
        "model": PROJ_ROOT / "model" / MODE,
        "artifact": PROJ_ROOT / "artifacts" / MODE,
    }

    __ASSET_STORE_DIR = PROJ_ROOT / "asset_store" / MODE

    def __init__(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type
        self._store_location = Path(self.__ASSET_STORE_DIR) / str(AssetType)
        self._asset_location = self.__ASSET_DIR_MAP[str(self._asset_type)]

    @property
    def asset_location(self) -> Path:
        """The base location for asset data files."""
        return self._asset_location

    @property
    def asset_store_location(self) -> Path:
        """The base location for asset data files."""
        return self._store_location

    def get_asset_filepath(
        self,
        passport: Passport,
        **kwargs,
    ) -> Path:
        """Builds the full filepath for an asset data file and ensures the stage directory exists."""
        return (
            self._asset_location
            / str(passport.stage)
            / f"{passport.name}.{str(passport.file_format)}"
        )

    @abstractmethod
    def get_passport_filepath(self, asset_id: ID) -> Path:
        """Builds the full filepath for an asset passport JSON file."""
        filename = f"{str(asset_id.stage)}_{asset_id.name}_passport.json"
        return self._store_location / filename

    @abstractmethod
    def get_asset_stage_location(self, stage: Stage) -> Path:
        """Builds the full filepath for an asset stage directory."""
        return self._asset_location / str(stage)
