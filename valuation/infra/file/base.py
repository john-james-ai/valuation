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
# Modified   : Sunday October 19th 2025 02:18:35 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #


from abc import ABC, abstractmethod
import os
from pathlib import Path

from dotenv import load_dotenv

from valuation.asset.identity.base import ID, Passport
from valuation.asset.types import AssetType

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
PROJ_ROOT = Path(__file__).resolve().parents[3]
MODE = os.getenv("MODE", "dev")
# ------------------------------------------------------------------------------------------------ #
#                                          LOCATIONS                                               #
# ------------------------------------------------------------------------------------------------ #
# Asset Store Locations
DATA_DIR = PROJ_ROOT / "data"
MODEL_DIR = PROJ_ROOT / "models"
REPORT_DIR = PROJ_ROOT / "reports"
# Asset Store Passport Locations
ASSET_STORE_DIR = PROJ_ROOT / "asset_store"
ASSET_STORE_DATASET_PASSPORT_DIR = ASSET_STORE_DIR / "dataset"
ASSET_STORE_MODEL_PASSPORT_DIR = ASSET_STORE_DIR / "model"
ASSET_STORE_REPORT_PASSPORT_DIR = ASSET_STORE_DIR / "report"


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

    __asset_type_stage_location_map = {
        AssetType.DATASET: {
            "store_location": ASSET_STORE_DATASET_PASSPORT_DIR,
            "asset_location": DATA_DIR,
        },
        AssetType.MODEL: {
            "store_location": ASSET_STORE_MODEL_PASSPORT_DIR,
            "asset_location": MODEL_DIR,
        },
        AssetType.REPORT: {
            "store_location": ASSET_STORE_REPORT_PASSPORT_DIR,
            "asset_location": REPORT_DIR,
        },
    }

    def __init__(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type
        self._store_location = self.__asset_type_stage_location_map[asset_type]["store_location"]
        self._asset_location = self.__asset_type_stage_location_map[asset_type]["asset_location"]

    @abstractmethod
    def get_asset_filepath(
        self,
        asset_id: Passport | ID,
        format: str = "parquet",
        mode: str = MODE,
        **kwargs,
    ) -> Path:
        """Builds the full filepath for an asset data file and ensures the stage directory exists."""
        pass

    @abstractmethod
    def get_passport_filepath(self, asset_id: ID, mode: str = MODE) -> Path:
        """Builds the full filepath for an asset passport JSON file."""
        pass

    @property
    def asset_location(self) -> Path:
        """The base location for asset data files."""
        return self._asset_location / MODE

    @property
    def asset_store_location(self) -> Path:
        """The base location for asset data files."""
        return self._store_location / MODE
