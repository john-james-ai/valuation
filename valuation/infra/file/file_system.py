#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/file_system.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 06:47:58 pm                                              #
# Modified   : Sunday October 19th 2025 03:00:07 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Optional, Union

import os
from pathlib import Path

from dotenv import load_dotenv

from valuation.asset.identity import AssetType, Passport, Stage

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
PROJ_ROOT = Path(__file__).resolve().parents[2]
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


class FileSystem:
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

    @property
    def asset_location(self) -> Path:
        """The base location for asset data files."""
        return self._asset_location

    @property
    def asset_location_current_mode(self) -> Path:
        """The base location for asset data files."""
        return self._asset_location / MODE

    @property
    def store_location(self) -> Path:
        """The base location for asset data files."""
        return self._store_location

    @property
    def store_location_current_mode(self) -> Path:
        """The base location for asset data files."""
        return self._store_location / MODE

    def get_asset_filepath(
        self,
        passport_or_stage: Union[Passport, Stage],
        name: Optional[str] = None,
        format: str = "parquet",
        mode: str = MODE,
    ) -> Path:
        """Construct the filepath for an asset's data file.

        Args:
            passport_or_stage (Union[Passport, Stage]): Either a Passport instance (contains name and stage)
                or a Stage enum value specifying the asset stage.
            name (str, optional): The asset name; when a Passport is provided this is ignored.
            format (str): The file format/extension to use (default "parquet").
            mode (str): The operating mode subdirectory (default from MODE environment).

        Returns:
            Path: The path to the asset data file.
        """
        if isinstance(passport_or_stage, Passport):
            stage = passport_or_stage.stage
            name = passport_or_stage.name
        else:
            stage = passport_or_stage

        asset_filepath = (
            Path(self._asset_location)
            / mode
            / stage.value
            / f"{name}_{stage.value}_{mode}.{format}"
        )

        return asset_filepath

    def get_passport_filepath(self, stage: Stage, name: str, mode: str = MODE) -> Path:
        """Construct the filepath for an asset's passport JSON file.

        Args:
            location (Union[Path, str]): Base directory where passport files are stored.
            asset_type (AssetType): The asset type enum value.
            stage (Stage): The asset stage enum value.
            name (str): The asset name.

        Returns:
            Path: The path to the passport JSON file.
        """

        return (
            self._store_location
            / mode
            / self._asset_type.value
            / f"{self._asset_type.value}_{stage.value}_{name}.json"
        )
