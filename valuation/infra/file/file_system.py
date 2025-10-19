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
# Modified   : Saturday October 18th 2025 08:21:07 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from pathlib import Path

from valuation.asset.identity import AssetType, Passport, Stage

# ------------------------------------------------------------------------------------------------ #
PROJ_ROOT = Path(__file__).resolve().parents[2]

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

    Methods:
        get_asset_filepath: Build the full path for an asset data file and ensure the stage directory exists.
        get_passport_filepath: Build the passport JSON filepath for an asset.
    """

    __asset_type_stage_location_map = {
        AssetType.DATASET: {
            "passport_location": ASSET_STORE_DATASET_PASSPORT_DIR,
            "asset_location": DATA_DIR,
        },
        AssetType.MODEL: {
            "passport_location": ASSET_STORE_MODEL_PASSPORT_DIR,
            "asset_location": MODEL_DIR,
        },
        AssetType.REPORT: {
            "passport_location": ASSET_STORE_REPORT_PASSPORT_DIR,
            "asset_location": REPORT_DIR,
        },
    }

    def __init__(self, asset_type: AssetType) -> None:
        self._asset_type = asset_type
        self._passport_location = self.__asset_type_stage_location_map[asset_type][
            "passport_location"
        ]
        self._asset_location = self.__asset_type_stage_location_map[asset_type]["asset_location"]

    @property
    def asset_location(self) -> Path:
        """The base location for asset data files."""
        return self._asset_location

    def get_asset_filepath(self, passport: Passport) -> Path:
        """Construct the filepath for an asset data file and ensure directories exist.

        Args:
            passport (Passport): Passport describing the asset (provides stage, name, format, etc.).

        Returns:
            Path: Full path to the asset data file.
        """
        Path(self._asset_location / passport.stage.value).mkdir(parents=True, exist_ok=True)
        return Path(
            self._asset_location
            / passport.stage.value
            / f"{passport.asset_type.value}_{passport.stage.value}_{passport.name}_{passport.created}.{passport.asset_format}"
        )

    def get_passport_filepath(self, stage: Stage, name: str) -> Path:
        """Construct the filepath for an asset's passport JSON file.

        Args:
            location (Union[Path, str]): Base directory where passport files are stored.
            asset_type (AssetType): The asset type enum value.
            stage (Stage): The asset stage enum value.
            name (str): The asset name.

        Returns:
            Path: The path to the passport JSON file.
        """
        return self._passport_location / f"{self._asset_type.value}_{stage.value}_{name}.json"
