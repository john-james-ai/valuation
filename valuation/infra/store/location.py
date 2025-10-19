#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/location.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 06:47:58 pm                                              #
# Modified   : Saturday October 18th 2025 08:20:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Union

from pathlib import Path

from valuation.asset.identity import AssetType, Passport, Stage


class FileSystem:
    """Utility providing filesystem path construction for assets and passports.

    This class encapsulates the logic for building consistent filepaths for asset data
    and passport JSON files on a local filesystem.

    Methods:
        get_asset_filepath: Build the full path for an asset data file and ensure the stage directory exists.
        get_passport_filepath: Build the passport JSON filepath for an asset.
    """

    def get_asset_filepath(self, location: Union[Path, str], passport: Passport) -> Path:
        """Construct the filepath for an asset data file and ensure directories exist.

        Args:
            location (Union[Path, str]): Base directory where assets are stored.
            passport (Passport): Passport describing the asset (provides stage, name, format, etc.).

        Returns:
            Path: Full path to the asset data file.
        """
        location = Path(location)
        Path(location / passport.stage.value).mkdir(parents=True, exist_ok=True)
        return Path(
            location
            / passport.stage.value
            / f"{passport.asset_type.value}_{passport.stage.value}_{passport.name}_{passport.created}.{passport.asset_format}"
        )

    def get_passport_filepath(
        self, location: Union[Path, str], asset_type: AssetType, stage: Stage, name: str
    ) -> Path:
        """Construct the filepath for an asset's passport JSON file.

        Args:
            location (Union[Path, str]): Base directory where passport files are stored.
            asset_type (AssetType): The asset type enum value.
            stage (Stage): The asset stage enum value.
            name (str): The asset name.

        Returns:
            Path: The path to the passport JSON file.
        """
        location = Path(location)
        return location / f"{asset_type.value}_{stage.value}_{name}.json"
