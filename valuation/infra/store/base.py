#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/base.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Sunday October 19th 2025 02:52:23 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Any, Dict, Optional

from abc import abstractmethod
from pathlib import Path

from loguru import logger
import pandas as pd

from valuation.asset.base import Asset
from valuation.asset.identity.base import ID, Passport
from valuation.asset.store import AssetStore
from valuation.asset.types import AssetType
from valuation.infra.exception import AssetStoreNotFoundError
from valuation.infra.file.base import FileSystem
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class AssetStoreBase(AssetStore):
    """Abstract base class for asset storage backends.

    Manages a directory-based store for serialized asset metadata (JSON files).

    Attributes:
        _location (Path): Filesystem path where asset JSON files are stored.
        _io (IOService): IO service used to read/write passport files.
    """

    def __init__(self, filesystem: FileSystem, io: IOService = IOService) -> None:

        self._io = io
        self._file_system = filesystem

    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """Type of asset managed by this store.

        Returns:
            AssetType: The enum value representing the asset type handled by the store.
        """
        pass

    @abstractmethod
    def passport_from_dict(self, passport_dict: Dict[str, Any]) -> Passport:
        """ "Create a Passport instance from a dictionary.

        Args:
            passport_dict (Dict[str, Any]): The dictionary representation of the passport.
        Returns:
            Passport: The created Passport instance.
        """

    def add(self, asset: Asset, overwrite: bool = False) -> None:
        """Add an asset to the store.

        Serializes and persists the asset's passport and triggers the asset to save its data.

        Args:
            asset (Asset): The asset instance to add.
            overwrite (bool, optional): If True, overwrite an existing asset with the same name.
                Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If an asset with the same passport already exists and overwrite is False.
        """
        passport_filepath = self._file_system.get_passport_filepath(asset_id=asset.passport)

        if passport_filepath.exists() and not overwrite:
            raise FileExistsError(f"{asset.passport.label} already exists in the store.")

        # Save passport using the to_dict method for formatting purposes
        self._io.write(filepath=passport_filepath, data=asset.passport.to_dict())

        # Save asset data
        asset.save()

        logger.debug(f"Added {asset.passport.label} to the store.")

    def get(self, asset_id: ID, **kwargs) -> Optional[Asset]:

        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(asset_id=asset_id)
        # Check existence
        if not passport_filepath.exists():
            raise FileNotFoundError(
                f"Passport file for '{asset_id.name}' (stage={str(asset_id.stage)}) not found."
            )
        # Obtain the passport dictionary
        passport_dict = self._io.read(filepath=passport_filepath)
        # Create the passport
        passport = self.passport_from_dict(passport_dict=passport_dict)
        # Instantiate the appropriate asset type
        asset = self.create_asset(passport=passport)
        return asset

    def remove(self, asset_id: ID, **kwargs) -> None:

        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(asset_id=asset_id)

        # Get the passport
        passport = self._get_passport(filepath=passport_filepath)

        # Get the asset filepath
        asset_filepath = self._file_system.get_asset_filepath(asset_id=passport)

        # Remove asset data file and passport
        self._remove_file(filepath=asset_filepath)
        self._remove_file(filepath=passport_filepath)

    @abstractmethod
    def create_asset(self, passport: Passport) -> Asset:
        """Create an asset instance based on the provided passport.

        Args:
            passport (Passport): The passport of the asset to create.

        Returns:
            Asset: The created asset instance.
        """

    @property
    def registry(self) -> pd.DataFrame:
        """List all asset passports in the store as a DataFrame.

        Iterates over JSON files in the store directory, parses them, and returns
        a pandas DataFrame summarizing the registry entries. Malformed JSON files
        are skipped with a warning.

        Returns:
            pandas.DataFrame: DataFrame containing registry information for each valid JSON file.

        Raises:
            AssetStoreNotFoundError: If the configured store directory does not exist.
        """
        registry = []

        # Get the

        store_location = self._file_system.asset_store_location

        if not store_location:
            raise AssetStoreNotFoundError(f"Error: Directory not found at '{store_location}'")

        registry = [
            data
            for path in store_location.glob("*.json")
            if (data := self._io.read(path)) is not None
        ]

        return pd.DataFrame(registry)

    def exists(self, asset_id: ID, **kwargs) -> bool:
        passport_filepath = self._file_system.get_passport_filepath(asset_id=asset_id)

        return passport_filepath.exists()

    def _remove_file(self, filepath: str | Path) -> None:
        """Remove a file if it exists.

        Args:
            filepath (str | Path): The path to the file to remove.

        Returns:
            None
        """
        path = Path(filepath)
        if path.exists():
            path.unlink()
            logger.debug(f"Removed file at '{filepath}'.")
        else:
            logger.debug(
                f"Attempted to remove a non-existent file at '{filepath}'. No action taken."
            )

    def _get_passport(self, filepath: Path) -> Passport:
        """Read and parse a passport JSON file.

        Args:
            filepath (Path): The path to the passport JSON file.

        Returns:
            Passport: The parsed Passport object.

        Raises:
            Exception: Any exception raised during IO or parsing will be propagated.
        """
        try:
            passport_dict = self._io.read(filepath=filepath)
            return self.passport_from_dict(passport_dict)
        except Exception as e:
            logger.error(f"Failed to read passport from '{filepath}': {e}")
            raise
