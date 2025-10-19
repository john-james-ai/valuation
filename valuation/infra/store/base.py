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
# Modified   : Saturday October 18th 2025 08:20:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Any, Dict, Optional, Union

from abc import abstractmethod
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from valuation.asset.base import Asset
from valuation.asset.identity import AssetType, Passport, Stage
from valuation.asset.store import AssetStore
from valuation.infra.exception import AssetStoreNotFoundError
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class AssetStoreBase(AssetStore):
    """Abstract base class for asset storage backends.

    Manages a directory-based store for serialized asset metadata (JSON files).

    Attributes:
        _location (Path): Filesystem path where asset JSON files are stored.
        _io (IOService): IO service used to read/write passport files.
    """

    def __init__(self, location: Optional[Path] = None, io: IOService = IOService) -> None:
        """Initialize the AssetStore.

        Ensures the storage directory exists and sets the IO service.

        Args:
            location (Optional[Path]): Directory path for storing asset JSON files. If None,
                the default ASSET_STORE_DIR is used.
            io (IOService): IO service used for reading and writing passport files.

        Returns:
            None
        """
        self._location = Path(location) or ASSET_STORE_DIR
        self._location.mkdir(parents=True, exist_ok=True)
        self._io = io
        self._file_system = FileSystem()

    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """Type of asset managed by this store.

        Returns:
            AssetType: The enum value representing the asset type handled by the store.
        """
        pass

    @property
    @abstractmethod
    def asset_location(self) -> AssetType:
        """Filesystem location where assets are stored.

        Returns:
            AssetType: The enum or descriptor indicating the asset storage location.
        """
        pass

    @abstractmethod
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
        passport_filepath = self._file_system.get_passport_filepath(
            location=self._location,
            asset_type=self.asset_type,
            stage=asset.passport.stage,
            name=asset.passport.name,
        )

        if passport_filepath.exists() and not overwrite:
            raise FileExistsError(f"{asset.passport.label} already exists in the store.")

        # Save passport using the to_dict method for formatting purposes
        self._io.write(filepath=passport_filepath, data=asset.passport.to_dict())

        # Save asset data
        asset.save()

        logger.debug(f"Added {asset.passport.label} to the store.")

    @abstractmethod
    def get(self, name: str, stage: Stage) -> Optional[Asset]:
        """Retrieve an asset from the store by name and stage.

        Args:
            name (str): The name of the asset to retrieve.
            stage (Stage): The stage of the asset to retrieve.

        Returns:
            Optional[Asset]: The retrieved asset instance, or None if not found.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(
            location=self._location, asset_type=self.asset_type, stage=stage, name=name
        )
        # Obtain the passport dictionary
        passport_dict = self._io.read(filepath=passport_filepath)
        # Create the passport
        passport = Passport.from_dict(passport_dict)
        # Instantiate the appropriate asset type
        asset = self.create_asset(passport=passport)
        return asset

    @abstractmethod
    def remove(self, name: str, stage: Stage) -> None:
        """Removes an asset from the store by name and stage.

        Deletes both the asset data file (if present) and its passport.

        Args:
            name (str): The name of the asset to remove.
            stage (Stage): The stage of the asset to remove.

        Returns:
            None

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(
            location=self._location, asset_type=self.asset_type, stage=stage, name=name
        )

        # Get the passport
        passport = self._get_passport(filepath=passport_filepath)

        # Get the asset filepath
        asset_filepath = self._file_system.get_asset_filepath(
            location=self._location, passport=passport
        )

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

        if not self._location.is_dir():
            raise AssetStoreNotFoundError(f"Error: Directory not found at '{self._location}'")

        registry = [
            data
            for path in self._location.glob("*.json")
            if (data := self._get_registry(path)) is not None
        ]

        return pd.DataFrame(registry)

    def exists(self, name: str, stage: Stage) -> bool:
        """Check if an asset exists in the store by name and stage.

        Args:
            name (str): The name of the asset to check.
            stage (Stage): The stage of the asset to check.

        Returns:
            bool: True if the asset exists, False otherwise.
        """
        passport_filepath = self._file_system.get_passport_filepath(
            location=self._location,
            asset_type=self.asset_type,
            stage=stage,
            name=name,
        )

        return passport_filepath.exists()

    def _remove_file(self, filepath: Union[str, Path]) -> None:
        """Remove a file if it exists.

        Args:
            filepath (Union[str, Path]): The path to the file to remove.

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
            return Passport.from_dict(passport_dict)
        except Exception as e:
            logger.error(f"Failed to read passport from '{filepath}': {e}")
            raise

    def _get_registry(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Safely read and parse a single JSON registry file.

        Args:
            filepath (Path): The path to the JSON file.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the parsed JSON data, or None if parsing fails.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Could not parse '{filepath.name}'. Skipping. Reason: {e}")
            return None
