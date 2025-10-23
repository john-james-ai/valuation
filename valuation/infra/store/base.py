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
# Modified   : Thursday October 23rd 2025 05:28:24 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional

from abc import abstractmethod
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd

from valuation.asset.base import Asset
from valuation.asset.identity.base import ID, Passport
from valuation.asset.store import AssetStore
from valuation.core.types import AssetType
from valuation.infra.exception import AssetStoreNotFoundError
from valuation.infra.file.base import FileSystem
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class AssetStoreBase(AssetStore):
    """Abstract base class for asset storage backends.

    Manages a directory-based store for serialized asset metadata (JSON files).

    Args:
        filesystem (FileSystem): Filesystem helper used to build paths for passports and assets.
        io (IOService): IO service used to read/write passport files.

    Attributes:
        _file_system (FileSystem): The filesystem helper instance.
        _io (IOService): IO service used to read/write passport files.
    """

    def __init__(self, filesystem: FileSystem, io: IOService = IOService) -> None:
        """Initialize the AssetStoreBase.

        Note:
            The constructor arguments are documented in the class docstring.

        Returns:
            None
        """
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

    @property
    def file_system(self) -> FileSystem:
        """Get the filesystem helper.

        Returns:
            FileSystem: The filesystem helper instance.
        """
        return self._file_system

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

    @abstractmethod
    def get(self, passport: Passport) -> Optional[Asset]:
        """Retrieve an asset by its ID (passport identifier).

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            Optional[Asset]: The reconstructed Asset instance.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """

    @abstractmethod
    def get_passport(self, asset_id: ID) -> Optional[Passport]:
        """Retrieve an asset passport by its ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.

        Returns:
            Optional[Passport]: The reconstructed Passport instance.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """

    @abstractmethod
    def remove(self, asset_id: ID, **kwargs) -> None:
        """Remove an asset and its passport by ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            None
        """
        pass

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
            if path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path, ignore_errors=True)
            logger.debug(f"Removed file at '{filepath}'.")
        else:
            logger.debug(
                f"Attempted to remove a non-existent file at '{filepath}'. No action taken."
            )
