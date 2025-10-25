#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/base.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 25th 2025 06:10:14 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod
from pathlib import Path
import shutil

from loguru import logger

from valuation.asset.identity.base import Passport
from valuation.core.types import AssetType
from valuation.infra.exception import AssetExistsError
from valuation.infra.file.base import FileSystem
from valuation.infra.file.io.fast import IOService


# ------------------------------------------------------------------------------------------------ #
class Asset(ABC):
    """
    An abstract base class for any object with a persistent state and a unique idasset.
    """

    def __init__(
        self,
        passport: Passport,
        file_system: FileSystem,
        asset: Any,
        io: type[IOService] = IOService,
    ) -> None:
        self._passport = passport
        self._file_system = file_system
        self._io = io
        self._asset_filepath = self._file_system.get_asset_filepath(passport=self._passport)
        self._asset = asset

    @property
    def name(self) -> str:
        """The asset's name."""
        return self._passport.name

    @property
    def passport(self) -> Passport:
        """The asset's unique and immutable passport."""
        return self._passport

    @property
    def asset(self) -> Any:
        return self._asset

    @property
    def file_exists(self) -> bool:
        """Indicates if the dataset file exists on disk."""
        return self._asset_filepath.exists() if self._asset_filepath else False

    @property
    @abstractmethod
    def asset_type(self) -> AssetType:
        """The type of asset."""
        pass

    def load(self) -> None:
        """Loads data from the source filepath into the internal DataFrame.

        This method uses the injected IO service to read the file. It can also
        enforce specific data types on the loaded columns.
        """
        try:
            self._asset = self._io.read(filepath=self._asset_filepath, **self._passport.read_kwargs)

            logger.debug(f"Artifact {self.passport.label} loaded.")
        except FileNotFoundError:
            raise FileNotFoundError(f"File {self._asset_filepath} not found. DataFrame is empty.")

    def save(self, overwrite: bool = False) -> None:
        """Saves the in-memory DataFrame to its canonical filepath.

        Fails safely by default if a file already exists at the location.

        Args:
            overwrite: If True, allows overwriting an existing file.

        Raises:
            ValueError: If the Artifact has no canonical filepath set.
            FileConflictError: If the file exists and `overwrite` is False.
        """
        self.save_as(self._asset_filepath, overwrite=overwrite)

    def save_as(self, filepath: Path | str, overwrite: bool = False) -> None:
        """Saves the in-memory DataFrame to a specified location.

        Fails safely by default if a file already exists at the location.

        Args:
            filepath: The target location to save the file.
            overwrite: If True, allows overwriting an existing file.

        Raises:
            FileConflictError: If the file exists and `overwrite` is False.
        """
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise AssetExistsError(
                f"Asset already exists at {filepath.name}. Set overwrite=True to replace it."
            )

        self._io.write(data=self._asset, filepath=filepath, **self._passport.write_kwargs)
        logger.debug(f"Artifact {self.passport.name} saved to {filepath}")

    def delete(self) -> None:
        """Deletes the file associated with this Artifact from the filesystem."""
        if not self._asset_filepath:
            raise ValueError("Filepath is not set. No file to delete.")
        logger.debug(f"Deleting file(s) {self._asset_filepath}")
        if self._asset_filepath.is_file():
            self._asset_filepath.unlink(missing_ok=True)
        else:
            shutil.rmtree(self._asset_filepath, ignore_errors=True)

    def exists(self) -> bool:
        """Checks if a file exists at the Artifact's canonical filepath.

        Returns:
            True if the file exists, False otherwise. Returns False if no
            filepath is associated with the Artifact.
        """
        if not self._asset_filepath:
            return False
        return self._asset_filepath.exists()
