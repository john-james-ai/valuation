#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/store.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 18th 2025 08:00:53 pm                                              #
# Modified   : Sunday October 19th 2025 07:27:39 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the AssetStore abstract base class."""
from typing import Optional

from abc import ABC, abstractmethod

from valuation.asset.base import Asset
from valuation.asset.stage import Stage

# ------------------------------------------------------------------------------------------------ #


class AssetStore(ABC):
    """Abstract interface for asset storage backends.

    Provides the contract for storing, retrieving and removing assets.

    Methods:
        add(asset, overwrite=False): Persist an asset and its passport.
        get(name, stage): Retrieve an asset by name and stage.
        remove(name, stage): Remove asset data and passport.
        exists(name, stage): Check whether an asset exists.
    """

    @abstractmethod
    def add(self, asset: Asset, overwrite: bool = False) -> None:
        """Add an asset to the store.

        Args:
            asset (Asset): The asset instance to add.
            overwrite (bool, optional): If True, overwrite an existing asset with the same name.
                Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If an asset with the same passport already exists and overwrite is False.
        """
        pass

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
        pass

    @abstractmethod
    def remove(self, name: str, stage: Stage) -> None:
        """Remove an asset from the store by name and stage.

        Deletes both the asset data file (if present) and its passport.

        Args:
            name (str): The name of the asset to remove.
            stage (Stage): The stage of the asset to remove.

        Returns:
            None

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        pass

    @abstractmethod
    def exists(self, name: str, stage: Stage) -> bool:
        """Check if an asset exists in the store by name and stage.

        Args:
            name (str): The name of the asset to check.
            stage (Stage): The stage of the asset to check.

        Returns:
            bool: True if the asset exists, False otherwise.
        """
        pass
