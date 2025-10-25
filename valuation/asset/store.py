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
# Modified   : Saturday October 25th 2025 04:06:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the AssetStore abstract base class."""
from typing import Optional

from abc import ABC, abstractmethod

from valuation.asset.base import Asset
from valuation.asset.identity.base import ID, Passport

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

    @abstractmethod
    def exists(self, asset_id: ID, **kwargs) -> bool:
        """Check if an asset exists in the store by its ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).
        """
        pass
