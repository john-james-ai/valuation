#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/dataset.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Sunday October 19th 2025 02:53:36 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Any, Dict, cast

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.asset.types import AssetType
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.store.base import AssetStoreBase


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(AssetStoreBase):
    """Store implementation for dataset assets.

    Initializes the AssetStoreBase with a DatasetFileSystem.

    Attributes:
        filesystem (DatasetFileSystem): The filesystem helper used by the base store.
    """

    def __init__(self) -> None:
        """Initialize the DatasetStore.

        Note:
            The superclass is initialized with a DatasetFileSystem instance.

        Returns:
            None
        """
        super().__init__(filesystem=DatasetFileSystem())

    @property
    def asset_type(self) -> AssetType:
        """The asset type managed by this store.

        Returns:
            AssetType: The AssetType for datasets (AssetType.DATASET).
        """
        return AssetType.DATASET

    def passport_from_dict(self, passport_dict: Dict[str, Any]) -> DatasetPassport:
        """Reconstruct a DatasetPassport from a dictionary.

        Args:
            passport_dict (Dict[str, Any]): Dictionary containing passport fields.

        Returns:
            DatasetPassport: The reconstructed DatasetPassport instance.
        """
        return cast(DatasetPassport, DatasetPassport.from_dict(passport_dict))

    def create_asset(self, passport: DatasetPassport) -> Dataset:
        """Instantiate a Dataset from its passport.

        Args:
            passport (DatasetPassport): The passport describing the dataset.

        Returns:
            Dataset: The created Dataset instance.
        """
        return Dataset(passport=passport)
