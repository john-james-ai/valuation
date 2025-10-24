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
# Modified   : Thursday October 23rd 2025 08:46:18 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""

from typing import Optional

from loguru import logger

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity.base import Passport
from valuation.asset.identity.dataset import DatasetID, DatasetPassport
from valuation.core.types import AssetType
from valuation.infra.file.base import MODE
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.store.base import AssetStoreBase

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(AssetStoreBase):
    """Store implementation for dataset assets.

    Initializes the AssetStoreBase with a DatasetFileSystem.

    Args:
        filesystem (DatasetFileSystem): The filesystem helper used by the base store.
    """

    _file_system: DatasetFileSystem

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

    @property
    def file_system(self) -> DatasetFileSystem:
        """The dataset file system."""

        return self._file_system

    def add(self, dataset: Dataset, overwrite: bool = False) -> None:
        """Add a dataset to the store.

        Saves the dataset passport and dataset data to the configured filesystem.

        Args:
            dataset (Dataset): The dataset instance to add.
            overwrite (bool): If True, overwrite an existing dataset with the same name. Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If the dataset already exists and overwrite is False.
            RuntimeError: If operation disallowed in current MODE (e.g., modifying raw data in prod).
        """
        if MODE == "prod" and str(dataset.passport.stage) == "raw":
            raise RuntimeError(
                "DatasetStore is in 'prod' mode. No changes to raw data are allowed."
            )

        passport_filepath = self._file_system.get_passport_filepath(dataset_id=dataset.passport.id)  # type: ignore

        if passport_filepath.exists() and not overwrite:
            msg = f"{dataset.passport.label} already exists in the store. To overwrite, set the overwrite flag to True."
            logger.error(msg)
            raise FileExistsError(msg)

        if passport_filepath.exists() and overwrite:
            msg = f"Asset {dataset.passport.label} already exists. Overwriting as per flag."
            logger.debug(msg)

        # Save passport using the to_dict method for formatting purposes
        self._io.write(filepath=passport_filepath, data=dataset.passport.to_dict())
        logger.debug(f"Saved passport for {dataset.passport.label}.")

        # Save asset data
        dataset.save()
        logger.debug(f"Saved dataset data for {dataset.passport.label} to the store.")

    def get(self, passport: DatasetPassport) -> Dataset:
        """Retrieve a dataset from the store by its passport.

        Args:
            passport (DatasetPassport): The passport of the dataset to retrieve.

        Returns:
            Dataset: The retrieved Dataset instance.
        """
        # Instantiate the appropriate asset type
        dataset = Dataset(passport=passport)
        return dataset

    def get_passport(self, dataset_id: DatasetID) -> Optional[Passport]:
        """Retrieve an asset passport by its ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.

        Returns:
            Optional[Passport]: The reconstructed Passport instance.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        passport_filepath = self._file_system.get_passport_filepath(dataset_id=dataset_id)  # type: ignore

        if not passport_filepath.exists():
            raise FileNotFoundError(f"Passport file not found at '{passport_filepath}'")

        data = self._io.read(passport_filepath)
        if data is None:
            return None

        passport = DatasetPassport.from_dict(data)

        return passport

    def remove(self, passport: DatasetPassport, **kwargs) -> None:
        """Remove a dataset from the store by its passport.

        Deletes both the dataset data file (if present) and its passport.

        Args:
            passport (DatasetPassport): The passport of the dataset to remove.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            None

        Raises:
            RuntimeError: If deletion of raw data is disallowed in current MODE.
        """

        if str(MODE).lower() == "prod" and str(passport.stage) == "raw":
            raise RuntimeError(
                "DatasetStore is in 'prod' mode. No deletion of raw data is allowed."
            )

        # Get asset filepath and remove it
        asset_filepath = self._file_system.get_asset_filepath(passport=passport)
        self._remove_file(filepath=asset_filepath)
        # Get passport filepath and remove it
        passport_filepath = self._file_system.get_passport_filepath(dataset_id=passport.id)  # type: ignore
        self._remove_file(filepath=passport_filepath)

    def exists(self, dataset_id: DatasetID, **kwargs) -> bool:
        """Check if a dataset exists in the store by its ID.

        Args:
            dataset_id (DatasetID): The unique identifier of the dataset (id component of passport).
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            bool: True if the dataset exists, False otherwise.
        """
        passport_filepath = self._file_system.get_passport_filepath(dataset_id=dataset_id)  # type: ignore
        return passport_filepath.exists()
