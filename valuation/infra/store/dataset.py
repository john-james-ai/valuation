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
# Modified   : Monday October 20th 2025 03:03:38 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional

from loguru import logger

from valuation.asset.dataset.base import Dataset
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

    def add(self, dataset: Dataset, overwrite: bool = False) -> None:
        """Add a dataset to the store.
        Args:
            dataset (Dataset): The dataset instance to add.
            overwrite (bool, optional): If True, overwrite an existing dataset with the same name.
                Defaults to False.
        """
        if MODE == "prod" and str(dataset.passport.stage) == "raw":
            raise RuntimeError(
                "DatasetStore is in 'prod' mode. No changes to raw data are allowed."
            )

        passport_filepath = self._file_system.get_passport_filepath(id_or_passport=dataset.passport)

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

    def get(self, dataset_id: DatasetID, **kwargs) -> Optional[Dataset]:
        """Retrieve a dataset from the store by its ID.

        Args:
            asset_id (ID): The unique identifier of the dataset.
        Returns:
            Optional[Asset]: The retrieved Dataset asset, or None if not found.
        """
        passport_filepath = self._file_system.get_passport_filepath(id_or_passport=dataset_id)
        # Obtain the passport dictionary
        passport_dict = self._io.read(filepath=passport_filepath)
        # Create the passport
        passport = DatasetPassport.from_dict(passport_dict)
        # Instantiate the appropriate asset type
        dataset = Dataset(passport=passport)
        return dataset

    def remove(self, dataset_id: DatasetID, **kwargs) -> None:
        """Remove a dataset from the store by its ID.
        Args:
            dataset_id (DatasetID): The unique identifier of the dataset.
        Returns:
            None
        """

        if MODE == "prod" and str(dataset_id.stage) == "raw":
            raise RuntimeError(
                "DatasetStore is in 'prod' mode. No deletion of raw data is allowed."
            )
        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(id_or_passport=dataset_id)

        # If passport exists remove it and associated asset
        if passport_filepath.exists():

            # Get the passport dictionary
            passport_dict = self._io.read(filepath=passport_filepath)

            # Convert to passport
            passport = DatasetPassport.from_dict(passport_dict)

            # Get the asset filepath
            asset_filepath = self._file_system.get_asset_filepath(id_or_passport=passport)

            # Remove passport if exists else log attempt
            if asset_filepath.exists():
                self._remove_file(filepath=asset_filepath)
                logger.debug(f"Removed asset at {asset_filepath} from the store.")
            else:
                msg = f"Attempted to remove non-existing dataset data with ID {dataset_id.label}"
                logger.debug(msg)

            # Remove passport
            self._remove_file(filepath=passport_filepath)
            logger.debug(f"Removed passport at {passport_filepath} from the store.")

        else:
            msg = f"Attempt to remove non-existing dataset passport with ID {str(dataset_id)}"
            logger.debug(msg)

    def exists(self, dataset_id: DatasetID, **kwargs) -> bool:
        passport_filepath = self._file_system.get_passport_filepath(id_or_passport=dataset_id)

        return passport_filepath.exists()
