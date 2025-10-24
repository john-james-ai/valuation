#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/model.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Friday October 24th 2025 11:26:21 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Model Store."""
"""Manages the Model Store."""

from typing import Optional

from loguru import logger
from mlforecast import MLForecast

from valuation.asset.identity.base import Passport
from valuation.asset.identity.model import ModelID, ModelPassport
from valuation.asset.model.base import Model
from valuation.asset.model.mlforecast import MLForecastModel
from valuation.core.types import AssetType
from valuation.infra.file.model import ModelFileSystem
from valuation.infra.store.base import AssetStoreBase

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
class ModelStore(AssetStoreBase):
    """Store implementation for model assets.

    Initializes the AssetStoreBase with a ModelFileSystem.

    Args:
        filesystem (ModelFileSystem): The filesystem helper used by the base store.
    """

    _file_system: ModelFileSystem
    _model = MLForecast

    def __init__(self) -> None:
        """Initialize the ModelStore.

        Note:
            The superclass is initialized with a ModelFileSystem instance.

        Returns:
            None
        """
        super().__init__(filesystem=ModelFileSystem())

    @property
    def asset_type(self) -> AssetType:
        """The asset type managed by this store.

        Returns:
            AssetType: The AssetType for models (AssetType.DATASET).
        """
        return AssetType.MODEL

    def add(self, model: MLForecastModel, overwrite: bool = False) -> None:
        """Add a model to the store.

        Saves the model passport and model data to the configured filesystem.

        Args:
            model (Model): The model instance to add.
            overwrite (bool): If True, overwrite an existing model with the same name. Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If the model already exists and overwrite is False.
            RuntimeError: If operation disallowed in current MODE (e.g., modifying raw data in prod).
        """

        passport_filepath = self._file_system.get_passport_filepath(model_id=model.passport.id)  # type: ignore

        if passport_filepath.exists() and not overwrite:
            msg = f"{model.passport.label} already exists in the store. To overwrite, set the overwrite flag to True."
            logger.error(msg)
            raise FileExistsError(msg)

        if passport_filepath.exists() and overwrite:
            msg = f"Model {model.passport.label} already exists. Overwriting as per flag."
            logger.debug(msg)

        # Save passport using the to_dict method for formatting purposes
        self._io.write(filepath=passport_filepath, data=model.passport.to_dict())
        logger.debug(f"Saved model passport for {model.passport.label}.")

        # Save model
        asset_filepath = self._file_system.get_asset_filepath(passport=model.passport)
        model.model.save(path=asset_filepath)  # type: ignore
        logger.debug(f"Saved model data for {model.passport.label} to the store.")

    def get(self, passport: ModelPassport) -> Model:
        """Retrieve a model from the store by its passport.

        Args:
            passport (ModelPassport): The passport of the model to retrieve.

        Returns:
            Model: The retrieved Model instance.
        """
        # Instantiate the appropriate asset type
        model = MLForecastModel(passport=passport)
        return model

    def get_passport(self, model_id: ModelID) -> Optional[Passport]:
        """Retrieve an asset passport by its ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.

        Returns:
            Optional[Passport]: The reconstructed Passport instance.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        passport_filepath = self._file_system.get_passport_filepath(model_id=model_id)  # type: ignore

        if not passport_filepath.exists():
            raise FileNotFoundError(f"Passport file not found at '{passport_filepath}'")

        data = self._io.read(passport_filepath)
        if data is None:
            return None

        passport = ModelPassport.from_dict(data)

        return passport

    def remove(self, passport: ModelPassport, **kwargs) -> None:
        """Remove a model from the store by its passport.

        Deletes both the model data file (if present) and its passport.

        Args:
            passport (ModelPassport): The passport of the model to remove.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            None

        Raises:
            RuntimeError: If deletion of raw data is disallowed in current MODE.
        """

        # Get asset filepath and remove it
        asset_filepath = self._file_system.get_asset_filepath(passport=passport)
        self._remove_file(filepath=asset_filepath)
        # Get passport filepath and remove it
        passport_filepath = self._file_system.get_passport_filepath(model_id=passport.id)  # type: ignore
        self._remove_file(filepath=passport_filepath)

    def exists(self, model_id: ModelID, **kwargs) -> bool:
        """Check if a model exists in the store by its ID.

        Args:
            model_id (ModelID): The unique identifier of the model (id component of passport).
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            bool: True if the model exists, False otherwise.
        """
        passport_filepath = self._file_system.get_passport_filepath(model_id=model_id)  # type: ignore
        return passport_filepath.exists()
