#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/model/base.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 04:31:27 pm                                              #
# Modified   : Friday October 24th 2025 10:02:20 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines the base Model asset."""
"""Defines the base Model asset."""
from typing import Any, Optional

from abc import abstractmethod

from loguru import logger

from valuation.asset.base import Asset
from valuation.asset.identity.model import ModelPassport
from valuation.core.types import AssetType
from valuation.flow.modeling.model_selection.base import ModelParams

# ------------------------------------------------------------------------------------------------ #


class Model(Asset):

    def __init__(
        self,
        passport: ModelPassport,
        params: ModelParams | None = None,
        model: Optional[Any] = None,
    ) -> None:
        self._passport = passport
        self._params = params
        self._model = model
        self._asset_filepath = None

    @property
    def model(self) -> Any:
        return self._model

    @property
    def passport(self) -> ModelPassport:
        """The model's unique idasset."""
        return self._passport

    @property
    def params(self) -> Optional[ModelParams]:
        """The model's hyperparameters."""
        return self._params

    @property
    def asset_type(self) -> AssetType:
        """The type of asset."""
        return AssetType.MODEL  # type: ignore

    @property
    def file_exists(self) -> bool:
        """Indicates if the model file exists on disk."""
        return self._asset_filepath.exists() if self._asset_filepath else False

    @abstractmethod
    def load(self) -> None:
        """Loads data from the source filepath into the internal DataFrame.

        This method uses the injected IO service to read the file. It can also
        enforce specific data types on the loaded columns.

        Args:
            dtypes: An optional dictionary mapping column names to desired
                data types (e.g., {'id': 'str'}).
            **kwargs: Additional keyword arguments to pass to the IO service's
                read method.
        """

    def save(self, overwrite: bool = False) -> None:
        """Saves the in-memory DataFrame to its canonical filepath.

        Fails safely by default if a file already exists at the location.

        Args:
            overwrite: If True, allows overwriting an existing file.
            **kwargs: Additional keyword arguments to pass to the IO service's
                write method.

        Raises:
            ValueError: If the Model has no canonical filepath set.
            FileConflictError: If the file exists and `overwrite` is False.
        """
        self._model.model.save(self._asset_filepath)  # type: ignore

    def delete(self) -> None:
        """Deletes the file associated with this Model from the filesystem."""

        logger.debug(f"Deleting file(s) {self._asset_filepath}")
        self._asset_filepath.unlink(missing_ok=True)  # type: ignore

    def exists(self) -> bool:
        """Checks if a file exists at the Model's canonical filepath.

        Returns:
            True if the file exists, False otherwise. Returns False if no
            filepath is associated with the Model.
        """
        if not self._asset_filepath:
            return False
        return self._asset_filepath.exists()
