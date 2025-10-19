#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/dataset_store.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Saturday October 18th 2025 08:20:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional, cast

from pathlib import Path

from valuation.asset.dataset import Dataset
from valuation.asset.identity import AssetType, DatasetStage, Passport
from valuation.config.filepaths import DATA_DIR
from valuation.infra.db.base import AssetStore
from valuation.infra.exception import DatasetExistsError, DatasetNotFoundError


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(AssetStore):

    def __init__(self, location: Path) -> None:
        """ """
        super().__init__(location=location)

    @property
    def asset_type(self) -> AssetType:
        """ """
        return AssetType.DATASET

    @property
    def asset_location(self) -> Path:
        """ """
        return DATA_DIR

    def add(self, dataset: Dataset, overwrite: bool = False) -> None:
        """Adds a dataset to the store.

        Args:
            dataset (Dataset): The dataset to add.
            overwrite (bool): If True, overwrites an existing dataset with the same name and stage.
                Defaults to False.

        """

        try:
            super().add(asset=dataset, overwrite=overwrite)
        except FileExistsError as e:
            raise DatasetExistsError(str(e)) from e

    def get(self, name: str, stage: DatasetStage) -> Optional[Dataset]:
        """Retrieves a dataset from the store by name and stage.

        Args:
            name (str): The name of the dataset.
            stage (DatasetStage): The stage of the dataset.

        Returns:
            Dataset: The retrieved dataset.
        """
        try:
            return cast(Dataset, super().get(name=name, stage=stage))

        except FileNotFoundError as e:
            raise DatasetNotFoundError(str(e)) from e

    def remove(self, name: str, stage: DatasetStage) -> None:
        """Removes a dataset from the store by name and stage.

        Args:
            name (str): The name of the dataset.
            stage (DatasetStage): The stage of the dataset.

        """
        try:
            super().remove(name=name, stage=stage)
        except FileNotFoundError as e:
            raise DatasetNotFoundError(str(e)) from e

    def create_asset(self, passport: Passport) -> Dataset:
        return Dataset(passport=passport)

    def get_asset_filepath(self, passport: Passport) -> Path:
        Path(self.asset_location / passport.stage.value).mkdir(parents=True, exist_ok=True)
        return (
            self.asset_location
            / passport.stage.value
            / f"{passport.asset_type.value}_{passport.stage.value}_{passport.name}_{passport.created}.{passport.asset_format}"
        )

    def get_passport_filepath(self, stage: DatasetStage, name: str) -> Path:
        return self._location / f"{self.asset_type.value}_{stage.value}_{name}.json"
