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
# Modified   : Saturday October 18th 2025 11:54:35 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional, cast

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity import Passport
from valuation.asset.stage import DatasetStage
from valuation.asset.types import AssetType
from valuation.infra.exception import DatasetExistsError, DatasetNotFoundError
from valuation.infra.store.base import AssetStoreBase


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(AssetStoreBase):

    def __init__(self) -> None:
        """ """
        super().__init__()

    @property
    def asset_type(self) -> AssetType:
        """ """
        return AssetType.DATASET

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
