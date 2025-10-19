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
# Modified   : Sunday October 19th 2025 02:15:14 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional, cast

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity import DatasetPassport
from valuation.asset.stage import DatasetStage
from valuation.asset.types import AssetType
from valuation.infra.exception import DatasetExistsError, DatasetNotFoundError
from valuation.infra.file.base import FileSystem
from valuation.infra.store.base import AssetStoreBase


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(AssetStoreBase):

    def __init__(self) -> None:
        """ """
        super().__init__()
        self._file_system = FileSystem(asset_type=self.asset_type)

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
        try:
            return cast(Dataset, super().get(name=name, stage=stage))
        except FileNotFoundError as e:
            raise DatasetNotFoundError(
                f"Dataset '{name}' at stage '{stage.value}' does not exist in the store."
            )

    def remove(self, name: str, stage: DatasetStage) -> None:

        # Get the filepath for the passport
        passport_filepath = self._file_system.get_passport_filepath(
            stage=stage,
            name=name,
        )

        # Get the passport
        passport = cast(DatasetPassport, self._get_passport(filepath=passport_filepath))

        # Get the asset filepath
        asset_filepath = self._file_system.get_asset_filepath(
            passport_or_stage=passport, entity=passport.entity
        )

        # Remove asset data file and passport
        self._remove_file(filepath=asset_filepath)
        self._remove_file(filepath=passport_filepath)

    def create_asset(self, passport: DatasetPassport) -> Dataset:
        return Dataset(passport=passport)
