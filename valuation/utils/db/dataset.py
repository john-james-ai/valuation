#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/db/dataset.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Saturday October 18th 2025 04:14:11 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Optional

from pathlib import Path

from valuation.utils.data import Dataset
from valuation.utils.db.base import EntityStore
from valuation.utils.exception import DatasetExistsError, DatasetNotFoundError
from valuation.utils.identity import DatasetStage, EntityType


# ------------------------------------------------------------------------------------------------ #
class DatasetStore(EntityStore):

    def __init__(self, location: Path) -> None:
        """ """
        super().__init__(location=location)

    def entity_type(self) -> EntityType:
        """ """
        return EntityType.DATASET

    def add(self, dataset: Dataset, overwrite: bool = False) -> None:
        """Adds a dataset to the store.

        Args:
            dataset (Dataset): The dataset to add.
            overwrite (bool): If True, overwrites an existing dataset with the same name and stage.
                Defaults to False.

        """
        try:
            super().add(entity=dataset, overwrite=overwrite)
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
            super().get(name=name, stage=stage)
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
