#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/dataset.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 02:01:46 pm                                                #
# Modified   : Tuesday October 21st 2025 01:45:02 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

from valuation.asset.identity.dataset import DatasetID, DatasetPassport
from valuation.core.entity import Entity
from valuation.core.stage import DatasetStage
from valuation.core.types import AssetType
from valuation.infra.file.base import MODE, FileSystem


# ------------------------------------------------------------------------------------------------ #
class DatasetFileSystem(FileSystem):
    """Utility providing filesystem path construction for datasets and dataset passports.

    This class encapsulates the logic for building consistent filepaths for dataset data
    and passport JSON files on a local filesystem.

    Args:
        asset_type (AssetType): The asset type used to determine store and asset base locations.

    Methods:
        get_asset_path(name: str, stage: DatasetStage) -> Path:
            Constructs the filesystem path for a dataset asset.
        get_passport_path(name: str, stage: DatasetStage) -> Path:
            Constructs the filesystem path for a dataset passport.
    """

    def __init__(self) -> None:
        super().__init__(asset_type=self.asset_type)

    @property
    def asset_type(self) -> AssetType:
        return AssetType.DATASET

    def get_asset_filepath(
        self,
        passport: DatasetPassport,
        **kwargs,
    ) -> Path:

        return Path(
            self._asset_location
            / MODE
            / str(passport.stage)
            / str(passport.entity)
            / f"{passport.name}.{str(passport.file_format)}"
        )

    def get_passport_filepath(self, dataset_id: DatasetID, **kwargs) -> Path:

        return Path(
            self._store_location
            / MODE
            / f"{str(dataset_id.asset_type)}_{str(dataset_id.entity)}_{str(dataset_id.stage)}_{dataset_id.name}_passport.json"
        )

    def get_stage_location(self, stage: DatasetStage) -> Path:
        """Builds the full filepath for an asset stage directory."""
        return Path(self._asset_location) / MODE / str(stage)

    def get_stage_entity_location(self, stage: DatasetStage, entity: Entity) -> Path:
        """Builds the full filepath for an asset stage/entity directory."""
        return Path(self._asset_location) / MODE / str(stage) / str(entity)
