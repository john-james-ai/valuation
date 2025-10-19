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
# Modified   : Sunday October 19th 2025 04:13:37 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

from valuation.asset.identity.dataset import DatasetID, DatasetPassport
from valuation.asset.types import AssetType
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
        id_or_passport: DatasetPassport | DatasetID,
        format: str = "parquet",
        mode: str = MODE,
        **kwargs,
    ) -> Path:

        return Path(
            self._asset_location
            / mode
            / str(id_or_passport.asset_type)
            / str(id_or_passport.entity)
            / f"{str(id_or_passport.stage)}_{id_or_passport.name}.{format}"
        )

    def get_passport_filepath(
        self, id_or_passport: DatasetPassport | DatasetID, mode: str = MODE, **kwargs
    ) -> Path:

        return Path(
            self._store_location
            / mode
            / f"{str(id_or_passport.asset_type)}_{str(id_or_passport.entity)}_{str(id_or_passport.stage)}_{id_or_passport.name}_passport.json"
        )
