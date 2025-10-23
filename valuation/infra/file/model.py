#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/model.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 02:01:46 pm                                                #
# Modified   : Thursday October 23rd 2025 04:20:13 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

from valuation.asset.identity.model import ModelID, ModelPassport
from valuation.core.stage import ModelStage
from valuation.core.types import AssetType
from valuation.infra.file.base import MODE, FileSystem


# ------------------------------------------------------------------------------------------------ #
class ModelFileSystem(FileSystem):
    """Utility providing filesystem path construction for models and model passports.

    This class encapsulates the logic for building consistent filepaths for model data
    and passport JSON files on a local filesystem.

    Args:
        asset_type (AssetType): The asset type used to determine store and asset base locations.

    Methods:
        get_asset_path(name: str, stage: ModelStage) -> Path:
            Constructs the filesystem path for a model asset.
        get_passport_path(name: str, stage: ModelStage) -> Path:
            Constructs the filesystem path for a model passport.
    """

    def __init__(self) -> None:
        super().__init__(asset_type=self.asset_type)

    @property
    def asset_type(self) -> AssetType:
        return AssetType.MODEL

    def get_asset_filepath(
        self,
        passport: ModelPassport,
        **kwargs,
    ) -> Path:

        return Path(
            self._asset_location
            / MODE
            / str(passport.stage)
            / f"{passport.name}.{str(passport.file_format)}"
        )

    def get_passport_filepath(self, model_id: ModelID, **kwargs) -> Path:

        return Path(
            self._store_location
            / MODE
            / f"{str(model_id.asset_type)}_{str(model_id.stage)}_{model_id.name}_passport.json"
        )

    def get_stage_location(self, stage: ModelStage) -> Path:
        """Builds the full filepath for an asset stage directory."""
        return Path(self._asset_location) / MODE / str(stage)
