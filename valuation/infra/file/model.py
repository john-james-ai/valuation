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
# Created    : Saturday October 25th 2025 09:36:31 am                                              #
# Modified   : Saturday October 25th 2025 10:05:19 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Filesystem path utilities for model assets and passports."""
from pathlib import Path

from valuation.asset.identity.model import ModelID
from valuation.core.stage import ModelStage
from valuation.core.types import AssetType
from valuation.infra.file.base import FileSystem

# ------------------------------------------------------------------------------------------------ #


class ModelFileSystem(FileSystem):
    """Filesystem path utilities for model assets and passports."""

    def __init__(self) -> None:
        super().__init__(asset_type=AssetType.MODEL)

    def get_passport_filepath(self, model_id: ModelID) -> Path:
        """Builds the full filepath for an asset passport JSON file."""
        return super().get_passport_filepath(asset_id=model_id)

    def get_asset_stage_location(self, stage: ModelStage) -> Path:
        """Builds the full filepath for an asset stage directory."""
        return super().get_asset_stage_location(stage=stage)
