#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/file/artifact.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 25th 2025 09:36:31 am                                              #
# Modified   : Saturday October 25th 2025 04:09:17 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #


from pathlib import Path

from valuation.asset.identity.artifact import ArtifactID, ArtifactPassport
from valuation.core.stage import ArtifactStage
from valuation.core.types import AssetType
from valuation.infra.file.base import FileSystem

# ------------------------------------------------------------------------------------------------ #


class ArtifactFileSystem(FileSystem):
    """Filesystem path utilities for artifact assets and passports."""

    def __init__(self) -> None:
        super().__init__(asset_type=AssetType.DATASET)

    def get_asset_filepath(
        self,
        passport: ArtifactPassport,
        **kwargs,
    ) -> Path:
        """Builds the full filepath for an asset data file and ensures the stage directory exists."""
        return super().get_asset_filepath(passport=passport, **kwargs)

    def get_passport_filepath(self, artifact_id: ArtifactID) -> Path:
        """Builds the full filepath for an asset passport JSON file."""
        return super().get_passport_filepath(asset_id=artifact_id)

    def get_asset_stage_location(self, stage: ArtifactStage) -> Path:
        """Builds the full filepath for an asset stage directory."""
        return super().get_asset_stage_location(stage=stage)
