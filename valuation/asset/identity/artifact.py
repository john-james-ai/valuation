#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/identity/artifact.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 25th 2025 09:47:04 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Dict, Optional, cast

from dataclasses import dataclass
from datetime import datetime

from valuation.asset.identity.base import ID, Passport
from valuation.core.file import FileFormat
from valuation.core.stage import ArtifactStage
from valuation.core.types import AssetType


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ArtifactPassport(Passport):
    """An immutable, unique idasset for a artifact."""

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        stage: ArtifactStage,
        file_format: Optional[FileFormat] = None,
        read_kwargs: Optional[Dict[str, str]] = None,
        write_kwargs: Optional[Dict[str, str]] = None,
    ) -> ArtifactPassport:
        """Creates a ArtifactPassport."""
        return cls(
            name=name,
            description=description,
            asset_type=AssetType.ARTIFACT,
            stage=stage,
            file_format=file_format,
            read_kwargs=read_kwargs if read_kwargs is not None else {},
            write_kwargs=write_kwargs if write_kwargs is not None else {},
            created=datetime.now(),
        )

    @property
    def id(self) -> ArtifactID:
        """Converts the Passport to an ID."""
        return ArtifactID.from_passport(passport=self)

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactPassport:
        """Creates a ArtifactPassport from a dictionary."""
        data["name"] = str(data["name"])
        data["description"] = str(data["description"])
        data["asset_type"] = AssetType(data["asset_type"])
        data["stage"] = ArtifactStage(data["stage"])
        data["file_format"] = FileFormat(data["file_format"])
        data["read_kwargs"] = dict(data.get("read_kwargs", {}))
        data["write_kwargs"] = dict(data.get("write_kwargs", {}))
        if data.get("created"):
            data["created"] = datetime.strptime(data["created"], "%Y%m%d-%H%M")

        return cls(**data)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ArtifactID(ID):
    """An immutable, unique identifier for a artifact asset."""

    name: str
    stage: ArtifactStage
    asset_type: AssetType = AssetType.ARTIFACT

    def label(self) -> str:
        return f"{str(self.asset_type).capitalize()} {self.name} of the {self.stage.value} stage"

    @classmethod
    def from_passport(cls, passport: ArtifactPassport) -> ArtifactID:
        """Creates an ArtifactID from a ArtifactPassport."""
        passport = cast(ArtifactPassport, passport)

        return cls(
            name=passport.name,
            asset_type=passport.asset_type,
            stage=passport.stage,
        )
