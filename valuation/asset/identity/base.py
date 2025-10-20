#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/identity/base.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Monday October 20th 2025 12:48:03 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Dict, Optional, cast

from dataclasses import dataclass
from datetime import datetime

from valuation.asset.stage import Stage
from valuation.asset.types import AssetType
from valuation.core.structure import DataClass
from valuation.infra.file.base import FileFormat


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Passport(DataClass):
    """An immutable, unique idasset for an asset."""

    name: str
    description: str
    asset_type: AssetType
    stage: Stage
    file_format: FileFormat = FileFormat.PARQUET
    created: Optional[datetime] = None

    @property
    def label(self) -> str:
        """Returns a string label for the Passport."""
        created = (
            f"created on {self.created.strftime('%Y-%m-%d')} at {self.created.strftime('%H:%M')}"
            if self.created
            else ""
        )
        return f"{self.asset_type.value.capitalize()} {self.name} of the {self.stage.value} stage {created}"

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        asset_type: AssetType,
        stage: Stage,
        file_format: FileFormat = FileFormat.PARQUET,
    ) -> Passport:
        """Creates a Passport."""
        return cls(
            name=name,
            description=description,
            asset_type=asset_type,
            stage=stage,
            file_format=file_format,
            created=datetime.now(),
        )

    @property
    def id(self) -> ID:
        """Converts the Passport to an ID."""
        return ID.from_passport(passport=self)

    @classmethod
    def from_dict(cls, data: dict) -> Passport:
        """Creates a Passport from a dictionary."""
        data["name"] = str(data["name"])
        data["description"] = str(data["description"])
        data["asset_type"] = AssetType(data["asset_type"])
        data["stage"] = Stage(data["stage"])
        data["file_format"] = FileFormat(data["file_format"])
        if data.get("created"):
            data["created"] = datetime.strptime(data["created"], "%Y%m%d-%H%M")

        return cls(**data)

    def to_dict(self) -> Dict[str, str]:
        """Converts the Passport to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "asset_type": str(self.asset_type),
            "stage": str(self.stage),
            "file_format": str(self.file_format),
            "created": self.created.strftime("%Y%m%d-%H%M") if self.created else "",
        }


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ID(DataClass):
    """An immutable, unique identifier for an asset."""

    name: str
    asset_type: AssetType
    stage: Stage

    def label(self) -> str:
        """Returns a string label for the ID."""
        return f"{str(self.asset_type).capitalize()} {self.name} of the {self.stage.value} stage"

    @classmethod
    def from_passport(cls, passport: Passport) -> ID:
        """Creates an ID from a Passport."""
        passport = cast(ID, passport)

        return cls(
            name=passport.name,
            asset_type=passport.asset_type,
            stage=passport.stage,
        )
