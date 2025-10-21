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
# Modified   : Tuesday October 21st 2025 02:22:14 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Any, Dict, cast

from dataclasses import dataclass, field
from datetime import datetime

from valuation.core.dataclass import DataClass
from valuation.core.file import FileFormat
from valuation.core.stage import Stage
from valuation.core.types import AssetType


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Passport(DataClass):
    """An immutable, unique idasset for an asset."""

    name: str
    description: str
    asset_type: AssetType
    stage: Stage
    created: datetime
    file_format: FileFormat
    read_kwargs: Dict[str, str] = field(default_factory=dict)
    write_kwargs: Dict[str, str] = field(default_factory=dict)

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
        file_format: FileFormat,
        read_kwargs: Dict[str, str] = {},
        write_kwargs: Dict[str, str] = {},
    ) -> Passport:
        """Creates a Passport."""
        return cls(
            name=name,
            description=description,
            asset_type=asset_type,
            stage=stage,
            created=datetime.now(),
            file_format=file_format,
            read_kwargs=read_kwargs,
            write_kwargs=write_kwargs,
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
        data["read_kwargs"] = dict(data.get("read_kwargs", {}))
        data["write_kwargs"] = dict(data.get("write_kwargs", {}))
        if data.get("created"):
            data["created"] = datetime.strptime(data["created"], "%Y%m%d-%H%M%S")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Passport to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "asset_type": str(self.asset_type),
            "stage": str(self.stage),
            "file_format": str(self.file_format),
            "read_kwargs": self.read_kwargs,
            "write_kwargs": self.write_kwargs,
            "created": self.created.strftime("%Y%m%d-%H%M%S") if self.created else "",
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
