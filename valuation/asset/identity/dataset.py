#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/identity/dataset.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Tuesday October 21st 2025 02:20:20 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Any, Dict, Optional, cast

from dataclasses import dataclass, field
from datetime import datetime

from valuation.asset.identity.base import ID, Passport
from valuation.core.entity import Entity
from valuation.core.file import FileFormat
from valuation.core.stage import DatasetStage
from valuation.core.types import AssetType


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetPassport(Passport):
    """An immutable, unique idasset for a dataset."""

    entity: Optional[Entity] = field(default=Entity.SALES)

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        entity: Entity,
        stage: DatasetStage,
        file_format: FileFormat,
        read_kwargs: Optional[Dict[str, str]] = None,
        write_kwargs: Optional[Dict[str, str]] = None,
    ) -> DatasetPassport:
        """Creates a DatasetPassport."""
        return cls(
            name=name,
            description=description,
            asset_type=AssetType.DATASET,
            entity=entity,
            stage=stage,
            file_format=file_format,
            read_kwargs=read_kwargs if read_kwargs is not None else {},
            write_kwargs=write_kwargs if write_kwargs is not None else {},
            created=datetime.now(),
        )

    @property
    def id(self) -> DatasetID:
        """Converts the Passport to an ID."""
        return DatasetID.from_passport(passport=self)

    @classmethod
    def from_dict(cls, data: dict) -> DatasetPassport:
        """Creates a DatasetPassport from a dictionary."""
        data["name"] = str(data["name"])
        data["description"] = str(data["description"])
        data["asset_type"] = AssetType(data["asset_type"])
        data["entity"] = Entity(data["entity"])
        data["stage"] = DatasetStage(data["stage"])
        data["file_format"] = FileFormat(data["file_format"])
        data["read_kwargs"] = dict(data.get("read_kwargs", {}))
        data["write_kwargs"] = dict(data.get("write_kwargs", {}))
        if data.get("created"):
            data["created"] = datetime.strptime(data["created"], "%Y%m%d-%H%M")

        return cls(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Converts the Passport to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "asset_type": str(self.asset_type),
            "entity": str(self.entity),
            "stage": str(self.stage),
            "file_format": str(self.file_format),
            "read_kwargs": self.read_kwargs,
            "write_kwargs": self.write_kwargs,
            "created": self.created.strftime("%Y%m%d-%H%M") if self.created else "",
        }


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetID(ID):
    """An immutable, unique identifier for a dataset asset."""

    name: str
    asset_type: AssetType
    entity: Entity
    stage: DatasetStage

    def label(self) -> str:
        return f"{str(self.entity).capitalize()} {str(self.asset_type).capitalize()} {self.name} of the {self.stage.value} stage"

    @classmethod
    def from_passport(cls, passport: DatasetPassport) -> DatasetID:
        """Creates an DatasetID from a DatasetPassport."""
        passport = cast(DatasetPassport, passport)

        return cls(
            name=passport.name,
            asset_type=passport.asset_type,
            entity=passport.entity if passport.entity is not None else "",
            stage=passport.stage,
        )
