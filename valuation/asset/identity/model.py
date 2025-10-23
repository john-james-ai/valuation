#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/identity/model.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 04:11:40 pm                                              #
# Modified   : Thursday October 23rd 2025 04:18:41 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""

from __future__ import annotations

from typing import Any, Dict, Optional, cast

from dataclasses import dataclass
from datetime import datetime

from valuation.asset.identity.base import ID, Passport
from valuation.core.file import FileFormat
from valuation.core.stage import ModelStage
from valuation.core.types import AssetType


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelPassport(Passport):
    """An immutable, unique idasset for a model."""

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        asset_type: AssetType = AssetType.MODEL,
        stage: ModelStage = ModelStage.FINAL,
        file_format: FileFormat = FileFormat.JSON,
        read_kwargs: Optional[Dict[str, str]] = None,
        write_kwargs: Optional[Dict[str, str]] = None,
    ) -> ModelPassport:
        """Creates a ModelPassport."""
        return cls(
            name=name,
            description=description,
            asset_type=asset_type,
            stage=stage,
            file_format=file_format,
            read_kwargs=read_kwargs if read_kwargs is not None else {},
            write_kwargs=write_kwargs if write_kwargs is not None else {},
            created=datetime.now(),
        )

    @property
    def id(self) -> ModelID:
        """Converts the Passport to an ID."""
        return ModelID.from_passport(passport=self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelPassport:
        """Creates a ModelPassport from a dictionary."""
        data["name"] = str(data["name"])
        data["description"] = str(data["description"])
        data["asset_type"] = AssetType(data["asset_type"])
        data["stage"] = ModelStage(data["stage"])
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
            "stage": str(self.stage),
            "file_format": str(self.file_format),
            "read_kwargs": self.read_kwargs,
            "write_kwargs": self.write_kwargs,
            "created": self.created.strftime("%Y%m%d-%H%M") if self.created else "",
        }


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelID(ID):
    """An immutable, unique identifier for a model asset."""

    name: str
    stage: ModelStage
    asset_type: AssetType = AssetType.MODEL

    def label(self) -> str:
        return f"{str(self.asset_type).capitalize()} {self.name} of the {self.stage.value} stage"

    @classmethod
    def from_passport(cls, passport: ModelPassport) -> ModelID:
        """Creates an ModelID from a ModelPassport."""
        passport = cast(ModelPassport, passport)

        return cls(
            name=passport.name,
            asset_type=passport.asset_type,
            stage=passport.stage,
        )
