#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/identity.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 18th 2025 08:20:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Dict, Optional

from dataclasses import dataclass
from datetime import datetime

from valuation.asset.stage import Stage
from valuation.asset.types import AssetType
from valuation.core.data import DataClass


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Passport(DataClass):
    """An immutable, unique idasset for an asset."""

    name: str
    description: str
    asset_type: AssetType
    stage: Stage
    asset_format: str
    created: Optional[datetime] = None

    @property
    def label(self) -> str:
        """Returns a string label for the Passport."""
        created = (
            f"created on {self.created.strftime('%Y-%m-%d')} at {self.created.strftime('%H:%M')}"
            if self.created
            else ""
        )
        return f"{self.asset_type.value.capitalize()} {self.name} of the  {self.stage.value} stage {created}"

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        stage: Stage,
        asset_type: AssetType,
        asset_format: str = "csv",
    ) -> Passport:
        """Creates a Passport."""
        return cls(
            name=name,
            description=description,
            asset_type=asset_type,
            stage=stage,
            asset_format=asset_format,
            created=datetime.now(),
        )

    @classmethod
    def from_dict(cls, data: dict) -> Passport:
        """Creates a Passport from a dictionary."""
        data["name"] = str(data["name"])
        data["description"] = str(data["description"])
        data["asset_type"] = AssetType(data["asset_type"])
        data["stage"] = Stage(data["stage"])
        data["asset_format"] = str(data["asset_format"])
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
            "asset_format": self.asset_format,
            "created": self.created.strftime("%Y%m%d-%H%M") if self.created else "",
        }
