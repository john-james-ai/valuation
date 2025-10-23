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
# Modified   : Thursday October 23rd 2025 04:32:29 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Any, Dict, Optional

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from valuation.core.dataclass import DataClass
from valuation.core.file import FileFormat
from valuation.core.stage import Stage
from valuation.core.types import AssetType


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Passport(DataClass, ABC):
    """An immutable, unique idasset for an asset."""

    name: str
    description: str
    asset_type: AssetType
    stage: Stage
    created: datetime
    file_format: Optional[FileFormat] = None
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
        file_format: Optional[FileFormat] = None,
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
    @abstractmethod
    def id(self) -> ID:
        """Converts the Passport to an ID."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict) -> Passport:
        """Creates a Passport from a dictionary."""
        pass

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
    """Base Class for Asset IDs."""
