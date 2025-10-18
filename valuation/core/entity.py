#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/entity.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 18th 2025 06:34:57 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Valuation package."""
from __future__ import annotations

from typing import Optional

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from valuation.config import filepaths
from valuation.utils.data import DataClass


# ------------------------------------------------------------------------------------------------ #
class Entity(ABC):
    """
    An abstract base class for any object with a persistent state and a unique identity.
    """

    def __init__(self, passport: Passport) -> None:
        self._passport = passport

    @property
    def name(self) -> str:
        """The entity's name."""
        return self._passport.name

    @property
    def passport(self) -> Passport:
        """The entity's unique and immutable passport."""
        return self._passport

    @abstractmethod
    def load(self) -> None:
        """Loads the entity's data from the filepath specified in its passport."""
        # Example: self.data = pd.read_csv(self.filepath)
        pass

    @abstractmethod
    def save(self) -> None:
        """Saves the entity's data to the filepath specified in its passport."""
        # Example: self.data.to_csv(self.filepath)
        pass


# ------------------------------------------------------------------------------------------------ #
class EntityType(Enum):
    """Defines the recognized types of Entities."""

    DATASET = "dataset"
    MODEL = "model"
    REPORT = "report"
    PLOT = "plot"

    def __str__(self):
        return self.value


# ------------------------------------------------------------------------------------------------ #
class Stage(Enum):
    """Defines the recognized types of Entities."""


class DatasetStage(Stage):

    RAW = "raw"
    INGEST = "ingest"
    CLEAN = "clean"
    PROCESSED = "processed"
    FEATURES = "feature_engineered"
    FINAL = "final"

    def __str__(self):
        return self.value


class ModelStage(Stage):
    INITIAL = "initial"
    TUNED = "tuned"
    FINAL = "final"

    def __str__(self):
        return self.value


@dataclass
class Passport(DataClass):
    """An immutable, unique identity for an entity."""

    name: str
    description: str
    stage: Stage
    type: EntityType
    format: str
    filepath: Optional[Path] = None
    created: Optional[datetime] = None
    completed: Optional[datetime] = None
    loaded: Optional[datetime] = None
    cost: Optional[float] = None

    def complete(self, created: datetime, completed: datetime, cost: float) -> None:
        """Stamps the entity with creation and completion timestamps and cost."""
        self.created = created
        self.completed = completed
        self.cost = cost

    def load(self) -> None:
        """Stamps the entity with a loaded timestamp."""
        self.loaded = datetime.now()

    @classmethod
    def create(
        cls,
        name: str,
        description: str,
        stage: Stage,
        type: EntityType = EntityType.DATASET,
        filepath: Optional[Path] = None,
        format: str = "csv",
    ) -> Passport:
        """Creates a Passport, inferring name from filepath if not provided."""
        filepath = (
            filepath
            or cls(
                name=name, description=description, type=type, stage=stage, format=format
            ).get_filepath()
        )
        return cls(
            name=name,
            description=description,
            type=type,
            stage=stage,
            format=format,
            filepath=filepath,
        )

    def get_filepath(self) -> Path:
        """Generates a standardized filepath for the entity based on its attributes."""
        dt = datetime.now().strftime("%Y%m%d%H%M%S")
        base_dir = filepaths.ENTITY_STORE_DIR / self.type.value / self.stage.value
        filename = f"{self.name}_{dt}.{self.format}"
        return base_dir / filename
