#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/db/base.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 11:19:18 pm                                                #
# Modified   : Saturday October 18th 2025 04:39:39 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Dataset Store."""
from typing import Any, Dict, Optional, Union, overload

from abc import ABC, abstractmethod
from dataclasses import asdict
import json
from pathlib import Path

from loguru import logger
import pandas as pd

from valuation import Entity
from valuation.archive.io.base import IOService
from valuation.config.filepaths import ENTITY_STORE_DIR
from valuation.utils.data import Dataset
from valuation.utils.exception import EntityStoreNotFoundError
from valuation.utils.identity import EntityType, Passport, Stage


# ------------------------------------------------------------------------------------------------ #
class EntityStore(ABC):
    """Abstract base class for entity storage backends.

    Manages a directory-based store for serialized entity metadata (JSON files).

    Attributes:
        _location (Path): Filesystem path where entity JSON files are stored.
        _io (IOService): IO service used to read/write passport files.
    """

    __entity_map = {"dataset": Dataset}

    def __init__(self, location: Optional[Path] = None, io: IOService = IOService) -> None:
        """Initialize the EntityStore."""
        self._location = Path(location) or ENTITY_STORE_DIR
        self._location.mkdir(parents=True, exist_ok=True)
        self._io = io

    @property
    @abstractmethod
    def entity_type(self) -> EntityType:
        """Type of entity managed by this store.

        Returns:
            EntityType: The enum value representing the entity type handled by the store.
        """
        pass

    @property
    def registry(self) -> pd.DataFrame:
        """List all entity passports in the store as a DataFrame.

        Iterates over JSON files in the store directory, parses them, and returns
        a pandas DataFrame summarizing the registry entries. Malformed JSON files
        are skipped with a warning.

        Returns:
            pandas.DataFrame: DataFrame containing registry information for each valid JSON file.

        Raises:
            EntityStoreNotFoundError: If the configured store directory does not exist.
        """
        registry = []

        if not self._location.is_dir():
            raise EntityStoreNotFoundError(f"Error: Directory not found at '{self._location}'")

        registry = [
            data
            for path in self._location.glob("*.json")
            if (data := self._read_registry(path)) is not None
        ]

        return pd.DataFrame(registry)

    @abstractmethod
    def add(self, entity: Entity, overwrite: bool = False) -> None:
        """Add an entity to the store.

        Serializes and persists the entity's passport and triggers the entity to save its data.

        Args:
            entity (Entity): The entity instance to add.
            overwrite (bool, optional): If True, overwrite an existing entity with the same name.
                Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If an entity with the same passport already exists and overwrite is False.
        """
        passport_filepath = self.get_passport_filepath(name_or_passport=entity.passport)

        if passport_filepath.exists() and not overwrite:
            raise FileExistsError(
                f"{entity.passport.type.value.capitalize()} already exists in the store."
            )

        # Save passport
        self._io.write(filepath=passport_filepath, data=asdict(entity.passport))

        # Save entity data
        entity.save()

    @abstractmethod
    def get(self, name: str, stage: Stage) -> Entity:
        """Retrieve an entity from the store by name and stage.

        Args:
            name (str): The name of the entity to retrieve.
            stage (Stage): The stage of the entity to retrieve.

        Returns:
            Entity: The retrieved entity instance.

        Raises:
            FileNotFoundError: If the passport file for the requested entity does not exist.
        """
        # Get the filepath for the passport
        filepath = self.get_passport_filepath(
            name_or_passport=name, type=self.entity_type, stage=stage
        )
        # Obtain the passport
        passport = self._get_passport(filepath=filepath)
        # Instantiate the appropriate entity type
        entity_class = self.__entity_map[self.entity_type.value]
        entity = entity_class(passport=passport)
        return entity

    @abstractmethod
    def remove(self, name: str, stage: Stage) -> None:
        """Removes an entity from the store by name and stage.

        Deletes both the entity data file (if present) and its passport.

        Args:
            name (str): The name of the entity to remove.
            stage (Stage): The stage of the entity to remove.

        Returns:
            None

        Raises:
            FileNotFoundError: If the passport file for the requested entity does not exist.
        """
        passport_filepath = self.get_passport_filepath(
            name_or_passport=name, type=self.entity_type, stage=stage
        )

        # Get the passport
        passport = self._get_passport(filepath=passport_filepath)

        # Get the entity data filepath
        entity_filepath = passport.filepath

        # Remove the entity data file
        if entity_filepath and entity_filepath.exists():
            entity_filepath.unlink()
            logger.info(
                f"Entity data file for {name}, stage: {stage.value} is removed from the store."
            )

        # Remove the passport file
        if passport_filepath and passport_filepath.exists():
            passport_filepath.unlink()
            logger.info(f"Passport for {name}, stage: {stage.value} is removed from the store.")
        logger.info(f"Entity '{name}' removed from the store.")

    @abstractmethod
    def exists(self, name: str, stage: Stage) -> bool:
        """Checks if an entity exists in the store by name and stage.

        Args:
            name (str): The name of the entity to check.
            stage (Stage): The stage of the entity to check.
        Returns:
            bool: True if the entity exists, False otherwise.

        """
        passport_filepath = self.get_passport_filepath(
            name_or_passport=name, type=self.entity_type, stage=stage
        )
        return passport_filepath.exists()

    def _read_registry(self, filepath: Path) -> Optional[Dict[str, Any]]:
        """Safely read and parse a single JSON registry file.

        Args:
            filepath (Path): The path to the JSON file.

        Returns:
            Optional[Dict[str, Any]]: A dictionary of the parsed JSON data, or None if parsing fails.
        """
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Could not parse '{filepath.name}'. Skipping. Reason: {e}")
            return None

    @overload
    def get_passport_filepath(self, name_or_passport: Passport) -> Path:
        """Generate the standardized filepath for a given Passport.

        Args:
            name_or_passport (Passport): The passport instance for which to generate the filepath.

        Returns:
            Path: The path to the passport JSON file.
        """

    @overload
    def get_passport_filepath(self, name_or_passport: str, type: EntityType, stage: Stage) -> Path:
        """Generate the standardized filepath for a passport described by name, type, and stage.

        Args:
            name_or_passport (str): The entity name.
            type (EntityType): The entity type.
            stage (Stage): The entity stage.

        Returns:
            Path: The path to the passport JSON file.
        """

    def get_passport_filepath(
        self,
        name_or_passport: Union[Passport, str],
        type: Optional[EntityType] = None,
        stage: Optional[Stage] = None,
    ) -> Path:
        """Construct the passport JSON filepath from a Passport or from name/type/stage.

        Args:
            name_or_passport (Union[Passport, str]): Either a Passport instance or the entity name.
            type (Optional[EntityType]): The entity type (required if name_or_passport is str).
            stage (Optional[Stage]): The entity stage (required if name_or_passport is str).

        Returns:
            Path: The full Path to the passport JSON file in the store.

        Raises:
            ValueError: If arguments are invalid (neither Passport nor complete name/type/stage provided).
        """
        if isinstance(name_or_passport, Passport):
            p = name_or_passport
            name = p.name
            entity_type = p.type.value
            stage_value = p.stage.value

        elif (
            isinstance(name_or_passport, str)
            and isinstance(type, EntityType)
            and isinstance(stage, Stage)
        ):
            name = name_or_passport
            entity_type = type.value
            stage_value = stage.value

        else:
            raise ValueError(
                "Invalid arguments for get_passport_filepath; expected either a Passport or (name, type, stage)."
            )

        return Path(self._location) / f"{entity_type}_{stage_value}_{name}.json"

    def _get_passport(self, filepath: Path) -> Passport:
        """Retrieve the passport dictionary for a given entity by filepath.

        Args:
            filepath (Path): The path to the passport JSON file.

        Returns:
            Passport: The Passport instance reconstructed from the JSON file.

        Raises:
            FileNotFoundError: If the passport file does not exist at the provided filepath.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                passport_dict = json.load(f)
                return Passport(**passport_dict)
        except FileNotFoundError:
            raise FileNotFoundError(f"Entity passport not found at '{filepath}'")
