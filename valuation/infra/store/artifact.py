#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/store/artifact.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 25th 2025 01:58:06 am                                              #
# Modified   : Saturday October 25th 2025 10:29:35 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Manages the Artifact Store."""

from typing import Optional

from valuation.asset.artifact import Artifact
from valuation.asset.identity.artifact import ArtifactID, ArtifactPassport
from valuation.asset.identity.base import Passport
from valuation.core.types import AssetType
from valuation.infra.file.artifact import ArtifactFileSystem
from valuation.infra.store.base import AssetStoreBase

# ------------------------------------------------------------------------------------------------ #


# ------------------------------------------------------------------------------------------------ #
class ArtifactStore(AssetStoreBase):
    """Store implementation for artifact assets.

    Initializes the AssetStoreBase with a ArtifactFileSystem.

    Args:
        filesystem (ArtifactFileSystem): The filesystem helper used by the base store.
    """

    _file_system: ArtifactFileSystem

    def __init__(self) -> None:
        """Initialize the ArtifactStore.

        Note:
            The superclass is initialized with a ArtifactFileSystem instance.

        Returns:
            None
        """
        super().__init__(filesystem=ArtifactFileSystem())

    @property
    def asset_type(self) -> AssetType:
        """The asset type managed by this store.

        Returns:
            AssetType: The AssetType for artifacts (AssetType.DATASET).
        """
        return AssetType.ARTIFACT

    @property
    def file_system(self) -> ArtifactFileSystem:
        """The artifact file system."""

        return self._file_system

    def add(self, artifact: Artifact, overwrite: bool = False) -> None:
        """Add a artifact to the store.

        Saves the artifact passport and artifact data to the configured filesystem.

        Args:
            artifact (Artifact): The artifact instance to add.
            overwrite (bool): If True, overwrite an existing artifact with the same name. Defaults to False.

        Returns:
            None

        Raises:
            FileExistsError: If the artifact already exists and overwrite is False.
            RuntimeError: If operation disallowed in current MODE (e.g., modifying raw data in prod).
        """
        super().add(asset=artifact, overwrite=overwrite)

    def get(self, passport: ArtifactPassport) -> Optional[Artifact]:
        """Retrieve a artifact from the store by its passport.

        Args:
            passport (ArtifactPassport): The passport of the artifact to retrieve.

        Returns:
            Artifact: The retrieved Artifact instance.
        """
        # Instantiate the appropriate asset type
        return super().get(passport=passport)  # type: ignore

    def get_passport(self, artifact_id: ArtifactID) -> Optional[Passport]:
        """Retrieve an asset passport by its ID.

        Args:
            asset_id (ID): Identifier containing name and stage for the asset.

        Returns:
            Optional[Passport]: The reconstructed Passport instance.

        Raises:
            FileNotFoundError: If the passport file for the requested asset does not exist.
        """
        return super().get_passport(asset_id=artifact_id)

    def remove(self, passport: ArtifactPassport, **kwargs) -> None:
        """Remove a artifact from the store by its passport.

        Deletes both the artifact data file (if present) and its passport.

        Args:
            passport (ArtifactPassport): The passport of the artifact to remove.
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            None

        Raises:
            RuntimeError: If deletion of raw data is disallowed in current MODE.
        """
        super().remove(asset_id=passport.id, **kwargs)

    def exists(self, artifact_id: ArtifactID, **kwargs) -> bool:
        """Check if a artifact exists in the store by its ID.

        Args:
            artifact_id (ArtifactID): The unique identifier of the artifact (id component of passport).
            **kwargs: Additional backend-specific keyword arguments (ignored by base).

        Returns:
            bool: True if the artifact exists, False otherwise.
        """
        return super().exists(asset_id=artifact_id)
