#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/asset/dataset.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 07:11:18 pm                                               #
# Modified   : Saturday October 18th 2025 08:20:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides data utilities."""
from __future__ import annotations

from typing import Optional, Union

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from valuation.asset.base import Asset, Passport
from valuation.asset.identity import AssetType
from valuation.infra.db.file_system import FileSystem
from valuation.core.data import DataClass
from valuation.utils.io.service import IOService


# ------------------------------------------------------------------------------------------------ #
#                                         FILE INFO                                                #
# ------------------------------------------------------------------------------------------------ #
@dataclass
class FileInfo(DataClass):
    """A container for metadata related to a file on the filesystem.

    Attributes:
        filepath: The absolute path to the file.
        filename: The name of the file including its extension.
        file_size_mb: The size of the file in megabytes.
        created_timestamp: The creation timestamp of the file.
        modified_timestamp: The last modification timestamp of the file.
    """

    filepath: Path
    filename: str
    file_size_mb: float
    created_timestamp: float
    modified_timestamp: float

    @property
    def is_stale(self) -> bool:
        """Determines if the file on disk is newer than when this info was captured.

        Compares the file's current modification time on disk with the timestamp
        captured when this FileInfo object was created.

        Returns:
            True if the file has been modified since this object was created,
            False otherwise.
        """
        current_mod_time = self.filepath.stat().st_mtime
        return current_mod_time > self.modified_timestamp

    @classmethod
    def from_filepath(cls, filepath: Union[Path, str]) -> FileInfo:
        """Creates a FileInfo instance by reading metadata from a file path.

        Args:
            filepath: The path to the file to be profiled.

        Returns:
            A new instance of FileInfo populated with the file's metadata.
        """
        filepath = Path(filepath)
        stat = filepath.stat()
        file_size_mb = stat.st_size / (1024 * 1024)

        # Use st_birthtime for creation time where available (macOS, some Linux/Windows)
        # Fall back to st_ctime on other Unix systems.
        created_timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
        modified_timestamp = stat.st_mtime

        return cls(
            filepath=filepath,
            filename=filepath.name,
            file_size_mb=file_size_mb,
            created_timestamp=created_timestamp,
            modified_timestamp=modified_timestamp,
        )


# ------------------------------------------------------------------------------------------------ #
#                                       DATASET PROFILE                                            #
# ------------------------------------------------------------------------------------------------ #


@dataclass
class DatasetProfile(DataClass):
    """A container for profiling information about a DataFrame's content.

    Attributes:
        nrows: The total number of rows in the DataFrame.
        ncols: The total number of columns in the DataFrame.
        n_duplicates: The number of duplicate rows.
        missing_values: The total count of missing (NaN) values.
        missing_values_pct: The percentage of cells that are missing values.
        memory_usage_mb: The total memory usage of the DataFrame in megabytes.
        info: A summary DataFrame containing per-column statistics.
    """

    nrows: int
    ncols: int
    n_duplicates: int
    missing_values: int
    missing_values_pct: float
    memory_usage_mb: float
    info: pd.DataFrame

    @classmethod
    def create(cls, df: pd.DataFrame) -> DatasetProfile:
        """Creates a DatasetProfile instance by analyzing a DataFrame.

        Args:
            df: The pandas DataFrame to profile.

        Returns:
            A new instance of DatasetProfile with calculated metrics.
        """
        nrows = len(df)
        ncols = len(df.columns)
        n_duplicates = df.duplicated().sum()
        missing_values = df.isnull().sum().sum()
        total_cells = nrows * ncols
        missing_values_pct = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        info_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Non-Null Count": df.notnull().sum().values,
                "Null Count": df.isnull().sum().values,
                "Dtype": df.dtypes.values,
            }
        )

        return cls(
            nrows=nrows,
            ncols=ncols,
            n_duplicates=n_duplicates,
            missing_values=missing_values,
            missing_values_pct=missing_values_pct,
            memory_usage_mb=memory_usage_mb,
            info=info_df,
        )


# ------------------------------------------------------------------------------------------------ #
#                                           DATASET                                                #
# ------------------------------------------------------------------------------------------------ #


class Dataset(Asset):
    
    __ASSET_TYPE = AssetType.DATASET

    def __init__(
        self,
        passport: Passport,
        df: pd.DataFrame = pd.DataFrame(),
        io: type[IOService] = IOService,
    ) -> None:
        self._passport = passport
        self._df = df
        self._io = io()
        self._file_system = FileSystem(Dataset.__ASSET_TYPE)

        self._fileinfo: Optional[FileInfo] = None
        self._profile: Optional[DatasetProfile] = None

    @property
    def data(self) -> pd.DataFrame:
        """The dataset as a Pandas DataFrame, loaded lazily."""
        self.refresh_data()
        self._df = self._df if self._df is not None else pd.DataFrame()
        return self._df

    @property
    def passport(self) -> Passport:
        """The dataset's unique idasset."""
        return self._passport

    @property
    def profile(self) -> Optional[DatasetProfile]:
        """A profile of the dataset's content, generated lazily."""
        self.refresh_profile()
        return self._profile

    @property
    def fileinfo(self) -> Optional[FileInfo]:
        """Metadata for the associated file, generated lazily."""
        self.refresh_fileinfo()
        return self._fileinfo

    @property
    def nrows(self) -> Optional[int]:
        """Returns the number of rows in the dataset."""
        return self.data.shape[0] if self.data is not None else 0

    @property
    def ncols(self) -> Optional[int]:
        """Returns the number of rows in the dataset."""
        return self.data.shape[1] if self.data is not None else 0

    def stamp_passport(self, passport: Passport) -> None:
        """Updates the dataset's passport.

        Args:
            passport: The new Passport to assign to the dataset.
        """
        self._passport = passport

    def refresh_data(self, force: bool = False) -> None:
        """Refreshes the in-memory DataFrame from the file if it is stale or missing.

        This method implements the lazy-loading logic. It will trigger a `load`
        operation if the DataFrame is empty, has never been loaded, is detected
        as stale, or if the refresh is forced.

        Args:
            force: If True, forces the data to be reloaded from the file
                regardless of its current state.
        """
        stale = self._fileinfo.is_stale if self._fileinfo else False
        if self._df is None or self._df.empty or stale or force:
            asset_filepath = self._file_system.get_(location=self.
            if self._passport.filepath:
                self.load()
            else:
                logger.warning("No filepath set; cannot refresh data.")

    def refresh_profile(self, force: bool = False) -> None:
        """Refreshes the dataset profile if it is missing or stale.

        Args:
            force: If True, forces the profile to be re-calculated.
        """
        stale = self._fileinfo.is_stale if self._fileinfo else False
        if self._profile is None or stale or force:
            # Guard against _df being None before checking .empty to satisfy static type checkers.
            if self._df is not None and not self._df.empty:
                self._profile = DatasetProfile.create(df=self._df)

    def refresh_fileinfo(self, force: bool = False) -> None:
        """Refreshes the file information if it is missing or stale.

        Args:
            force: If True, forces the file metadata to be re-read.
        """
        stale = self._fileinfo.is_stale if self._fileinfo else False
        if (self._fileinfo is None or stale or force) and self._passport.filepath:
            self._fileinfo = FileInfo.from_filepath(filepath=self._passport.filepath)

    def load(self, dtypes: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Loads data from the source filepath into the internal DataFrame.

        This method uses the injected IO service to read the file. It can also
        enforce specific data types on the loaded columns.

        Args:
            dtypes: An optional dictionary mapping column names to desired
                data types (e.g., {'id': 'str'}).
            **kwargs: Additional keyword arguments to pass to the IO service's
                read method.
        """
        logger.debug(f"Loading data from {self._passport.filepath}")
        if not self._passport.filepath:
            raise ValueError("Filepath is not set. Cannot load data.")

        self._df = self._io.read(filepath=self._passport.filepath, **kwargs)

        if dtypes is not None and self._df is not None and not self._df.empty:
            valid_dtypes = {k: v for k, v in dtypes.items() if k in self._df.columns}
            self._df = self._df.astype(valid_dtypes)

    def save(self, overwrite: bool = False, **kwargs) -> None:
        """Saves the in-memory DataFrame to its canonical filepath.

        Fails safely by default if a file already exists at the location.

        Args:
            overwrite: If True, allows overwriting an existing file.
            **kwargs: Additional keyword arguments to pass to the IO service's
                write method.

        Raises:
            ValueError: If the Dataset has no canonical filepath set.
            FileConflictError: If the file exists and `overwrite` is False.
        """
        if not self._passport.filepath:
            raise ValueError("Filepath is not set. Use save_as() instead.")
        self.save_as(self._passport.filepath, overwrite=overwrite, **kwargs)

    def save_as(self, filepath: Union[Path, str], overwrite: bool = False, **kwargs) -> None:
        """Saves the in-memory DataFrame to a specified location.

        Fails safely by default if a file already exists at the location.

        Args:
            filepath: The target location to save the file.
            overwrite: If True, allows overwriting an existing file.
            **kwargs: Additional keyword arguments to pass to the IO service's
                write method.

        Raises:
            FileConflictError: If the file exists and `overwrite` is False.
        """
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise DatasetExistsError(
                f"File {filepath} already exists. Set overwrite=True to replace it."
            )
        logger.debug(f"Saving data to {filepath}")
        self._io.write(data=self._df, filepath=filepath, **kwargs)

    def delete(self) -> None:
        """Deletes the file associated with this Dataset from the filesystem."""
        if not self._passport.filepath:
            raise ValueError("Filepath is not set. No file to delete.")
        logger.debug(f"Deleting file {self._passport.filepath}")
        self._passport.filepath.unlink(missing_ok=True)

    def exists(self) -> bool:
        """Checks if a file exists at the Dataset's canonical filepath.

        Returns:
            True if the file exists, False otherwise. Returns False if no
            filepath is associated with the Dataset.
        """
        if not self._passport.filepath:
            return False
        return self._passport.filepath.exists()
