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
# Modified   : Saturday October 25th 2025 06:10:20 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides data utilities."""
from __future__ import annotations

from typing import Optional

from dataclasses import dataclass
from pathlib import Path
import shutil

from loguru import logger
import polars as pl

from valuation.asset.base import Asset, Passport
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.dataclass import DataClass
from valuation.core.types import AssetType
from valuation.infra.exception import DatasetExistsError
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.file.io.fast import IOService

# ------------------------------------------------------------------------------------------------ #
DTYPES = {}
DTYPES = {
    "category": pl.Utf8,
    "store": pl.Int64,
    "date": pl.Datetime,
    "upc": pl.Int64,
    "week": pl.Int64,
    "qty": pl.Int64,
    "move": pl.Int64,
    "ok": pl.Int64,
    "price": pl.Float64,
    "revenue": pl.Float64,
    "profit": pl.Float64,
    "year": pl.Int64,
    "start": pl.Datetime,
    "end": pl.Datetime,
    "gross_margin_pct": pl.Float64,
    "gross_margin": pl.Float64,
    "gross_profit": pl.Float64,
    "price_hex": pl.Utf8,
    "profit_hex": pl.Utf8,
}

DTYPES_CAPITAL = {k.capitalize(): v for k, v in DTYPES.items()}
DTYPES_UPPER = {k.upper(): v for k, v in DTYPES.items()}
DTYPES.update(DTYPES_CAPITAL)
DTYPES.update(DTYPES_UPPER)

# Derive helper column lists from polars DTYPES mapping
NUMERIC_TYPES = (pl.Int64, pl.Int32, pl.Float64, pl.Float32)
NUMERIC_COLUMNS = [k for k, v in DTYPES.items() if v in NUMERIC_TYPES]
DATETIME_COLUMNS = [k for k, v in DTYPES.items() if "Datetime" in str(v)]
STRING_COLUMNS = [k for k, v in DTYPES.items() if v == pl.Utf8]

NUMERIC_PLACEHOLDER = -1  # Placeholder for missing numeric values
STRING_PLACEHOLDER = "Unknown"  # Placeholder for missing string values
# ------------------------------------------------------------------------------------------------ #


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
        try:
            current_mod_time = self.filepath.stat().st_mtime
            return current_mod_time > self.modified_timestamp
        except Exception:
            return True

    @classmethod
    def from_filepath(cls, filepath: Path | str) -> FileInfo:
        """Creates a FileInfo instance by reading metadata from a file path.

        Args:
            filepath: The path to the file to be profiled.

        Returns:
            A new instance of FileInfo populated with the file's metadata.
        """
        filepath = Path(filepath)
        try:
            stat = filepath.stat()
            file_size_mb = stat.st_size / (1024 * 1024)

            # Use st_birthtime for creation time where available (macOS, some Linux/Windows)
            # Fall back to st_ctime on other Unix systems.
            created_timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
            modified_timestamp = stat.st_mtime
        except Exception:
            file_size_mb = 0.0
            created_timestamp = 0.0
            modified_timestamp = 0.0

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
    info: pl.DataFrame

    @classmethod
    def create(cls, df: pl.DataFrame) -> DatasetProfile:
        """Creates a DatasetProfile instance by analyzing a Polars DataFrame.

        Args:
            df (pl.DataFrame): The Polars DataFrame to profile.

        Returns:
            DatasetProfile: A new instance of DatasetProfile with calculated metrics.
        """
        # Basic counts
        nrows = df.height
        ncols = df.width

        # Duplicates: difference between total rows and unique row count
        n_duplicates = nrows - df.unique().height

        # Missing (null) values per column: pl.DataFrame.null_count() returns one row with counts
        null_counts_df = df.null_count()
        # null_counts_df is a single-row DataFrame: extract values
        null_counts_vals = [int(null_counts_df[row][0]) for row in range(0)] if False else None
        # Simpler: convert to series via to_dict
        null_counts = []
        try:
            # null_counts_df has column names matching original; collect as eager dict
            null_counts_dict = {
                col: int(null_counts_df.select(pl.col(col)).to_numpy()[0][0])
                for col in null_counts_df.columns
            }
        except Exception:
            # Fallback: use scanning via apply (less efficient)
            null_counts_dict = {
                col: int(df.select(pl.col(col).is_null().sum()).to_series()[0])
                for col in df.columns
            }

        missing_values = sum(null_counts_dict.values())

        total_cells = nrows * ncols
        missing_values_pct = (missing_values / total_cells) * 100 if total_cells > 0 else 0.0

        # Estimated memory usage (bytes -> MB)
        try:
            memory_usage_mb = df.estimated_size() / (1024 * 1024)
        except Exception:
            memory_usage_mb = 0.0

        # Construct per-column info as a Polars DataFrame
        cols = df.columns
        non_null_counts = [nrows - null_counts_dict.get(col, 0) for col in cols]
        null_counts_list = [null_counts_dict.get(col, 0) for col in cols]
        dtypes = [str(df.schema[col]) for col in cols]

        info_df = pl.DataFrame(
            {
                "Column": cols,
                "Non-Null Count": non_null_counts,
                "Null Count": null_counts_list,
                "Dtype": dtypes,
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

    def __init__(
        self,
        passport: DatasetPassport,
        df: pl.DataFrame | pl.LazyFrame = pl.LazyFrame(),
        io: type[IOService] = IOService,
    ) -> None:
        self._passport = passport
        self._df = df
        self._io = io()

        self._fileinfo = None
        self._profile = None
        self._asset_filepath = None
        self._file_system = DatasetFileSystem()

        self._validate()
        self._initialize()

    @property
    def data(self) -> pl.DataFrame | pl.LazyFrame:
        """The dataset as a Polars DataFrame, loaded lazily."""
        self._lazy_load_data()
        if isinstance(self._df, pl.LazyFrame):
            self._df = self._df.collect()
        return self._df

    @property
    def profile(self) -> Optional[DatasetProfile]:
        """A profile of the dataset's content, generated lazily."""
        self._lazy_load_profile()
        return self._profile

    @property
    def fileinfo(self) -> Optional[FileInfo]:
        """Metadata for the associated file, generated lazily."""
        return self._fileinfo

    @property
    def passport(self) -> Passport:
        """The dataset's unique idasset."""
        return self._passport

    @property
    def asset_type(self) -> AssetType:
        """The type of asset."""
        return AssetType.DATASET  # type: ignore

    @property
    def data_in_memory(self) -> bool:
        """Indicates if the DataFrame is loaded in memory."""
        if self._df is None:
            return False
        if isinstance(self._df, pl.LazyFrame):
            # LazyFrame is a query plan, not actual data in memory
            return False
        return self._df.height > 0

    @property
    def file_exists(self) -> bool:
        """Indicates if the dataset file exists on disk."""
        return self._asset_filepath.exists() if self._asset_filepath else False

    @property
    def file_fresh(self) -> bool:
        """Indicates if the file on disk is fresh compared to in-memory data."""
        return self._fileinfo and not self._fileinfo.is_stale if self._fileinfo else False

    @property
    def nrows(self) -> Optional[int]:
        """Returns the number of rows in the dataset."""
        if self.data is None:
            return 0
        if isinstance(self.data, pl.LazyFrame):
            # For LazyFrame, we need to collect to get the count
            # This could be expensive for large datasets
            try:
                return self.data.select(pl.len()).collect().item()
            except Exception:
                # If collection fails, return None
                return None
        return self.data.height

    @property
    def ncols(self) -> Optional[int]:
        """Returns the number of columns in the dataset."""
        if self.data is None:
            return 0
        # Use collect_schema() for LazyFrame to avoid warning
        if isinstance(self.data, pl.LazyFrame):
            return len(self.data.collect_schema())
        return self.data.width

    def stamp_passport(self, passport: Passport) -> None:
        """Updates the dataset's passport.

        Args:
            passport: The new Passport to assign to the dataset.
        """
        self._passport = passport

    def load(self) -> None:
        """Loads data from the source filepath into the internal DataFrame.

        This method uses the injected IO service to read the file. It can also
        enforce specific data types on the loaded columns.
        """
        try:
            self._df = self._io.read(filepath=self._asset_filepath, **self._passport.read_kwargs)
            self._df = self._normalize_dtypes(self._df)
            self._fileinfo = FileInfo.from_filepath(filepath=self._asset_filepath)
            logger.debug(f"Dataset {self.passport.label} loaded.")
        except FileNotFoundError:
            logger.warning(f"File {self._asset_filepath} not found. DataFrame is empty.")
            self._df = pl.DataFrame()

    def save(self, overwrite: bool = False) -> None:
        """Saves the in-memory DataFrame to its canonical filepath.

        Fails safely by default if a file already exists at the location.

        Args:
            overwrite: If True, allows overwriting an existing file.

        Raises:
            ValueError: If the Dataset has no canonical filepath set.
            FileConflictError: If the file exists and `overwrite` is False.
        """
        self.save_as(self._asset_filepath, overwrite=overwrite)

    def save_as(self, filepath: Path | str, overwrite: bool = False) -> None:
        """Saves the in-memory DataFrame to a specified location.

        Fails safely by default if a file already exists at the location.

        Args:
            filepath: The target location to save the file.
            overwrite: If True, allows overwriting an existing file.

        Raises:
            FileConflictError: If the file exists and `overwrite` is False.
        """
        filepath = Path(filepath)
        if filepath.exists() and not overwrite:
            raise DatasetExistsError(
                f"File {filepath} already exists. Set overwrite=True to replace it."
            )

        self._io.write(data=self._df, filepath=filepath, **self._passport.write_kwargs)
        logger.debug(f"Dataset {self.passport.name} saved to {filepath}")

    def delete(self) -> None:
        """Deletes the file associated with this Dataset from the filesystem."""
        if not self._asset_filepath:
            raise ValueError("Filepath is not set. No file to delete.")
        logger.debug(f"Deleting file(s) {self._asset_filepath}")
        if self._asset_filepath.is_file():
            self._asset_filepath.unlink(missing_ok=True)
        else:
            shutil.rmtree(self._asset_filepath, ignore_errors=True)
        self._fileinfo = None

    def exists(self) -> bool:
        """Checks if a file exists at the Dataset's canonical filepath.

        Returns:
            True if the file exists, False otherwise. Returns False if no
            filepath is associated with the Dataset.
        """
        if not self._asset_filepath:
            return False
        return self._asset_filepath.exists()

    def _has_data(self, df: pl.DataFrame | pl.LazyFrame) -> bool:
        """Check if a DataFrame or LazyFrame contains data.

        Args:
            df: The DataFrame or LazyFrame to check.

        Returns:
            True if the frame contains data, False otherwise.
        """
        if df is None:
            return False

        if isinstance(df, pl.LazyFrame):
            # For LazyFrame, check if it has any columns via collect_schema()
            # This is the recommended way to avoid the performance warning
            return len(df.collect_schema()) > 0

        # For DataFrame, check height
        return df.height > 0

    def _validate(self) -> None:
        """Validates the dataset's passport and filepath."""
        # Validate the passport.
        if not isinstance(self._passport, DatasetPassport):
            raise TypeError("passport must be an instance of DatasetPassport.")
        # Confirm that the passport is for a dataset asset.
        if self._passport.asset_type != AssetType.DATASET:
            raise ValueError("passport must be for a dataset asset.")

        # Get the canonical filepath from the file system.
        self._asset_filepath = self._file_system.get_asset_filepath(passport=self._passport)

        # If self._df is provided, yet the file exists, raise a FileExistsError to avoid overwriting.
        if self._has_data(self._df) and self._asset_filepath.exists():
            raise DatasetExistsError(
                f"Dataset file {self._asset_filepath} already exists. Cannot initialize with data to avoid overwriting."
            )
        # If self._df is empty and no file exists, raise an exception.
        if not self._has_data(self._df) and not self._asset_filepath.exists():
            raise FileNotFoundError(
                f"No data provided and dataset file {self._asset_filepath} does not exist."
            )

    def _initialize(self) -> None:
        """Initializes the dataset."""

        # If the filepath exists, set the fileinfo object
        if self._asset_filepath and self._asset_filepath.exists():
            self._fileinfo = FileInfo.from_filepath(filepath=self._asset_filepath)

        # If self._df exists and is not empty, normalize dtypes
        if self._has_data(self._df):
            self._df = self._normalize_dtypes(self._df)

    def _normalize_dtypes(self, df: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
        """Applies predefined data types to the DataFrame's columns.

        Args:
            df: The DataFrame or LazyFrame to normalize.

        Returns:
            The DataFrame/LazyFrame with enforced data types.
        """
        if DTYPES is not None and df is not None and self._has_data(df):
            # Use collect_schema() for LazyFrame to avoid warning
            if isinstance(df, pl.LazyFrame):
                schema = df.collect_schema()
            else:
                schema = df.schema

            valid_dtypes = {k: v for k, v in DTYPES.items() if k in schema}
            if valid_dtypes:
                df = df.cast(valid_dtypes)
        return df

    def _lazy_load_data(self) -> None:
        """Ensures data is fresh with minimal reads."""
        if self.file_exists:
            if not self.data_in_memory:
                # No DataFrame in memory; however, file exists - load it
                self.load()
            elif not self.file_fresh:
                # File exists but is stale - reload it
                self.load()
            else:
                # DataFrame is present and fresh - do nothing
                pass

    def _lazy_load_profile(self) -> None:
        """Refreshes the dataset profile if it is missing or stale."""
        current_data = self._df
        self._lazy_load_data()
        # Use a robust Polars DataFrame equality helper (handles None, schema, shape, and row content)
        if not self._dataframes_equal(current_data, self._df):
            # Data has changed - refresh it
            self._profile = DatasetProfile.create(df=self._df)
            logger.debug("Dataset profile refreshed due to data change.")
            return
        elif not self._profile:
            self._profile = DatasetProfile.create(df=self._df)
            logger.debug("Dataset profile created.")

    @staticmethod
    def _dataframes_equal(
        a: Optional[pl.DataFrame | pl.LazyFrame], b: Optional[pl.DataFrame | pl.LazyFrame]
    ) -> bool:
        """Compare two Polars DataFrames or LazyFrames for equality.

        Comparison strategy:
            1. If both are None -> equal.
            2. If only one is None -> not equal.
            3. If types differ (one LazyFrame, one DataFrame) -> not equal.
            4. Compare schemas (column names and dtypes).
            5. For DataFrames: Compare height/width and row content.
            6. For LazyFrames: Compare schemas only (query plans may differ but produce same result).

        Args:
            a: First DataFrame or LazyFrame.
            b: Second DataFrame or LazyFrame.

        Returns:
            bool: True if DataFrames/LazyFrames are equal, False otherwise.
        """
        if a is None or b is None:
            return a is b

        # Check if types match
        if type(a) != type(b):
            return False

        # Compare schemas (works for both without triggering execution)
        if a.schema != b.schema:
            return False

        # For DataFrames, do full comparison including data
        if isinstance(a, pl.DataFrame) and isinstance(b, pl.DataFrame):
            # Quick shape check
            if a.height != b.height or a.width != b.width:
                return False
            # Row-wise content check
            return a.rows() == b.rows()

        # For LazyFrames, schema comparison is sufficient
        # We avoid collecting/executing the query plans
        return True
