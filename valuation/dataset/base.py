#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/base.py                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Saturday October 11th 2025 01:31:12 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from loguru import logger
import pandas as pd

from valuation.config.data_prep import (
    DataPrepBaseConfig,
    DataPrepSingleOutputConfig,
    DataPrepSISOConfig,
)
from valuation.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class DataPrep(ABC):
    """Abstract base class for data preparation."""

    def __init__(self, io: IOService = IOService) -> None:
        self._io = io

    def load(self, filepath: Path) -> pd.DataFrame:
        """Loads a single data file from the raw data directory.

        Args:
            filepath(Path): The path to the file to be loaded.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """

        df = self._io.read(filepath=filepath)
        return df

    def save(self, df: pd.DataFrame, filepath: Path) -> None:
        """Saves a DataFrame to the processed data directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            filepath (Path): The path to the file to be saved.
        """
        self._io.write(data=df, filepath=filepath)

    def delete(self, filepath: Path) -> None:
        """Deletes a file from the processed data directory.

        Args:
            filepath (Path): The path of the file to delete
        """
        filepath.unlink(missing_ok=True)

    def exists(self, filepath: Path) -> bool:
        """Checks if a file exists in the processed data directory.

        Args:
            filepath (Path): The path to a file for the existence check

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return filepath.exists()

    @abstractmethod
    def prepare(self, config: DataPrepBaseConfig) -> None:
        """Abstract method to be implemented by subclasses for data preparation."""
        pass

    @abstractmethod
    def _use_cache(self, config: DataPrepBaseConfig) -> bool:
        """Abstract method that controls the use of cache in alignment on a force flag."""


# ------------------------------------------------------------------------------------------------ #
class DataPrepSingleOutput(DataPrep):

    @abstractmethod
    def prepare(self, config: Union[DataPrepSISOConfig, DataPrepSingleOutputConfig]) -> None:
        """Abstract method to be implemented by subclasses for data preparation."""
        pass

    def _use_cache(self, config: Union[DataPrepSISOConfig, DataPrepSingleOutputConfig]) -> bool:
        """Determines whether to use cached data based on file existence and force flag.

        Args:
            config (DataPrepSISOConfig): Configuration object containing core settings.
        Returns:
            bool: True if cached data should be used, False otherwise.

        """

        if config.core_config.force:
            self.delete(filepath=config.output_filepath)
            use_cache = False
        else:
            use_cache = (
                self.exists(filepath=config.output_filepath) and not config.core_config.force
            )

        if use_cache:
            logger.info(
                f"{config.core_config.task_name} - Output file already exists. Using cached data."
            )
        else:
            logger.info(f"{config.core_config.task_name}  - Starting")
        return use_cache
