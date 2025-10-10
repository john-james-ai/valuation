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
# Modified   : Friday October 10th 2025 03:03:47 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod

import pandas as pd

from valuation.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from valuation.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
class DataPrep(ABC):
    """Abstract base class for data preparation."""

    def __init__(self, io: IOService = IOService) -> None:
        self._io = io

    def load(self, filename: str) -> pd.DataFrame:
        """Loads a single data file from the raw data directory.

        Args:
            filename (str): The name of the file to load from the raw data directory.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded data.
        """
        filepath = RAW_DATA_DIR / filename
        df = self._io.read(filepath=filepath)
        return df

    def save(self, df: pd.DataFrame, filename: str) -> None:
        """Saves a DataFrame to the processed data directory.

        Args:
            df (pd.DataFrame): The DataFrame to save.
            filename (str): The name of the file to save the DataFrame to within
                the processed data directory.
        """
        filepath = PROCESSED_DATA_DIR / filename
        self._io.write(data=df, filepath=filepath)

    def delete(self, filename: str) -> None:
        """Deletes a file from the processed data directory.

        Args:
            filename (str): The name of the file to delete from the processed data directory.
        """
        filepath = PROCESSED_DATA_DIR / filename
        filepath.unlink(missing_ok=True)

    def exists(self, filename: str) -> bool:
        """Checks if a file exists in the processed data directory.

        Args:
            filename (str): The name of the file to check for existence in the processed data
                directory.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        filepath = PROCESSED_DATA_DIR / filename
        return filepath.exists()

    @abstractmethod
    def prepare(self, *args, **kwargs) -> pd.DataFrame:
        """Process the data and return a DataFrame."""
        pass
