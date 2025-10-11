#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/profile.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 11th 2025 12:47:55 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Dataset Profile Module"""
from pathlib import Path
from typing import Dict

import pandas as pd
from pydantic import Field
from pydantic.dataclasses import dataclass
from tqdm import tqdm

from valuation.config.data_prep import DataPrepSingleOutputConfig
from valuation.dataset.base import DataPrepSingleOutput


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ProfileConfig(DataPrepSingleOutputConfig):
    """Holds all parameters for the profiling process."""

    raw_data_directory: Path
    category_filenames: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    core_features: list[str] = Field(
        default_factory=lambda: ["STORE", "UPC", "WEEK", "MOVE", "QTY", "PRICE", "PROFIT", "OK"]
    )


# ------------------------------------------------------------------------------------------------ #


class SalesDataProfile(DataPrepSingleOutput):
    """Creates a profile of the Dominick's Fine Foods Dataset."""

    def prepare(self, config: ProfileConfig) -> None:
        """Transforms raw sales data files into a cleaned and aggregated dataset.

        This method orchestrates the loading, cleaning, revenue calculation,
        gross profit calculation, aggregation, and saving of sales data.

        Args:
            category_filenames (dict): A dictionary mapping category names to
                their corresponding filenames.
            force (bool): If True, forces reprocessing even if the output file exists. Defaults to
                False.
        Returns:
            None
        """
        # Check if the output file already exists and not forcing reprocessing
        if self._use_cache(config=config):
            return

        # Set up the progress bar
        pbar = tqdm(config.category_filenames.items(), total=len(config.category_filenames))

        category_profiles = []

        # Iterate through category sales files
        for _, category_filename in pbar:
            pbar.set_description(
                f"Processing category: {category_filename['category']} from  {category_filename['filename']}"
            )

            # Create profile for the category
            profile = self._profile_category(config=config, category_filename=category_filename)

            # Append to list of profiles
            category_profiles.append(profile)

        # Concatenate all datasets
        profile = pd.DataFrame(category_profiles)

        # Save processed dataset
        self.save(df=profile, filepath=config.output_filepath)

    def _profile_category(self, config: ProfileConfig, category_filename: Dict[str, str]) -> Dict:
        """Profiles a single category sales data file.

        Args:
            category_id (str): The category id of the file to profile.
            category_filename (Dict[str,str]): A dictionary containing the filename and category
                name.

        Returns:
            pd.DataFrame: A DataFrame containing the profile of the specified category.
        """
        filepath = config.raw_data_directory / category_filename["filename"]
        df = self.load(filepath=filepath)

        profile = {
            "filename": category_filename["filename"],
            "category": category_filename["category"],
            "stores": df["STORE"].nunique() if "STORE" in df.columns else 0,
            "weeks": df["WEEK"].nunique() if "WEEK" in df.columns else 0,
            "num_records": len(df),
            "num_columns": len(df.columns),
            "missing_values": df[config.core_features].isnull().sum().sum(),
            "missing_values_%": df[config.core_features].isnull().sum().sum() / df.shape[0] * 100,
            "invalid_records": len(df[df["OK"] == 0]) if "OK" in df.columns else 0,
            "invalid_records_%": (
                len(df[df["OK"] == 0]) / df.shape[0] * 100 if "OK" in df.columns else 0
            ),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "file_size_mb": (filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0),
        }

        return profile
