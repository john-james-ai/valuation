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
# Modified   : Friday October 10th 2025 05:13:48 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from typing import Dict

from loguru import logger
import pandas as pd
from tqdm import tqdm

from valuation.config import DATASET_PROFILE_FILEPATH, RAW_DATA_DIR
from valuation.dataset.base import DataPrep

# ------------------------------------------------------------------------------------------------ #


class SalesDataProfile(DataPrep):
    """Creates a profile of the Dominick's Fine Foods Dataset."""

    def prepare(self, category_filenames: dict, force: bool = False) -> None:
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
        category_profiles = []

        # Check if output file already exists and not forcing reprocessing
        if not force and self.exists(filepath=DATASET_PROFILE_FILEPATH):
            logger.info(
                f"Profile for the dataset already exists at {DATASET_PROFILE_FILEPATH}.\n \
                    Skipping the profile. To re-profile, set force=True."
            )
            return

        # If force is True and file exists, log that we are reprocessing and
        # remove the existing file
        if force and self.exists(filepath=DATASET_PROFILE_FILEPATH):
            logger.info(
                f"Force reprocessing the data profile. \
                    Existing files will be overwritten."
            )
            self.delete(filepath=DATASET_PROFILE_FILEPATH)

        logger.info("Profiling dataset...")

        # Set up the progress bar
        pbar = tqdm(category_filenames.items(), total=len(category_filenames))

        # Iterate through category sales files
        for category_id, category_filename in pbar:
            pbar.set_description(
                f"Processing category: {category_filename['category']} from file: {category_filename['filename']}"
            )

            # Load, clean, calculate revenue, and aggregate
            profile = self._profile_category(category_filename)
            category_profiles.append(profile)

        # Concatenate all datasets
        profile = pd.DataFrame(category_profiles)

        # Save processed dataset
        self.save(df=profile, filepath=DATASET_PROFILE_FILEPATH)

    def _profile_category(self, category_filename: Dict[str, str]) -> Dict:
        """Profiles a single category sales data file.

        Args:
            category_id (str): The category id of the file to profile.
            category_filename (Dict[str,str]): A dictionary containing the filename and category
                name.

        Returns:
            pd.DataFrame: A DataFrame containing the profile of the specified category.
        """
        filepath = RAW_DATA_DIR / category_filename["filename"]
        df = self.load(filepath=filepath)

        profile = {
            "filename": category_filename["filename"],
            "category": category_filename["category"],
            "stores": df["store"].nunique() if "store" in df.columns else 0,
            "weeks": df["week"].nunique() if "week" in df.columns else 0,
            "num_records": len(df),
            "num_columns": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "num_duplicates": df.duplicated().sum(),
            "invalid_records": len(df[df["ok"] == 0]) if "ok" in df.columns else 0,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
            "file_size_mb": filepath.stat().st_size / (1024 * 1024) if filepath.exists() else 0,
        }

        return profile
