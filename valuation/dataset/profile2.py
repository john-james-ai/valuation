#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset.py                                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 03:39:32 am                                                #
# Modified   : Friday October 10th 2025 06:31:17 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import os

import pandas as pd
from tqdm import tqdm

from valuation.config import RAW_DATA_DIR
from valuation.utils.io import IOService


class Dominicks:
    def __init__(self, category_filenames: dict, io: IOService = IOService) -> None:
        """Initializes the Dominicks dataset handler.
        Args:
            category_filenames (dict): A dictionary mapping category ids to their filenames and
                categories.
            io (IOService): An instance of the IOService class for file operations. Defaults to
                IOService.

        Returns: None.
        """
        self._category_filenames = category_filenames
        self._io = io
        self._category_data = {}
        self._category_info = None
        self._dataset_info = {}
        self._dataset = {}

    def load_file(self, category_id: str) -> pd.DataFrame:
        """Loads the DataFrame for a specific category.

        Args:
            category_id (str): The category id of the file to retrieve.
        Returns:
            pd.DataFrame: The DataFrame corresponding to the specified category.
        """
        fileinfo = self._category_filenames.get(category_id)

        if not fileinfo:
            raise ValueError(f"Category '{category_id}' not found in category filenames.")

        filepath = RAW_DATA_DIR / fileinfo["filename"]
        return self._io.read(filepath=filepath)

    def load_dataset(self) -> None:
        """Loads all category data into the dataset dictionary."""
        # Set up the progress bar
        pbar = tqdm(
            self._category_filenames.keys(),
            desc="Loading category data",
            total=len(self._category_filenames),
        )

        # Iterate over each category and load its data
        for category_id in pbar:
            category = self._category_filenames[category_id]["category"]
            pbar.set_description(f"Loading category '{category}'")
            df = self.load_file(category_id=category_id)
            self._category_data[category_id] = df

    def build_dataset(self) -> None:
        """Generates a dataframe containing information about each category file."""
        category_table = []
        self._dataset_info = {
            "n_categories": len(self._category_filenames),
            "filesize_mb": 0,
            "memory_mb": 0,
            "n_rows": 0,
        }
        if not self._category_data:
            self.load_dataset()

        pbar = tqdm(
            self._category_data.items(),
            desc="Generating category info",
            total=len(self._category_data),
        )
        for category_id, df in pbar:

            category_filename = self._category_filenames.get(category_id)

            if not category_filename:
                raise ValueError(
                    f"Category ID '{category_id}' not found \
                    in category filenames."
                )

            category = category_filename["category"]
            pbar.set_description(f"Processing category '{category}'")
            filename = category_filename["filename"]
            filepath = RAW_DATA_DIR / filename
            df = self._category_data[category_id]
            filesize = os.path.getsize(filepath) / (1024 * 1024)
            memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
            n_rows = df.shape[0]

            category_info = {
                "filename": filename,
                "category_id": category_id,
                "category": category,
                "stores": df["STORE"].nunique(),
                "weeks": df["WEEK"].nunique(),
                "filesize_mb": filesize,
                "memory_mb": memory,
                "n_rows": n_rows,
                "n_columns": df.shape[1],
            }

            self._dataset_info["filesize_mb"] += filesize
            self._dataset_info["memory_mb"] += memory
            self._dataset_info["n_rows"] += n_rows

            category_table.append(category_info)
        self._category_info = pd.DataFrame(category_table)
        self._dataset = {
            "data": self._category_data,
            "info": self._dataset_info,
            "categories": self._category_info,
        }
