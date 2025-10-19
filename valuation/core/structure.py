#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/core/structure.py                                                        #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 07:11:18 pm                                               #
# Modified   : Sunday October 19th 2025 08:21:17 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Provides data utilities."""
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

from abc import ABC
from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum, StrEnum

import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------------------ #
# mypy: allow-any-generics
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
IMMUTABLE_TYPES: Tuple = (
    str,
    int,
    float,
    bool,
    Enum,
    StrEnum,
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
)
SEQUENCE_TYPES: Tuple = (
    list,
    tuple,
)
# ------------------------------------------------------------------------------------------------ #
NUMERICS = [
    "int16",
    "int32",
    "int64",
    "float16",
    "float32",
    "float64",
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
]


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataClass(ABC):  # noqa
    """Base Class for Data Transfer Objects"""

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if type(v) in IMMUTABLE_TYPES
            ),
        )

    def __str__(self) -> str:
        """Pretty prints the dataclass and all nested dataclasses recursively."""
        blocks = self._collect_blocks(self)
        return "\n".join(self._format_block(name, data) for name, data in blocks)

    @staticmethod
    def _collect_blocks(obj: "DataClass", prefix: str = "") -> List[Tuple[str, Dict[str, Any]]]:
        """
        Recursively collects all dataclass blocks for printing.
        Returns a list of tuples: (display_name, data_dict)
        """
        blocks = []
        current_name = prefix if prefix else obj.__class__.__name__
        current_block = {}

        # Iterate over fields of the passed object, not self
        for field in fields(obj):
            key = field.name
            value = getattr(obj, key)

            # If the value is a dataclass instance, recursively collect its blocks
            if is_dataclass(value) and not isinstance(value, type):
                nested_prefix = f"{current_name}.{key}"
                blocks.extend(DataClass._collect_blocks(value, prefix=nested_prefix))
            # If it's a list of dataclasses, handle each one
            elif isinstance(value, list) and value and is_dataclass(value[0]):
                for idx, item in enumerate(value):
                    if is_dataclass(item) and not isinstance(item, type):
                        nested_prefix = f"{current_name}.{key}[{idx}]"
                        blocks.extend(DataClass._collect_blocks(item, prefix=nested_prefix))
            # If it's a dict of dataclasses, handle each one
            elif isinstance(value, dict) and value:
                first_val = next(iter(value.values()), None)
                if is_dataclass(first_val) and not isinstance(first_val, type):
                    for dict_key, item in value.items():
                        nested_prefix = f"{current_name}.{key}[{dict_key}]"
                        blocks.extend(DataClass._collect_blocks(item, prefix=nested_prefix))
                else:
                    # Regular dict with immutable values
                    if type(value) in IMMUTABLE_TYPES or DataClass._is_simple_dict(value):
                        current_block[key] = value
            else:
                # Only add immutable types to current block
                if type(value) in IMMUTABLE_TYPES:
                    current_block[key] = value

        # Add current block at the beginning
        if current_block:
            blocks.insert(0, (current_name, current_block))

        return blocks

    @staticmethod
    def _is_simple_dict(value: Any) -> bool:
        """Check if a dict contains only immutable values."""
        if not isinstance(value, dict):
            return False
        return all(type(v) in IMMUTABLE_TYPES for v in value.values())

    def _format_block(self, name: str, data: Dict[str, Any]) -> str:
        """Formats a single block for pretty printing."""
        width = 32
        breadth = width * 2
        s = f"\n{name.center(breadth, ' ')}"
        s += "\n" + "=" * breadth
        for k, v in data.items():
            s += f"\n{k.rjust(width, ' ')} | {v}"
        s += "\n"
        return s

    def as_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the DataClass object."""
        return asdict(self)


# ------------------------------------------------------------------------------------------------ #
#                                        DATASAET SPLITTER                                         #
# ------------------------------------------------------------------------------------------------ #
class DataFrameSplitter:
    """A utility class for splitting and sampling data."""

    def split_by_size(
        self,
        df: pd.DataFrame,
        train_size: Union[int, float],
        val_size: Union[int, float] = 0.0,
        shuffle: bool = False,
        random_state: int = None,
    ) -> Dict[str, pd.DataFrame]:

        if shuffle:
            df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # Extract the umber of observations in the dataset
        n = len(df)

        # Convert dataset sizes to integers if they are given as fractions
        train_size = train_size if isinstance(train_size, int) else int(n * train_size)
        val_size = val_size if isinstance(val_size, int) else int(n * val_size)

        # Validate sizes
        if train_size + val_size > n:
            raise ValueError(
                "The sum of train_size, val_size, and test_size exceeds the dataset size."
            )

        # Split the dataset
        train_end = train_size
        val_end = train_end + val_size

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]  # Test set is the remainder

        # Return the splits as a dictionary
        splits = {
            "meta": {
                "n_train": len(train_df),
                "n_validation": len(val_df),
                "n_test": len(test_df),
                "n_total": n,
            },
            "data": {"train": train_df, "validation": val_df, "test": test_df},
        }

        return splits

    def split_by_proportion_of_values(
        self, df: pd.DataFrame, val_col: str, train_size: float, val_size: float = 0.0
    ) -> Dict[str, pd.DataFrame]:
        """Splits the DataFrame into training, validation, and test sets based on proportions of unique values in a specified column.
        Args:
            df (pd.DataFrame): The DataFrame to split.
            val_col (str): The column name to base the split on.
            train_size (float): The proportion of unique values to include in the training set
                (between 0 and 1).
            val_size (float, optional): The proportion of unique values to include in the validation
                set (between 0 and 1). Defaults to 0.0.
        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing the training, validation, and test
                DataFrames.
        """
        # Validate arguments
        if not (0 < train_size < 1):
            raise ValueError("train_size must be a float between 0 and 1.")
        if not (0 <= val_size < 1):
            raise ValueError("val_size must be a float between 0 and 1.")
        if train_size + val_size >= 1:
            raise ValueError("The sum of train_size and val_size must be less than 1.")
        if val_col not in df.columns:
            raise ValueError(f"{val_col} is not a column in the DataFrame.")

        # Get unique values in the validation column
        unique_values = sorted(df[val_col].unique())
        n_values = len(unique_values)

        # Extract the end values for each split
        train_end = int(n_values * train_size)
        val_end = train_end + int(n_values * val_size)

        # Specify the unique values in each split
        train_values = unique_values[:train_end]
        val_values = unique_values[train_end:val_end]
        test_values = unique_values[val_end:]

        # Create the splits
        train_df = df[df[val_col].isin(train_values)]
        val_df = df[df[val_col].isin(val_values)]
        test_df = df[df[val_col].isin(test_values)]

        # Return the splits as a dictionary
        splits = {
            "parameters": {
                "val_col": val_col,
                "train_size": train_size,
                "val_size": val_size,
            },
            "meta": {
                "n_train": len(train_df),
                "n_validation": len(val_df),
                "n_test": len(test_df),
                "n_total": len(df),
            },
            "data": {"train": train_df, "validation": val_df, "test": test_df},
        }
        return splits
