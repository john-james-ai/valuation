#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/data.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 07:11:18 pm                                               #
# Modified   : Friday October 10th 2025 11:09:41 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Dict, Union

import pandas as pd


# ------------------------------------------------------------------------------------------------ #
class DataFramePartitioner:
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

    def sample(
        self,
        df: pd.DataFrame,
        n: int = None,
        frac: float = None,
        random_state: int = None,
    ) -> pd.DataFrame:
        """Takes a random sample of the data.

        Args:
            df (pd.DataFrame): The DataFrame to sample from.
            n (int, optional): The number of items to sample. Cannot be used with frac.
                Defaults to None.
            frac (float, optional): The fraction of items to sample. Cannot be used with n.
                Defaults to None.
            random_state (int, optional): Seed for the random number generator for
                reproducibility. Defaults to None.

        Returns:
            pd.DataFrame: A randomly sampled DataFrame.
        """
        if (n is None and frac is None) or (n is not None and frac is not None):
            raise ValueError("Must provide either 'n' or 'frac', but not both.")

        return df.sample(n=n, frac=frac, random_state=random_state)
