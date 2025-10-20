#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/file.py                                                            #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 12:50:02 am                                                #
# Modified   : Sunday October 19th 2025 07:17:37 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path


# ------------------------------------------------------------------------------------------------ #
def get_filenames_in_directory(directory):
    path = Path(directory)
    filenames = [entry.name for entry in path.iterdir() if entry.is_file()]
    return filenames


# ------------------------------------------------------------------------------------------------ #
def is_directory_empty(directory):
    """
    Checks if a given directory is empty using pathlib.

    Args:
        directory (str or Path): The path to the directory.

    Returns:
        bool: True if the directory is empty, False otherwise.
    """
    path = Path(directory)

    # Check if the path exists and is a directory
    if not path.is_dir():
        return False

    # iterdir() returns an iterator over the contents of the directory.
    # any() returns True if any element in the iterable is True, False otherwise.
    # If iterdir() yields no elements, any() will return False, meaning the directory is empty.
    return not any(path.iterdir())
