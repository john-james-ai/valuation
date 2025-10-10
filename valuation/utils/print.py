#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/print.py                                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 09:20:48 pm                                               #
# Modified   : Thursday October 9th 2025 09:21:43 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Print Module"""
from datetime import date, datetime

import numpy as np

# ------------------------------------------------------------------------------------------------ #
IMMUTABLE_TYPES: tuple = (
    str,
    int,
    float,
    bool,
    np.int16,
    np.int32,
    np.int64,
    np.int8,
    np.uint8,
    np.uint16,
    np.float16,
    np.float32,
    np.float64,
    np.float128,
    np.bool_,
    datetime,
    date,
)


# ------------------------------------------------------------------------------------------------ #
class Printer:
    """Class for printing in various formats.

    Attributes
    ----------
    _width : int
        The width of the printed output.
    """

    def __init__(self, width: int = 80) -> None:
        """
        Initializes the Printer with a specified width.

        Args:
            width (int): The width of the printed output. Default is 80.
        """
        self._width = width

    def print_header(self, title: str) -> None:
        """
        Prints a formatted header.

        Args:
            title (str): The title to be printed in the header.
        """
        breadth = self._width - 4
        header = f"\n\n# {breadth * '='} #\n"
        header += f"#{title.center(self._width - 2, ' ')}#\n"
        header += f"# {breadth * '='} #\n"
        print(header)

    def print_trailer(self) -> None:
        """
        Prints a formatted trailer.
        """
        breadth = self._width - 4
        trailer = f"\n\n# {breadth * '='} #\n"
        print(trailer)

    def print_dict(self, title: str, data: dict) -> None:
        """
        Prints a dictionary in a formatted manner.

        Args:
            title (str): The title to be printed above the dictionary.
            data (dict): The dictionary to be printed.
        """
        breadth = int(self._width / 2)
        s = f"\n\n{title.center(self._width, ' ')}"
        for k, v in data.items():
            if isinstance(v, IMMUTABLE_TYPES):
                s += f"\n{k.rjust(breadth, ' ')} | {v}"
        s += "\n\n"
        print(s)
