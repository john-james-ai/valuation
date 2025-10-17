#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/io/csv/base.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 11:00:51 am                                              #
# Modified   : Thursday October 16th 2025 11:03:23 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Defines base classes and enumerations for CSV IO operations."""
from enum import Enum


# ------------------------------------------------------------------------------------------------ #
#                                       COMPRESSION TYPE                                           #
# ------------------------------------------------------------------------------------------------ #
class CompressionType(Enum):
    NONE = None
    INFER = "infer"  # Not supported by Dask
    GZIP = "gzip"
    BZ2 = "bz2"
    ZIP = "zip"  # Not supported by Dask
    XZ = "xz"
    TAR = "tar"  # Not supported by Dask
    ZSTD = "zstd"  # Not supported by Dask
