#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/infra/exception.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 17th 2025 05:55:14 pm                                                #
# Modified   : Saturday October 25th 2025 10:02:07 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

"""Project-specific exceptions used across the infra layer.

This module defines exception types raised by storage and dataset operations.
"""


class ArtifactNotFoundError(Exception):
    """Raised when a requested artifact is not found."""

    pass


class DatasetNotFoundError(Exception):
    """Raised when a requested dataset file or passport cannot be found."""

    pass


class ArtifactExistsError(Exception):
    """Raised when attempting to create an artifact that already exists."""

    pass


class DatasetExistsError(Exception):
    """Raised when attempting to create a dataset file that already exists."""

    pass


class AssetExistsError(Exception):
    """Raised when attempting to create an asset that already exists in the store."""

    pass


class AssetStoreNotFoundError(Exception):
    """Raised when the configured asset store cannot be located or instantiated."""

    pass
