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
# Modified   : Sunday October 19th 2025 03:36:53 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
class EntityNotFoundError(Exception):
    """Exception raised when there is a file conflict, such as attempting to overwrite an existing file without permission."""

    pass


class DatasetNotFoundError(Exception):
    """Exception raised when there is a file conflict, such as attempting to overwrite an existing file without permission."""

    pass


class DatasetExistsError(Exception):
    """Exception raised when there is a file conflict, such as attempting to overwrite an existing file without permission."""

    pass


class AssetExistsError(Exception):
    """Exception raised when there is a file conflict, such as attempting to overwrite an existing file without permission."""

    pass


class AssetStoreNotFoundError(Exception):
    """Exception raised when there is a file conflict, such as attempting to overwrite an existing file without permission."""

    pass
