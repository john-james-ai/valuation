#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/transform.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 02:06:32 am                                                #
# Modified   : Friday October 24th 2025 02:13:48 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import numpy as np


# ------------------------------------------------------------------------------------------------ #
# --- 1. Define Transformation Functions ---
def encode_log(y):
    """Log(1 + y) transformation (used before training)."""
    return np.log1p(y)


def decode_exp(y):
    """Inverse transformation: exp(y) - 1 (used after prediction)."""
    return np.expm1(y)
