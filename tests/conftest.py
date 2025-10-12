#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/conftest.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 08:23:13 pm                                              #
# Modified   : Saturday October 11th 2025 09:51:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import pytest

from valuation.config.data_prep import DTYPES
from valuation.config.filepaths import TRAIN_DATA_FILEPATH
from valuation.utils.io import IOService

# ------------------------------------------------------------------------------------------------ #
FINANCIALS_FILEPATH = "data/external/financials.yaml"
SALES_FILEPATH = "data/processed/train.csv"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=False)
def financials():
    return IOService.read(filepath=FINANCIALS_FILEPATH)["financials"]


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=False)
def sales():
    df = IOService.read(filepath=TRAIN_DATA_FILEPATH)
    return df.astype({k: v for k, v in DTYPES.items() if k in df.columns})
