#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/conftest.py                                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 11th 2025 08:23:13 pm                                              #
# Modified   : Monday October 20th 2025 03:03:28 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import pandas as pd
import pytest

from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.entity import Entity
from valuation.core.stage import DatasetStage
from valuation.infra.file.base import FileFormat
from valuation.infra.file.io import IOService

# ------------------------------------------------------------------------------------------------ #
DATASET_FILEPATH = "tests/data/wbat.csv"


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="function", autouse=True)
def auto_set_test_mode_env(monkeypatch):
    """
    A fixture that automatically patches os.environ for every test.

    This sets MODE="test" in the environment variables. Pytest's
    monkeypatch automatically restores the original state after the test.
    """
    # Use monkeypatch to set the environment variable
    # monkeypatch will automatically handle restoring the original
    # value (or unsetting it) after the test.
    monkeypatch.setenv("MODE", "test")

    # Your app's code that calls load_dotenv() will now
    # load this value from os.environ.

    # If your app *re-reads* the .env file during the test,
    # you might need the file-based approach. But if it reads
    # on startup (like 99% of apps), this is the better way.

    yield

    # No teardown code needed! monkeypatch handles it.


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=False)
def dataset_passport() -> DatasetPassport:
    passport = DatasetPassport.create(
        name="test_dataset",
        description="Test dataset for unit tests.",
        entity=Entity.SALES,
        stage=DatasetStage.TEST,
        file_format=FileFormat.PARQUET,
    )
    return passport


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=False)
def dataset(dataset_passport: DatasetPassport) -> Dataset:
    """Fixture for dataset tests."""

    df = IOService.read(filepath=DATASET_FILEPATH)

    dataset = Dataset(passport=dataset_passport, df=df)

    if dataset is None:
        pytest.skip("Dataset not found in store.")

    return dataset


# ------------------------------------------------------------------------------------------------ #
@pytest.fixture(scope="session", autouse=False)
def sales_df() -> pd.DataFrame:
    """Fixture for sales dataframe."""

    data = IOService.read(filepath=DATASET_FILEPATH)
    data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
    return data
