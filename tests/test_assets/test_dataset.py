#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_assets/test_dataset.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 09:13:25 pm                                                #
# Modified   : Saturday October 25th 2025 03:07:13 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from copy import copy
from datetime import datetime
import inspect
from pathlib import Path
import shutil

from loguru import logger
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from valuation.asset.dataset.dataset import Dataset, DatasetProfile
from valuation.core.types import AssetType
from valuation.infra.exception import DatasetExistsError, DatasetNotFoundError
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.file.io import IOService

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
DATASET_FILEPATH = Path("tests/data/test_assets/test_datasets/test_dataset.csv")

double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.dataset
class TestSalesDataset:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        shutil.rmtree("data/test/test", ignore_errors=True)
        shutil.rmtree("asset_store/dataset/test", ignore_errors=True)
        logger.info("Test environment setup complete.")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_validation(self, dataset_passport, sales_df, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Validate passport type
        with pytest.raises(TypeError):
            Dataset(passport="invalid_passport")  # type: ignore

        # Validate asset type
        passport = copy(dataset_passport)
        passport.asset_type = AssetType.REPORT
        with pytest.raises(TypeError):
            Dataset(passport=passport)

        # Correct bad name
        passport = copy(dataset_passport)
        with pytest.raises(DatasetExistsError):
            Dataset(passport=passport, df=sales_df)

        # No data provided and file does not exist
        passport = copy(dataset_passport)
        with pytest.raises((DatasetNotFoundError, FileNotFoundError)):
            Dataset(passport=passport)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_passport_existing_file(self, dataset, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Save the original dataset to file
        dataset.save()
        # Create a new dataset with same passport
        new_passport = dataset.passport
        logger.info(f"Original passport: \n{dataset.passport}")
        logger.info(f"New passport: \n{new_passport}")
        fs = DatasetFileSystem()
        filepath = fs.get_asset_filepath(passport=new_passport)
        logger.info(f"Dataset file path: {filepath}")
        ds = Dataset(passport=new_passport)
        assert isinstance(ds, Dataset)
        assert ds.passport == new_passport
        assert isinstance(ds.data, pd.DataFrame)  # Should load from file.

        # Fileinfo
        assert dataset.fileinfo is not None
        assert dataset.fileinfo.filepath.exists()
        assert dataset.fileinfo.filename == f"{dataset.passport.name}.parquet"
        assert dataset.fileinfo.file_size_mb > 0.0
        assert dataset.fileinfo.created_timestamp is not None
        assert dataset.fileinfo.modified_timestamp is not None
        assert dataset.fileinfo.is_stale is False

        ds.load()
        assert_frame_equal(dataset.data, ds.data)
        assert ds.profile is not None
        assert ds.profile.nrows == dataset.profile.nrows
        assert ds.profile.ncols == dataset.profile.ncols
        assert ds.profile.n_duplicates == dataset.profile.n_duplicates
        assert ds.profile.missing_values == dataset.profile.missing_values
        assert ds.profile.memory_usage_mb == dataset.profile.memory_usage_mb
        assert ds.profile.info.equals(dataset.profile.info)
        # Logging
        for record in caplog.records:
            logger.info(f"LOG: {record.msg}")
        logger.info(f"Dataset Profile:\n{ds.profile}")
        logger.info(f"Dataset Info:\n{ds.profile.info}")
        logger.info(f"Dataset Fileinfo:\n{ds.fileinfo}")

        dataset.delete()
        assert not dataset.fileinfo.filepath.exists()
        assert not dataset.exists()

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_df_passport_no_file(self, dataset_passport, sales_df, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        passport = copy(dataset_passport)

        dataset = Dataset(passport=dataset_passport, df=sales_df)
        assert dataset.passport == dataset_passport
        assert_frame_equal(dataset.data, sales_df)

        # Profile
        assert dataset.profile is not None
        assert isinstance(dataset.profile, DatasetProfile)
        assert dataset.profile.nrows == sales_df.shape[0]
        assert dataset.profile.ncols == sales_df.shape[1]
        assert dataset.profile.n_duplicates == 0
        assert dataset.profile.missing_values == sales_df.isna().sum().sum()
        assert dataset.profile.memory_usage_mb > 0.0
        assert isinstance(dataset.profile.info, pd.DataFrame)

        # Fileinfo
        assert dataset.fileinfo is None
        # Properties
        assert dataset.asset_type == AssetType.DATASET
        assert dataset.data_in_memory is True
        assert dataset.file_exists is False
        assert dataset.file_fresh is False
        assert dataset.nrows == sales_df.shape[0]
        assert dataset.ncols == sales_df.shape[1]

        # Save dataset to file
        dataset.save()
        assert dataset.fileinfo is not None
        assert dataset.fileinfo.filepath.exists()
        assert dataset.exists()
        assert dataset.data_in_memory is True
        assert dataset.file_exists is True
        assert dataset.file_fresh is True

        # Save in an alternative location and format
        dataset.save_as(filepath=DATASET_FILEPATH)
        assert DATASET_FILEPATH.exists()

        assert dataset.data is not None
        assert isinstance(dataset.data, pd.DataFrame)
        assert not dataset.exists()

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_stale_dataset(self, dataset, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        # Save the dataset to file
        dataset.save(overwrite=True)
        assert dataset.fileinfo is not None
        assert dataset.fileinfo.is_stale is False

        # Print fileinfo
        logger.info(f"Dataset Fileinfo before modification:\n{dataset.fileinfo}")

        # Modify the dataset file to simulate staleness
        df = IOService.read(filepath=dataset._asset_filepath)
        df = df.sample(frac=0.5).reset_index(drop=True)  # Modify the data
        IOService.write(data=df, filepath=dataset._asset_filepath)

        logger.info(f"Dataset FileInfo after modification:\n{dataset.fileinfo}")
        # Determine staleness
        assert dataset.fileinfo.is_stale is True
        logger.info(f"Dataset file is stale: {dataset.fileinfo.is_stale}")
        # Refresh fileinfo
        dataset.load()
        logger.info(f"Dataset Fileinfo after refresh:\n{dataset.fileinfo}")
        assert dataset.fileinfo.is_stale is False

        dataset.delete()
        assert not dataset.fileinfo.filepath.exists()
        assert not dataset.exists()
        assert dataset.fileinfo is None

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
