#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /tests/test_infra/test_store/test_dataset_store.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 19th 2025 02:58:43 pm                                                #
# Modified   : Saturday October 25th 2025 11:04:45 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import cast

from datetime import datetime
import inspect

from loguru import logger
import pytest

from valuation.asset.dataset import Dataset
from valuation.asset.identity.dataset import DatasetID
from valuation.infra.file.dataset import DatasetFileSystem
from valuation.infra.store.dataset import DatasetStore

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.dataset_store
class TestDatasetStore:  # pragma: no cover
    # ============================================================================================ #
    def test_setup(self, dataset_passport, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        store = DatasetStore()
        store.remove(dataset_id=dataset_passport.id)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)  # type: ignore

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_add(self, dataset, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        store = DatasetStore()
        store.add(dataset=dataset)
        fs = DatasetFileSystem()
        passport_filepath = fs.get_passport_filepath(id_or_passport=dataset.passport)
        asset_filepath = fs.get_asset_filepath(id_or_passport=dataset.passport)
        assert passport_filepath.exists()
        assert asset_filepath.exists()

        with pytest.raises(FileExistsError):
            store.add(dataset=dataset, overwrite=False)

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_get(self, dataset_passport, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        store = DatasetStore()
        dataset_id = DatasetID.from_passport(passport=dataset_passport)
        dataset = store.get(dataset_id=dataset_id)
        dataset = cast(Dataset, dataset)
        assert dataset is not None
        assert dataset.passport.name == dataset_passport.name
        assert dataset.passport.stage == dataset_passport.stage
        # assert dataset.passport.entity == dataset_passport.entity
        assert dataset.passport.asset_type == dataset_passport.asset_type
        assert dataset.passport.file_format == dataset_passport.file_format
        assert dataset.data is not None
        assert not dataset.data.empty

        logger.info(f"Retrieved dataset:\n{dataset.data.head()}")

        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)

    # ============================================================================================ #
    def test_remove(self, dataset_passport, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        dataset_id = DatasetID.from_passport(passport=dataset_passport)
        store = DatasetStore()
        store.remove(dataset_id=dataset_id)
        fs = DatasetFileSystem()
        passport_filepath = fs.get_passport_filepath(id_or_passport=dataset_passport.id)
        asset_filepath = fs.get_asset_filepath(id_or_passport=dataset_passport)
        assert not passport_filepath.exists()
        assert not asset_filepath.exists()

        # Attempt to remove non-existing dataset
        store.remove(dataset_id=dataset_id)  # Check logs for message
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
