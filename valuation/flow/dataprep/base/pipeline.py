#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/base/pipeline.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Saturday October 25th 2025 11:04:40 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Any, Dict, Union

from pathlib import Path

from loguru import logger
import polars as pl

from valuation.asset.dataset import DTYPES
from valuation.asset.identity.dataset import DatasetPassport
from valuation.flow.base.pipeline import Pipeline, PipelineBuilder
from valuation.infra.file.io import IOService
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipeline(Pipeline):

    _dataset_store: DatasetStore

    def __init__(
        self,
        dataset_store: type[DatasetStore] = DatasetStore,
    ) -> None:
        super().__init__()
        self._dataset_store = dataset_store()

    def add_source(self, source: Union[str, DatasetPassport]) -> DataPrepPipeline:
        self._source = source
        return self

    def add_target(self, target: DatasetPassport) -> DataPrepPipeline:
        self._target = target
        return self

    def _load(
        self,
        filepath: Path,
        lazy: bool = True,
        cast_dtypes: bool = True,
        **kwargs,
    ) -> Union[pl.LazyFrame, pl.DataFrame, Dict[str, Any]]:
        """
        Load data from file using IOService with lazy loading support.

        Args:
            filepath: Path to the data file
            lazy: If True, return LazyFrame for DataFrames; if False, return DataFrame
            cast_dtypes: If True, cast columns to dtypes defined in DTYPES
            **kwargs: Additional arguments passed to IOService.read

        Returns:
            Union[pl.LazyFrame, pl.DataFrame, Dict]: Loaded data with correct dtypes
        """
        try:
            logger.debug(
                f"Loading data from {filepath} with lazy={lazy} and cast_dtypes={cast_dtypes} from {__name__}"
            )
            # Load data with lazy parameter
            data = IOService.read(filepath=filepath, lazy=lazy, **kwargs)

            # Cast dtypes for DataFrames only
            if cast_dtypes and isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                # Get columns and build cast dict
                columns = (
                    data.collect_schema().names()
                    if isinstance(data, pl.LazyFrame)
                    else data.columns
                )
                cast_dict = {col: DTYPES[col] for col in columns if col in DTYPES}

                # Apply casting if needed
                if cast_dict:
                    data = data.cast(cast_dict)

            return data

        except Exception as e:
            logger.critical(f"Failed to load data from {filepath.name} with exception: {e}")
            raise e


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipelineBuilder(PipelineBuilder):
    """Builds Data Preparation Pipelines."""

    def build(self) -> None:
        raise NotImplementedError("DataPrepPipelineBuilder.build() is not implemented.")
