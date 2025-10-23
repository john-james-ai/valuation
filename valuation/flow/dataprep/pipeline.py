#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/pipeline.py                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Thursday October 23rd 2025 10:03:46 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Any, Dict, Union

from pathlib import Path

from loguru import logger
import pandas as pd

from valuation.asset.dataset.base import DTYPES
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

    def _load(self, filepath: Path, **kwargs) -> pd.DataFrame | Dict[str, Any]:

        try:

            data = IOService.read(filepath=filepath, **kwargs)
            # Ensure correct data types
            if isinstance(data, pd.DataFrame):
                data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
            return data
        except Exception as e:
            logger.critical(f"Failed to load data from {filepath.name} with exception: {e}")
            raise e


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipelineBuilder(PipelineBuilder):
    """Builds Data Preparation Pipelines."""

    def build(self) -> None:
        raise NotImplementedError("DataPrepPipelineBuilder.build() is not implemented.")
