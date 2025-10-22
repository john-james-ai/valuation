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
# Modified   : Wednesday October 22nd 2025 11:46:35 am                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Optional, Union

from dataclasses import dataclass

from valuation.asset.dataset.base import Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.flow.base.pipeline import Pipeline, PipelineBuilder, PipelineConfig, PipelineResult
from valuation.flow.validation import Validation
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepPipelineConfig(PipelineConfig):
    """Holds all parameters for the pipeline."""

    source: str
    target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DataPrepPipelineResult(PipelineResult):
    """Holds the results of a pipeline execution."""

    num_errors: int = 0
    num_warnings: int = 0

    dataset: Optional[Dataset] = None


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipeline(Pipeline):

    _config: DataPrepPipelineConfig
    _dataset_store: DatasetStore

    def __init__(
        self,
        dataset_store: type[DatasetStore] = DatasetStore,
        result: type[DataPrepPipelineResult] = DataPrepPipelineResult,
    ) -> None:
        super().__init__()
        self._dataset_store = dataset_store()
        self._result = result(name=self.__class__.__name__)

    def add_source(self, source: Union[str, DatasetPassport]) -> DataPrepPipeline:
        self._source = source
        return self

    def add_target(self, target: DatasetPassport) -> DataPrepPipeline:
        self._target = target
        return self

    def _update_metrics(self, validation: Validation) -> None:
        self._result.num_errors += validation.num_errors
        self._result.num_warnings += validation.num_warnings


# ------------------------------------------------------------------------------------------------ #
class DataPrepPipelineBuilder(PipelineBuilder):
    """Builds Data Preparation Pipelines."""

    def build(self) -> None:
        raise NotImplementedError("DataPrepPipelineBuilder.build() is not implemented.")
