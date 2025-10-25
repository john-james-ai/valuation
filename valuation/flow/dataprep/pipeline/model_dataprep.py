#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/pipeline/model_dataprep.py                                 #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Saturday October 25th 2025 03:30:40 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Pipeline Base Module"""
from __future__ import annotations

from typing import Dict, Optional

from dataclasses import dataclass

from loguru import logger
import pandas as pd

from valuation.asset.dataset.dataset import Dataset
from valuation.asset.identity.dataset import DatasetPassport
from valuation.core.state import Status
from valuation.flow.base.pipeline import PipelineResult
from valuation.flow.dataprep.base.pipeline import DataPrepPipeline, DataPrepPipelineBuilder
from valuation.flow.dataprep.base.task.feature import FeatureEngineeringTask
from valuation.flow.dataprep.task.densify import DensifySalesDataTask
from valuation.infra.store.dataset import DatasetStore
from valuation.utils.split import TimeSeriesDataSplitter

# ------------------------------------------------------------------------------------------------ #
YEAR_TO_SPLIT = 1996


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelDataPipelineResult(PipelineResult):
    """Holds the results of a sales pipeline execution.

    Attributes:
        (inherited) DataPrepPipelineResult fields containing execution metadata and outputs.
    """

    train_val_dataset: Optional[Dataset] = None
    test_dataset: Optional[Dataset] = None

    records_in: int = 0
    records_out: int = 0
    records_imputed: int = 0
    precent_imputed: float = 0.0

    train_val_size: int = 0
    test_size: int = 0
    train_val_percent: float = 0.0
    test_percent: float = 0.0

    def end_pipeline(self) -> None:
        super().end_pipeline()
        self.records_imputed = self.records_out - self.records_in
        self.precent_imputed = (
            round(self.records_imputed / self.records_out * 100, 2) if self.records_out > 0 else 0.0
        )
        if self.records_out > 0:
            self.train_val_percent = round(self.train_val_size / self.records_out * 100, 2)
            self.test_percent = round(self.test_size / self.records_out * 100, 2)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ModelDataPipelineConfig:
    """Configuration for the Model Data Pipeline.

    Attributes:
        source (DatasetPassport): Passport for the source dataset.
        train_val_target (DatasetPassport): Passport for the training/validation target dataset.
        test_target (DatasetPassport): Passport for the test target dataset.
    """

    source: DatasetPassport
    target: DatasetPassport
    train_val_target: DatasetPassport
    test_target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
class ModelDataPipeline(DataPrepPipeline):

    _dataset_store: DatasetStore
    _result: ModelDataPipelineResult

    def __init__(
        self,
        config: ModelDataPipelineConfig,
        dataset_store: type[DatasetStore] = DatasetStore,
        result: type[ModelDataPipelineResult] = ModelDataPipelineResult,
    ) -> None:
        super().__init__(dataset_store=dataset_store)
        self._result = result(name=self.__class__.__name__)
        self._config = config

    def run(self, force: bool = False) -> Optional[ModelDataPipelineResult]:
        self._result.start_pipeline()

        try:
            if self._endpoint_exists():
                if force:
                    logger.info(
                        f"Datasets {self._config.train_val_target.label} and {self._config.test_target.label} already exist in the datastore. \nOverwriting as 'force' is set to True."  # type: ignore
                    )
                    # Remove any existing datasets if force is True
                    self._dataset_store.remove(passport=self._config.train_val_target)
                    self._dataset_store.remove(passport=self._config.test_target)
                else:
                    logger.info(
                        f"Datasets {self._config.train_val_target.label} and {self._config.test_target.label} already exist in the datastore. \nSkipping processing."  # type: ignore
                    )
                    self._result.train_val_dataset = self._dataset_store.get(
                        passport=self._config.train_val_target
                    )
                    self._result.test_dataset = self._dataset_store.get(
                        passport=self._config.test_target
                    )
                    self._result.status_obj = Status.SKIPPED
                    self._result.end_pipeline()
                    logger.info(self._result)
                    return self._result

            # Load data
            dataset = self._dataset_store.get(passport=self._config.source)
            df = dataset.data
            self._result.records_in = df.shape[0]

            for task in self._tasks:
                logger.info(f"Running task: {task.__class__.__name__}")
                df = task.run(df=df)
                if not task.validation.validate(df=df, classname=self.__class__.__name__):
                    msg = f"Validation failed for task: {task.__class__.__name__}"
                    logger.error(msg)
                    self._result.status_obj = Status.FAIL
                    raise ValueError(msg)

            # Store the full dataset after preprocessing
            full_dataset = Dataset(passport=self._config.target, df=df)
            self._dataset_store.remove(passport=self._config.target)  # Just in case it exists
            self._dataset_store.add(dataset=full_dataset)

            # Split the data into train_val and test sets
            splitter = TimeSeriesDataSplitter(year_to_split=YEAR_TO_SPLIT)
            split_dfs = splitter.split(df=df)

            self._create_and_persist_datasets(data=split_dfs)

            self._update_metrics(data=split_dfs)

            self._result.status_obj = Status.SUCCESS

            # Finalize the result and log it.
            self._result.end_pipeline()
            logger.info(self._result)

            return self._result

        except Exception as e:
            self._result.status_obj = Status.FAIL
            logger.critical(f"Pipeline execution failed with exception: {e}")
            self._result.end_pipeline()
            raise e

    def _create_and_persist_datasets(self, data: Dict[str, pd.DataFrame]) -> None:
        """Creates and persists the train_val and test datasets.

        Args:
            data (Dict[str, pd.DataFrame]): Dictionary containing 'train_val' and 'test' DataFrames.
        """
        train_val_dataset = Dataset(passport=self._config.train_val_target, df=data["train_val"])
        test_dataset = Dataset(passport=self._config.test_target, df=data["test"])
        self._dataset_store.add(dataset=train_val_dataset)
        self._dataset_store.add(dataset=test_dataset)
        self._result.train_val_dataset = train_val_dataset
        self._result.test_dataset = test_dataset

    def _endpoint_exists(self) -> bool:
        """Checks if the target dataset already exists in the dataset store.

        Returns:
            bool: True if the target dataset exists, False otherwise.
        """
        if self._config.train_val_target is None or self._config.test_target is None:
            return False

        return self._dataset_store.exists(
            dataset_id=self._config.train_val_target.id
        ) and self._dataset_store.exists(dataset_id=self._config.test_target.id)

    def _update_metrics(self, data: Dict[str, pd.DataFrame]) -> None:
        self._result.records_out = data["train_val"].shape[0] + data["test"].shape[0]
        self._result.train_val_size = data["train_val"].shape[0]
        self._result.test_size = data["test"].shape[0]


# ------------------------------------------------------------------------------------------------ #
class ModelDataPipelineBuilder(DataPrepPipelineBuilder):
    """Builder for TransformSalesDataPipeline instances.

    Use the fluent `with_*` methods to configure source, target and tasks, then call `build()`.

    Attributes:
        _pipeline (TransformSalesDataPipeline): The pipeline instance being built.
    """

    def __init__(self) -> None:
        super().__init__()
        self.reset()
        self._tasks = []
        self._config = None

    def reset(self) -> None:
        """Reset the internal builder state and create a fresh pipeline instance.

        Returns:
            None
        """
        self._pipeline = ModelDataPipeline

    def with_config(self, config: ModelDataPipelineConfig) -> ModelDataPipelineBuilder:

        self._config = config
        return self

    def with_densify(self) -> ModelDataPipelineBuilder:
        """Add the aggregate task to the pipeline and configure its validator.

        Returns:
            ModelDataPipelineBuilder: The builder instance.
        """

        task = DensifySalesDataTask()
        self._tasks.append(task)
        return self

    def with_feature_engineering(self) -> ModelDataPipelineBuilder:
        """Add the filter partial years task to the pipeline and configure its validator.
        Args:
            min_weeks (int): Minimum number of weeks required for a year to be considered full.
                Defaults to 50.

        Returns:
            ModelDataPipelineBuilder: The builder instance.
        """
        task = FeatureEngineeringTask()
        self._tasks.append(task)
        return self

    def build(
        self,
    ) -> ModelDataPipeline:
        """Finalize and return the built TransformSalesDataPipeline.

        Returns:
            TransformSalesDataPipeline: The constructed pipeline instance.
        """
        pipeline = self._pipeline(config=self._config)  # type: ignore
        for task in self._tasks:
            pipeline.add_task(task)
        self.reset()
        return pipeline
