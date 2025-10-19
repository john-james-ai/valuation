#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/app/dataprep/task.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Sunday October 19th 2025 01:22:33 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from typing import Dict, List, Optional, Union, cast

from abc import abstractmethod
from dataclasses import dataclass, field

from loguru import logger
import pandas as pd

from valuation.app.base.task import Task, TaskConfig, TaskResult
from valuation.app.state import Status
from valuation.app.validation import Validation
from valuation.asset.dataset.base import DTYPES, Dataset
from valuation.asset.identity import DatasetPassport, DatasetID
from valuation.infra.store.dataset import DatasetStore


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetTaskConfig(TaskConfig):
    """Base configuration class for tasks."""

    source: Union[DatasetPassport, Dict[str, str]]
    target: DatasetPassport


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetTaskResult(TaskResult):
    
    dataset_name: Optional[str] = None

    # Record counts
    records_in: Optional[int] = None
    records_out: Optional[int] = None
    pct_change: Optional[float] = None

    # Validation results
    validation: Validation = field(default_factory=Validation)

    # Contains the output data from the task
    dataset: Dataset = field(default=None)


    def end_task(self) -> None:

        super().end_task()
        if self.records_in is None or self.records_out is None or self.records_in == 0:
            self.pct_change = None
        else:
            self.pct_change = round(
                ((self.records_in - self.records_out) / self.records_in) * 100, 2
            )



# ------------------------------------------------------------------------------------------------ #
class DatasetTask(Task):


    def __init__(
        self,
        config: DatasetTaskConfig,
        dataset_store: DatasetStore = DatasetStore,
    ) -> None:
        super().__init__(config=config)

        self._dataset_store = dataset_store
        self._result = DatasetTaskResult(task_name=self.task_name, config=self._config)



    @abstractmethod
    def _execute(self, df: pd.DataFrame, **kwargs) -> Dataset:
        """Executes the core logic of the task.

        Subclasses must implement this method to perform the specific
        data processing or transformation that the task is responsible for.

        Args:
            dataset (Dataset): The input dataset to be processed.

        Returns:
            Dataset: The processed output dataset.
        """

    @abstractmethod
    def _validate_result(self, result: TaskResult) -> TaskResult:
        """Validates the output data and updates the TaskResult.

        Subclasses must implement specific validation logic to ensure
        the output data meets expected standards. This method should
        update the `validation` attribute of the provided `TaskResult`
        object.

        Args:
            result: The TaskResult object containing the output data
                to be validated.

        Returns:
            The updated TaskResult object with validation results.
        """
        pass

    @abstractmethod
    def run(self, dataset: Dataset) -> Dataset:
        """Executes the full task lifecycle: execution, validation, and reporting.

        This method orchestrates the task's operation within a context that
        handles timing, status updates, and error logging. It ensures that a
        complete TaskResult object is returned, whether the task succeeds or fails.

        Args:
            data: Optional[pd.DataFrame]: The input data to be processed by the task.

        Returns:
            TaskResult: An object containing the final status, metrics,
                validation info, and output data of the task run.

        Raises:
            RuntimeError: If input data is missing or empty, or if the
                validation fails.
        """
        

        try:

                # 1. Capture the size of the input dataset.
                result.records_in = cast(int, dataset.nrows)

                # 2. Check validity of target configuration
                if not isinstance(self._config.target, Passport):
                    raise RuntimeError("Target configuration must be a Passport instance.")

                # 3. Check if output already exists to potentially skip processing.
                if self._dataset_store.exists(
                    name=self._config.target.name, stage=self._config.target.stage
                ):

                    result.status = Status.EXISTS.value
                    # Get the output dataset from the asset store
                    dataset_out = self._dataset_store.get(
                        name=self._config.target.name,
                        stage=self._config.target.stage,
                        entity
                    )
                    # Cast to a dataset object and assign to result
                    dataset_out = cast(Dataset, dataset_out)
                    result.records_out = cast(int, dataset_out.nrows)
                    result.dataset = dataset_out
                    return result

                # 4. Otherwise execute the task
                df_out = self._execute(df=dataset.data)

                # 2. Create the output dataset object and count output records.
                dataset_out = Dataset(passport=self._config.target, df=df_out)
                result.records_out = cast(int, result.dataset.nrows)

                # Validate the result by calling the subclass's implementation.
                result = self._validate_result(result=result)

                # Handle validation failure.
                if not result.validation.is_valid:  # type: ignore
                    self._handle_validation_failure(validation=result.validation)
        finally:
            return self._finalize(result=result, dataset=dataset_out)

    

    def _validate_columns(
        self, validation: Validation, data: pd.DataFrame, required_columns: List[str]
    ) -> Validation:
        """Validates that required columns are present and of correct types.

        Args:
            validation (Validation): The current validation object to update.
            data (pd.DataFrame): The DataFrame to validate.
            required_columns (List[str]): List of required column names.

        Returns:
            Validation: The updated validation object."""
        for col in required_columns:
            if col not in data.columns:
                validation.add_message(f"Missing required column: '{col}'.")

            else:
                dtype = str(data[col].dtype)
                if not dtype == DTYPES[col]:
                    validation.add_message(
                        f"Column '{col}' of type {dtype} should be type {DTYPES[col]}."
                    )
        return validation

    def _handle_validation_failure(self, validation: Validation) -> None:
        """Handles logging and raises an exception on validation failure.

        This method centralizes the failure logic. It logs all specific
        validation messages and then raises a `RuntimeError` to halt execution.

        Args:
            validation: The Validation object containing failure messages.

        Raises:
            RuntimeError: Always raised to ensure execution is halted and
                the failure is propagated.
        """

        msg = f"{self.__class__.__name__} - Validation Failed"
        logger.error(msg)
        logger.error(f"Validation Messages:\n{validation.messages}")
        raise RuntimeError(msg)




# ------------------------------------------------------------------------------------------------ #
@dataclass
class SISODatasetTaskConfig(DatasetTaskConfig):
    """Base configuration class for tasks."""

    source: DatasetPassport
    target: DatasetPassport
# ------------------------------------------------------------------------------------------------ #
class SISODatasetTask(DatasetTask):
    def __init__(
        self,
        config: SISODatasetTaskConfig,
        dataset_store: DatasetStore = DatasetStore,
    ) -> None:
        super().__init__(config=config, dataset_store=dataset_store)
        # Recast config to SISO type for easier access
        self._config = cast(SISODatasetTaskConfig, self._config)

    @abstractmethod
    def _execute(self, df: pd.DataFrame, **kwargs) -> Dataset:
        """Executes the core logic of the task.

        Subclasses must implement this method to perform the specific
        data processing or transformation that the task is responsible for.

        Args:
            dataset (Dataset): The input dataset to be processed.

        Returns:
            Dataset: The processed output dataset.
        """        
        
    @abstractmethod
    def _validate_result(self, result: TaskResult) -> TaskResult:
        """Validates the output data and updates the TaskResult.

        Subclasses must implement specific validation logic to ensure
        the output data meets expected standards. This method should
        update the `validation` attribute of the provided `TaskResult`
        object.

        Args:
            result: The TaskResult object containing the output data
                to be validated.

        Returns:
            The updated TaskResult object with validation results.
        """
        pass

    def run(self, dataset: Dataset) -> Optional[Dataset]:
        
        # Initaialize the result object and start the task
        result = DatasetTaskResult(task_name=self.task_name, config=self._config)
        result.start_task()
        
        # Check if output already exists to potentially skip processing.
        output_dataset_id = DatasetID.from_passport(self._config.target)        
        if self._dataset_store.exists(output_dataset_id):
            dataset_out = self._dataset_store.get(
                name=self._config.source.name,
                stage=self._config.source.stage
            )
            dataset_out = cast(Dataset, dataset_out)
            result.status = Status.EXISTS.value
            result.end_task()
            logger.info(result)
            return dataset_out
        try:

            # 1. Capture the size of the input dataset.
            result.records_in = cast(int, dataset.nrows)

            # 2. Execute the task
            df_out = self._execute(df=dataset.data)

            # 3. Create the output dataset object and count output records.
            result.dataset = Dataset(passport=self._config.target, df=df_out)
            result.records_out = cast(int, result.dataset.nrows)            

            # 4. Validate the result 
            result = self._validate_result(result=result)
            result = cast(DatasetTaskResult, result)

            # Handle validation failure.
            if not result.validation.is_valid:  # type: ignore
                result.status = Status.FAILURE.value
                self._handle_validation_failure(validation=result.validation)
            else:
                result.status = Status.SUCCESS.value
                self._dataset_store.add(dataset=result.dataset)
                
        finally: 
            result.end_task()
            logger.info(result)     
            result = cast(DatasetTaskResult, result)
            return result.dataset
                   
            
        

    
    def _finalize(self, dataset: Dataset, status: Status) -> None:

        self._result.dataset = dataset
        self._result.records_out = cast(int, dataset.nrows)        
        self._dataset_store.add(dataset=dataset)
        self._result.end_task(status=status)        