#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/base.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 02:27:30 am                                                #
# Modified   : Wednesday October 15th 2025 01:33:16 pm                                             #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Base classes for data preparation tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
import pandas as pd

from valuation.config.data import DTYPES
from valuation.utils.data import DataClass
from valuation.utils.io import IOService


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskConfig(DataClass):
    """Base configuration class for tasks."""

    dataset_name: str
    input_location: Path
    output_location: Path


# ------------------------------------------------------------------------------------------------ #
class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    SUCCESS = "Success"
    FAILURE = "Failure"
    CRITICAL = "Critical Failure"
    WARNING = "Warning"
    SKIPPED = "Existing File - Skipped"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TaskReport(DataClass):
    """Holds the results of a task execution."""

    task_name: str
    dataset_name: str
    config: TaskConfig = field(default=None)  # Optional until setup
    started: Optional[datetime] = field(default=None)  # Optional until setup
    # The Engine manages this state
    ended: Optional[datetime] = field(default=None)  # Optional until teardown
    elapsed: Optional[float] = field(default=None)  # Optional until teardown
    records_in: int = 0  # Known at beginning of execute
    records_out: int = 0  # Only known after execute
    pct_change: Optional[float] = field(default=None)  # Only known after execute
    status: Optional[str] = field(default=None)  # Only known after execute
    validation: Optional[Validation] = field(default=None)  # Only known after execute

    def finalize(self) -> Optional[float]:
        """Calculates the percentage change in records from input to output."""
        if self.records_in is None or self.records_out is None or self.records_in == 0:
            self.pct_change = None
        else:
            self.pct_change = round(
                ((self.records_out - self.records_in) / self.records_in) * 100, 2
            )


# ------------------------------------------------------------------------------------------------ #
@dataclass
class Validation:
    """
    Holds the results of a data validation process, tracking overall status,
    failures, failed records grouped by reason, and general messages.
    """

    is_valid: bool = True
    num_failures: int = 0
    # Standardize to always be a dict mapping reason string to failed records DataFrame
    failed_records: Dict[str, pd.DataFrame] = field(default_factory=dict)

    # Internal list for non-record-specific messages (e.g., "Missing column X")
    _messages: List[str] = field(default_factory=list)

    def add_failed_records(self, reason: str, records: pd.DataFrame) -> None:
        """
        Adds records that failed validation under a specific reason.
        Note: This does NOT automatically add a message to _messages.
        You must call add_message() separately for a general message, or rely
        on the @property messages to summarize these failures.
        """
        if reason in self.failed_records:
            # Append new failures to existing ones for this reason
            self.failed_records[reason] = pd.concat(
                [self.failed_records[reason], records], ignore_index=True
            )
        else:
            # Create a new entry
            self.failed_records[reason] = records

        # Update overall failure count and status
        self.is_valid = False
        self.num_failures += len(records)

    def add_message(self, message: str) -> None:
        """
        Adds a general validation message (e.g., 'Missing file') that isn't tied
        to specific records. This always indicates a failure.
        """
        self.is_valid = False
        self.num_failures += 1
        self._messages.append(message)
        # Note: We do NOT increment num_failures here, as this typically
        # represents a structural failure, not a record-level failure count.
        # If a message relates to a failure count, the user must update it separately.

    @property
    def messages(self) -> str:
        """
        Generates a summary of all validation messages, combining general messages
        and a count-based summary of record failures.
        """
        # 1. Start with general messages
        all_messages = self._messages[:]  # Make a copy

        # 2. Add summary of record-level failures
        if self.failed_records:
            all_messages.append("\n--- Record-Level Validation Summary ---")
            for reason, df in self.failed_records.items():
                count = len(df)
                all_messages.append(f"{count} records failed validation due to: {reason}")
            all_messages.append("--------------------------------------")

        # 3. Add final summary
        if not self.is_valid:
            all_messages.append(f"\nTotal Failures Recorded: {self.num_failures}")

        return "\n".join(all_messages)


# ------------------------------------------------------------------------------------------------ #
class Task(ABC):
    """
    Abstract base class for a Single Input, Single Output (SISO) task.

    This class provides the standard lifecycle for data processing: setup,
    load, execute, validate, save, and guaranteed teardown/reporting.

    Attributes:
        _config (TaskConfig): The task configuration.
        _io (IOService): The service used for data input/output operations.
        _task_result (Optional[Union[pd.DataFrame, Any]]): The result of the task execution.
        _task_report (TaskReport): The report object tracking task status and metrics.
    """

    def __init__(self, config: TaskConfig, io: IOService = IOService) -> None:
        """
        Initializes the Task with configuration and I/O service.

        Args:
            config: Configuration object defining task parameters (e.g., input/output paths).
            io: I/O service implementation for loading and saving data.
        """
        self._config = config
        self._io = io
        self._is_valid: bool = True  # Default to True until validation
        self._task_result: Optional[Union[pd.DataFrame, Any]] = None
        self._task_report = TaskReport(
            task_name=self.__class__.__name__,
            dataset_name=self._config.dataset_name,
            config=self._config,
        )

    @property
    def config(self) -> TaskConfig:
        """Gets the task configuration."""
        return self._config

    @property
    def result(self) -> Optional[Union[pd.DataFrame, Any]]:
        """
        Gets the task result.

        Returns:
            The output data from the task, which may be a DataFrame or any other type.
        """
        return self._task_result

    @property
    def report(self) -> TaskReport:
        """
        Gets the task report.

        Returns:
            The TaskReport object containing execution status, metrics, and logs.
        """
        return self._task_report

    @property
    def is_valid(self) -> bool:
        """
        Gets the validation status of the task result.

        Returns:
            True if the task result passed validation, False otherwise.
        """
        return self._is_valid

    def run(
        self, data: Optional[pd.DataFrame] = None, force: bool = False
    ) -> Optional[pd.DataFrame]:
        """
        Runs the SISO task through its complete lifecycle: setup, execution, and cleanup.

        The execution is wrapped in a try/except/finally block to guarantee resource
        cleanup and final reporting, regardless of success, skip, or critical failure.

        Args:
            force: If True, ignores existing output and executes the task,
                deleting any existing output file first.

        Raises:
            Exception: Re-raises any critical exception encountered during execution.
            RuntimeError: Raised if data validation fails.
        """
        self._setup()

        try:

            if self._output_exists(force=force):
                self._task_report.status = TaskStatus.SKIPPED.value
                self._task_report.records_out = 0
                self._is_valid = True
                return  # Triggers the 'finally' block

            # Load input data if required and update record count
            input_data = data if data is not None else self._load(filepath=self._config.input_location)  # type: ignore
            self._task_report.records_in = len(input_data)

            # Execute the task and count output records
            self._task_result = self._execute(data=input_data)
            self._task_report.records_out = (
                len(self._task_result) if self._task_result is not None else 0
            )

            # Validate the output and update the report
            validation = self._validate(data=self._task_result)
            self._is_valid = validation.is_valid
            self._task_report.validation = validation

            # if validation fails, handle it (raises RuntimeError)
            if not validation.is_valid:
                self._handle_validation_failure()
            else:
                self._task_report.status = TaskStatus.SUCCESS.value
                logger.info(f"{self.__class__.__name__} - Validation Succeeded")
                self._save(df=self._task_result, filepath=self._config.output_location)

        except Exception as e:
            # Only update status to CRITICAL if it wasn't already set to FAILURE by validation handler
            if self._task_report.status != TaskStatus.FAILURE.value:
                self._task_report.status = TaskStatus.CRITICAL.value

            logger.exception(f"{self.__class__.__name__} - Critical Failure: {e}")
            raise e

        finally:
            self._teardown()
            return self._task_result

    @abstractmethod
    def _execute(self, data: Union[pd.DataFrame, Any]) -> Union[pd.DataFrame, Any]:
        """
        Executes the core logic of the task.

        Subclasses must implement the specific data transformation or processing logic here.

        Args:
            data: The input data loaded by `_load`.

        Returns:
            The processed output data.
        """
        pass

    @abstractmethod
    def _validate(self, data: Union[pd.DataFrame, Any]) -> Validation:
        """
        Validates the output data.

        Subclasses must implement checks to ensure the output meets quality and structural standards.

        Args:
            data: The output data generated by `_execute`.

        Returns:
            A Validation object indicating if the data is valid and any associated messages.
        """
        pass

    def _validate_columns(
        self, validation: Validation, data: pd.DataFrame, required_columns: List[str]
    ) -> Validation:
        """
        Validates that the DataFrame contains all required columns with correct data types.

        Args:
            validation: The Validation object to update with any issues found.
            data: The DataFrame to validate.
            required_columns: List of column names that must be present in the DataFrame.

        Returns:
            The updated Validation object.
        """
        for col in required_columns:
            if col not in data.columns:
                validation.add_message(f"Missing {col} column.")
            else:
                if not str(data[col].dtype) == DTYPES[col]:
                    validation.add_message(
                        f"{col} column is not of type {DTYPES[col]}. It is type {data[col].dtype}."
                    )
        return validation

    def _setup(self) -> None:
        """
        Sets up the task environment and records the start time in the report.
        """
        self._task_report.started = datetime.now()

    def _teardown(self) -> None:
        """
        Cleans up the task environment, calculates elapsed time, finalizes the report,
        and logs the report's final status.
        """
        self._task_report.ended = datetime.now()
        self._task_report.elapsed = (
            self._task_report.ended - self._task_report.started
        ).total_seconds()  # type: ignore
        self._task_report.finalize()
        logger.info(self._task_report)

    def _handle_validation_failure(self) -> None:
        """
        Handles all steps required when output data validation fails.

        This method logs failure details, updates the report status to FAILURE,
        and raises a RuntimeError to halt execution.

        Raises:
            RuntimeError: Always raised to ensure execution flow jumps to the
                outer exception handler and finally block.
        """

        validation = self._task_report.validation  # Retrieve the data here!

        # 1. Prepare and log the main error message
        msg = f"{self.__class__.__name__} - Validation Failed"
        logger.error(msg)

        # 2. Log all specific validation messages
        if validation and validation.messages:
            logger.error(validation.messages)

        # 3. Update the task report status
        self._task_report.status = TaskStatus.FAILURE.value

        # 4. Critical: Raise the exception to halt execution
        raise RuntimeError(msg)

    def _load(self, filepath: Path) -> pd.DataFrame:
        """
        Loads a single data file from the raw data directory using the I/O service.

        The loaded data is also subjected to a type-casting check using DTYPES.

        Args:
            filepath: The path to the file to be loaded.

        Returns:
            A DataFrame containing the loaded data with applied dtypes.
        """
        logger.debug(f"Loading data from {filepath}")

        data = self._io.read(filepath=filepath)
        # Ensure correct data types
        if isinstance(data, pd.DataFrame):
            logger.debug(f"Applying data types to loaded DataFrame")
            data = data.astype({k: v for k, v in DTYPES.items() if k in data.columns})
        else:
            logger.debug(
                f"Loaded data is type {type(data)} and not a DataFrame. Skipping dtype application."
            )
        return data

    def _save(self, df: pd.DataFrame, filepath: Path) -> None:
        """
        Saves a DataFrame to the processed data directory using the I/O service.

        Args:
            df: The DataFrame to save.
            filepath: The path to the file to be saved.
        """
        logger.debug(f"Saving data to {filepath}")
        self._io.write(data=df, filepath=filepath)

    def _delete(self, location: Path) -> None:
        """
        Deletes a file from the specified location.

        Args:
            location: The path of the file to delete.
        """
        location.unlink(missing_ok=True)

    def _exists(self, location: Path) -> bool:
        """
        Checks if a file exists at the specified location.

        Args:
            location: The path to a file for the existence check.

        Returns:
            True if the file exists, False otherwise.
        """
        return location.exists()

    def _output_exists(self, force: bool = False) -> bool:
        """
        Determines whether the task should be skipped because the output file already exists.

        If `force` is True, the existing file is deleted, and the task proceeds.

        Args:
            force: If True, forces the task to run by deleting existing output.

        Returns:
            True if the task should be skipped (i.e., output file exists and force is False),
            False otherwise.
        """
        if force:
            self._delete(location=self._config.output_location)
            output_exists = False
        else:
            output_exists = self._exists(location=self._config.output_location) and not force

        if output_exists:
            logger.info(
                f"{self.__class__.__name__} - Output file already exists. Using cached data."
            )
        else:
            logger.info(f"{self.__class__.__name__}  - Starting")
        return output_exists
