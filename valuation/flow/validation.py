#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/validation.py                                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 08:29:37 pm                                              #
# Modified   : Tuesday October 21st 2025 08:18:42 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for data validation results."""
from __future__ import annotations

from typing import Dict, List

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path

from loguru import logger
import pandas as pd

from valuation.asset.dataset.base import DTYPES
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Validation:
    """Container for a collection of validators and overall validation state.

    Attributes:
        _validators (Dict[str, Validator]): Mapping of validator names to validator instances.
        _is_valid (bool): Overall validation flag, True if all validators passed.
    """

    def __init__(self) -> None:
        self._validators = {}
        self._is_valid = True

    def add_validator(self, name: str, validator: Validator) -> None:
        """Register a validator under a name.

        Args:
            name (str): The name to register the validator under.
            validator (Validator): The validator instance to register.

        Returns:
            None
        """
        self._validators[name] = validator

    def validate(self, data: pd.DataFrame, classname: str) -> bool:
        """Run all registered validators against provided data.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            bool: True if all validators pass, False otherwise.
        """
        for validator in self._validators.values():
            validator.validate(data=data, classname=classname)
            if not validator.is_valid:
                self._is_valid = False
        return self._is_valid


# ------------------------------------------------------------------------------------------------ #
class Validator(ABC):
    """Base class for data validators.

    Attributes:
        _classname (Optional[str]): Last validated class name.
        _is_valid (bool): Whether last validation passed.
        _num_failures (int): Count of failures recorded.
        _failed_records (Dict[str, Dict[str, pd.DataFrame]]): Per-class failure reasons and DataFrames.
        _messages (List[str]): General validation messages.
    """

    def __init__(self) -> None:
        self._classname = None
        self._is_valid = True
        self._num_failures = 0
        self._failed_records: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._messages: List[str] = []

    @property
    def is_valid(self) -> bool:
        """Indicates whether the last validation was successful.

        Returns:
            bool: True if last validation passed, False otherwise.
        """
        return self._is_valid

    @property
    def num_failures(self) -> int:
        """Returns the number of failures from the last validation.

        Returns:
            int: Number of failures recorded.
        """
        return self._num_failures

    @property
    def messages(self) -> str:
        """Generate a summary string combining general messages and record-level summaries.

        Returns:
            str: The consolidated validation messages and summaries.
        """
        # 1. Start with general messages
        all_messages = self._messages[:]  # Make a copy

        # 2. Add summary of record-level failures
        if self._failed_records:
            all_messages.append(f"\n\n--- Record-Level Validation Summary ---")
            for self._classname, reasons in self._failed_records.items():
                for reason, df in reasons.items():
                    count = len(df)
                    all_messages.append(f"{count} records failed validation due to: {reason}")
            all_messages.append("--------------------------------------")

        # 3. Add final summary
        if not self._is_valid:
            all_messages.append(f"\nTotal Failures Recorded: {self.num_failures}")

        return "\n".join(all_messages)

    @abstractmethod
    def _validate(self, data: pd.DataFrame, classname: str) -> None:
        """Performs the actual validation logic.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        pass

    def validate(self, data: pd.DataFrame, classname: str) -> bool:
        """Public method to validate data and generate a report.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            bool: True if validation passed, False otherwise.
        """
        logger.debug(f"Validating data for class {classname}...")
        self._validate(data=data, classname=classname)
        self.report(classname=classname)
        return self._is_valid

    def report(self, classname: str) -> None:
        """Print or log the validation report for a given class.

        Args:
            classname (str): The name of the class for which the report is generated.

        Returns:
            None
        """
        if self._is_valid:
            logger.info(f"Validation Report for {classname}: No issues found.")
        else:
            logger.info(f"Validation Report for {classname}:\n{self.messages}")
            self.log_failed_records()

    def log_failed_records(self) -> None:
        """Log failed record DataFrames to CSV files for each reason.

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for self._classname, reasons in self._failed_records.items():
            for reason, df in reasons.items():
                log_filepath = self._get_log_filepath(reason=reason, timestamp=timestamp)
                logger.info(
                    f"Logging failed {len(df)} records for reason: {reason} to {log_filepath.name}"
                )
                IOService.write(data=df, filepath=log_filepath)

    def add_message(self, message: str) -> None:
        """Add a general (non-record) validation message and mark validation as failed.

        Args:
            message (str): Message describing the validation issue.

        Returns:
            None
        """
        self._is_valid = False
        self._num_failures += 1
        self._messages.append(message)
        # Note: We do NOT increment num_failures here, as this typically
        # represents a structural failure, not a record-level failure count.
        # If a message relates to a failure count, the user must update it separately.

    def add_failed_records(self, reason: str, records: pd.DataFrame) -> None:
        """Add failed records for a specific reason and update failure counts.

        Args:
            reason (str): The reason for failure.
            records (pd.DataFrame): The DataFrame containing the failed records.

        Returns:
            None
        """
        self._is_valid = False
        self._num_failures += len(records)

        # Ensure there's a nested dict for this classname
        self._failed_records.setdefault(self._classname, {})

        existing = self._failed_records[self._classname].get(reason)
        if existing is None:
            # store a copy to avoid accidental external mutation
            self._failed_records[self._classname][reason] = records.copy()
        else:
            # both `existing` and `records` are DataFrames; concat them
            self._failed_records[self._classname][reason] = pd.concat(
                [existing, records], ignore_index=True
            )

    def _get_log_filepath(self, reason: str, timestamp: str | None = None) -> Path:
        """Generate a log file path for failed records and ensure parent directories exist.

        Args:
            reason (str): The reason for failure used to name the file.
            timestamp (str | None): Timestamp string to include in the directory name. If None, current timestamp is used.

        Returns:
            Path: Path to the log file with parent directories created.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_reason = reason.replace(" ", "_").replace("'", "")
        file_path = (
            Path("logs")
            / f"dataset.{timestamp}"
            / "validation_failed_records"
            / f"{safe_reason}.csv"
        )

        # Sanitize any double dots in the path string representation
        file_path = Path(str(file_path).replace("..", "."))

        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path


# ------------------------------------------------------------------------------------------------ #
class MissingColumnValidator(Validator):
    """Validator ensuring required columns are present.

    Args:
        required_columns (List[str]): List of column names that must be present in the DataFrame.
    """

    def __init__(self, required_columns: List[str]) -> None:
        super().__init__()
        self._required_columns = required_columns

    def _validate(self, data: pd.DataFrame, classname: str) -> None:
        """Validate that all required columns are present.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        logger.debug(f"\tValidating required columns for class {classname}.")
        for col in self._required_columns:
            if col not in data.columns:
                message = f"Class: {classname}: Required column {col} is missing from the dataset."
                self.add_message(message)


# ------------------------------------------------------------------------------------------------ #
class ColumnTypeValidator(Validator):
    """Validator checking that columns match expected dtypes.

    Args:
        column_types (Dict[str, type]): Mapping from column name to expected Python type.
    """

    def __init__(self, column_types: Dict[str, type]) -> None:
        super().__init__()
        self._column_types = column_types

    def _validate(self, data: pd.DataFrame, classname: str) -> None:
        """Validate that columns have expected data types.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        logger.debug(f"\tValidating column data types for class {classname}.")

        for col in data.columns:
            dtype = str(data[col].dtype)
            if not dtype == DTYPES[col]:
                self.add_message(
                    f"Class {classname}: Column '{col}' of type {dtype} should be type {DTYPES[col]}."
                )


# ------------------------------------------------------------------------------------------------ #
class NonNegativeColumnValidator(Validator):
    """Validator ensuring specified columns contain only non-negative values.

    Args:
        columns (List[str]): List of column names that must be non-negative.
    """

    def __init__(self, columns: List[str]) -> None:
        super().__init__()

        self._columns = columns

    def _validate(self, data: pd.DataFrame, classname: str) -> None:
        """Validate that specified columns contain no negative values.

        Args:
            data (pd.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        logger.debug(f"\tValidating non-negative values for class {classname}.")
        for col in self._columns:
            if col in data.columns:
                negative_values = data[data[col] < 0]
                if not negative_values.empty:
                    reason = f"Class: {classname}: Negative Values in column '{col}'"
                    self.add_failed_records(reason=reason, records=negative_values)


# ------------------------------------------------------------------------------------------------ #
class ValidationBuilder:
    """Builder for constructing Validation objects with various validators."""

    def __init__(self) -> None:
        self._validation = Validation()

    def with_missing_column_validator(self, required_columns: List[str]) -> ValidationBuilder:
        name = "ColumnValidator"
        self._validation.add_validator(name, MissingColumnValidator(required_columns))
        return self

    def with_column_type_validator(self, column_types: Dict[str, type]) -> ValidationBuilder:
        name = "ColumnTypeValidator"
        self._validation.add_validator(name, ColumnTypeValidator(column_types))
        return self

    def with_non_negative_column_validator(self, columns: List[str]) -> ValidationBuilder:
        name = "NonNegativeColumnValidator"
        self._validation.add_validator(name, NonNegativeColumnValidator(columns))
        return self

    def build(self) -> Validation:
        return self._validation
