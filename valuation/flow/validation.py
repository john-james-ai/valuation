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
# Modified   : Wednesday October 22nd 2025 03:44:16 am                                             #
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

    def __init__(self) -> None:
        self._validators = {}
        self._is_valid = True

    @property
    def is_valid(self) -> bool:
        """Indicates whether all validations passed.

        Returns:
            bool: True if all validations passed, False otherwise.
        """
        return self._is_valid

    @property
    def validators(self) -> Dict[str, Validator]:
        """Returns the registered validators.

        Returns:
            Dict[str, Validator]: Mapping of validator names to instances.
        """
        return self._validators

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
        logger.debug(f"Validating data for class {classname}...")
        for validator in self._validators.values():
            validator.validate(data=data, classname=classname)
            if not validator.is_valid:
                self._is_valid = False
        return self._is_valid

    def report(self) -> None:
        """Generate validation reports for all validators.

        Returns:
            None
        """
        for _, validator in self._validators.items():
            validator.report()


# ------------------------------------------------------------------------------------------------ #
class Validator(ABC):
    """Base class for data validators.

    Attributes:
        _classname (Optional[str]): Last validated class name.
        _is_valid (bool): Whether last validation passed.
        _num_errors (int): Count of failures recorded.
        _error_records (Dict[str, Dict[str, pd.DataFrame]]): Per-class failure anomaly_types and DataFrames.
        _messages (List[str]): General validation messages.
    """

    def __init__(self) -> None:
        self._classname = None
        self._is_valid = True
        self._num_errors = 0
        self._num_warnings = 0
        self._anomalies: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, int]]]]] = {}
        self._anomaly_records: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}

    @property
    def is_valid(self) -> bool:
        """Indicates whether the last validation was successful.

        Returns:
            bool: True if last validation passed, False otherwise.
        """
        return self._is_valid

    @property
    def num_errors(self) -> int:
        """Returns the number of failures from the last validation.

        Returns:
            int: Number of failures recorded.
        """
        return self._num_errors

    @property
    def num_warnings(self) -> int:
        """Returns the number of failures from the last validation.

        Returns:
            int: Number of failures recorded.
        """
        return self._num_warnings

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
        logger.debug(f"\tRunning validator {self.__class__.__name__} for class {classname}...")
        self._validate(data=data, classname=classname)
        return self._is_valid

    def report(self) -> None:
        """Print or log the validation report for a given class.

        Args:
            classname (str): The name of the class for which the report is generated.

        Returns:
            None
        """
        if self._is_valid:
            logger.debug(f"{self.__class__.__name__} validation passed. No issues found.")
        else:
            errors = []
            for classname, validator in self._anomalies.items():
                for validator_name, error_dict in validator.items():
                    for anomaly_type, column in error_dict.items():
                        for column, count in column.items():
                            errors.append(
                                {
                                    "classname": classname,
                                    "validator": validator_name,
                                    "anomaly_type": anomaly_type,
                                    "column": column,
                                    "count": count,
                                }
                            )
            if errors:
                error_df = pd.DataFrame(errors)
                logger.info(
                    f"\tValidation Report for   {self.__class__.__name__}:\n{error_df.to_string(index=False)}"
                )

    def _log_error_records(self) -> None:
        """Log failed record DataFrames to CSV files for each anomaly_type.

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for _, anomaly_types in self._anomaly_records.items():
            for anomaly_type, df in anomaly_types.items():
                log_filepath = self._get_log_filepath(
                    anomaly_type=anomaly_type, timestamp=timestamp
                )
                logger.debug(
                    f"Logging failed {len(df)} records for anomaly_type: {anomaly_type} to {log_filepath.name}"
                )
                IOService.write(data=df, filepath=log_filepath)

    def _add_anomalies(
        self, severity: str, classname: str, anomaly_type: str, column: str, count: int
    ) -> None:

        self._is_valid = False
        self._num_errors += count
        self._anomalies.setdefault(classname, {})
        self._anomalies[classname].setdefault(severity, {})
        self._anomalies[classname][severity].setdefault(self.__class__.__name__, {})
        self._anomalies[classname][severity][self.__class__.__name__].setdefault(anomaly_type, {})
        self._anomalies[classname][severity][self.__class__.__name__][anomaly_type].setdefault(
            column, {}
        )
        self._anomalies[classname][severity][self.__class__.__name__][anomaly_type][column] += count

    def _add_error_records(self, classname: str, anomaly_type: str, records: pd.DataFrame) -> None:
        """Add failed records for a specific anomaly_type and update failure counts.

        Args:
            classname (str): The name of the class (e.g., file) being validated.
            anomaly_type (str): The anomaly_type for failure.
            records (pd.DataFrame): The DataFrame containing the failed records.

        Returns:
            None
        """
        # 1. Add this check: no point running if there are no error records
        if records.empty:
            return

        self._is_valid = False
        self._num_errors += len(records)

        validator_name = self.__class__.__name__

        # 2. This gets the dict for {'anomaly_type': pd.DataFrame, ...}
        #    It creates the classname and validator_name keys if they don't exist.
        error_dict = self._anomaly_records.setdefault(classname, {}).setdefault(validator_name, {})

        # 3. Get the existing DataFrame, or an empty one if this is the first time.
        existing_records = error_dict.setdefault(anomaly_type, pd.DataFrame())

        # 4. Concat and re-assign.
        #    Use .copy() on records to match the 'store a copy' intent from your original code.
        error_dict[anomaly_type] = pd.concat([existing_records, records.copy()], ignore_index=True)

    def _get_log_filepath(self, anomaly_type: str, timestamp: str | None = None) -> Path:
        """Generate a log file path for failed records and ensure parent directories exist.

        Args:
            anomaly_type (str): The anomaly_type for failure used to name the file.
            timestamp (str | None): Timestamp string to include in the directory name. If None, current timestamp is used.

        Returns:
            Path: Path to the log file with parent directories created.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_anomaly_type = anomaly_type.replace(" ", "_").replace("'", "")
        file_path = (
            Path("logs")
            / f"dataset.{timestamp}"
            / "validation_error_records"
            / f"{safe_anomaly_type}.csv"
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
        for col in self._required_columns:
            if col not in data.columns:
                self._add_errors(
                    classname=classname, anomaly_type="MissingColumn", column=col, count=1
                )
                msg = f"{classname} {self.__class__.__name__} validation found column'{col}' missing in the dataset."
                logger.debug(msg)


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

        for col in data.columns:
            dtype = str(data[col].dtype)
            if not dtype == DTYPES[col]:
                self._add_errors(
                    classname=classname, anomaly_type="InvalidColumnType", column=col, count=1
                )
                msg = f"{classname} {self.__class__.__name__} validation found column '{col}' to have an invalid type '{dtype}'; expected '{DTYPES[col]}'."
                logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class NonNegativeValidator(Validator):
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
        for col in self._columns:
            if col in data.columns:
                negative_values = data[data[col] < 0]
                if not negative_values.empty:
                    anomaly_type = "NegativeValue"
                    self._add_errors(
                        classname=classname,
                        anomaly_type=anomaly_type,
                        column=col,
                        count=len(negative_values),
                    )
                    self._add_error_records(
                        classname=classname, anomaly_type=anomaly_type, records=negative_values
                    )
                    msg = f"{classname} {self.__class__.__name__} validation found {len(negative_values)} rows with negative values in the '{col}' column."
                    logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class RangeValidator(Validator):
    def __init__(self, column: str, min_value: float, max_value: float) -> None:
        super().__init__()

        self._column = column
        self._min_value = min_value
        self._max_value = max_value

    def _validate(self, data: pd.DataFrame, classname: str) -> None:

        below_min = data[data[self._column] < self._min_value]
        above_max = data[data[self._column] > self._max_value]
        out_of_range = pd.concat([below_min, above_max])
        if not out_of_range.empty:
            anomaly_type = "OutOfRangeValue"
            self._add_errors(
                classname=classname,
                anomaly_type=anomaly_type,
                column=self._column,
                count=len(out_of_range),
            )
            self._add_error_records(
                classname=classname, anomaly_type=anomaly_type, records=out_of_range
            )
            msg = f"{classname} {self.__class__.__name__} validation found {len(out_of_range)} rows with out of range values in the '{self._column}' column."
            logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class ValidationBuilder:
    """Builder for constructing Validation objects with various validators."""

    def __init__(self) -> None:
        self._validation = Validation()

    def with_missing_column_validator(self, required_columns: List[str]) -> ValidationBuilder:
        name = "ColumnValidator"
        self._validation.add_validator(name, MissingColumnValidator(list(required_columns)))
        return self

    def with_column_type_validator(self, column_types: Dict[str, type]) -> ValidationBuilder:
        name = "ColumnTypeValidator"
        self._validation.add_validator(name, ColumnTypeValidator(column_types))
        return self

    def with_non_negative_column_validator(self, columns: List[str]) -> ValidationBuilder:
        name = "NonNegativeValidator"
        self._validation.add_validator(name, NonNegativeValidator(columns))
        return self

    def with_range_validator(
        self, column: str, min_value: float, max_value: float
    ) -> ValidationBuilder:
        name = "RangeValidator"
        self._validation.add_validator(
            name, RangeValidator(column=column, min_value=min_value, max_value=max_value)
        )
        return self

    def build(self) -> Validation:
        return self._validation
