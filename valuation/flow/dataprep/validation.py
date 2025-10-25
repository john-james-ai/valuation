#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/dataprep/validation.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 08:29:37 pm                                              #
# Modified   : Saturday October 25th 2025 08:43:27 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for df validation results."""
from __future__ import annotations

from typing import DefaultDict, Dict, List

from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger
import polars as pl

from valuation.core.validation import Severity
from valuation.infra.file.io import IOService


# ------------------------------------------------------------------------------------------------ #
class Validation:

    def __init__(self) -> None:
        self._validators: Dict[str, Validator] = {}
        self._is_valid = True
        self._num_errors = 0
        self._num_warnings = 0

    @property
    def is_valid(self) -> bool:
        """Indicates whether all validations passed.

        Returns:
            bool: True if all validations passed, False otherwise.
        """
        return self._is_valid

    @property
    def num_errors(self) -> int:
        """Returns the total number of validation errors.

        Returns:
            int: Number of validation errors.
        """
        return self._num_errors

    @property
    def num_warnings(self) -> int:
        """Returns the total number of validation warnings.

        Returns:
            int: Number of validation warnings.
        """
        return self._num_warnings

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

    def validate(self, df: pl.DataFrame, classname: str) -> bool:
        """Run all registered validators against provided df.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            bool: True if all validators pass, False otherwise.
        """
        logger.debug(f"Validating df for class {classname}...")
        for validator in self._validators.values():
            validator.validate(df=df, classname=classname)
            if not validator.is_valid:
                self._is_valid = False
            self._num_errors += validator.num_errors
            self._num_warnings += validator.num_warnings
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
    """Base class for df validators.

    Attributes:
        _classname (Optional[str]): Last validated class name.
        _is_valid (bool): Whether last validation passed.
        _num_errors (int): Count of failures recorded.
        _anomaly_records (Dict[str, Dict[str, pl.DataFrame]]): Per-class failure anomaly_types and DataFrames.
        _messages (List[str]): General validation messages.
    """

    def __init__(self) -> None:
        self._classname = None
        self._is_valid = True
        self._num_errors = 0
        self._num_warnings = 0
        self._anomalies = []
        # Nested defaultdicts ending with an empty polars DataFrame
        self._anomaly_records: DefaultDict[
            str, DefaultDict[str, DefaultDict[str, DefaultDict[str, pl.DataFrame]]]
        ] = defaultdict(  # Level 1: classname
            lambda: defaultdict(  # Level 2: severity
                lambda: defaultdict(  # Level 3: validator_name
                    lambda: defaultdict(
                        lambda: pl.DataFrame()
                    )  # Level 4: anomaly_type -> pl.DataFrame
                )
            )
        )

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
    def _validate(self, df: pl.DataFrame, classname: str) -> None:
        """Performs the actual validation logic.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        pass

    def validate(self, df: pl.DataFrame, classname: str) -> bool:
        """Public method to validate df and generate a report.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            bool: True if validation passed, False otherwise.
        """
        logger.debug(f"\tRunning validator {self.__class__.__name__} for class {classname}...")
        self._validate(df=df, classname=classname)
        return self._is_valid

    def report(self) -> None:
        """Print or log the validation report for a given class.

        Returns:
            None
        """
        if self._is_valid:
            logger.debug(f"{self.__class__.__name__} validation passed. No issues found.")
        else:
            anomalies = pl.DataFrame(self._anomalies)
            warnings = anomalies.filter(pl.col("severity") == str(Severity.WARNING))
            if warnings.height > 0:
                # Convert to pandas for string rendering (polars DataFrame.to_string() is not supported)
                logger.debug(
                    f"\tValidation Warning Report for {self.__class__.__name__}:\n{warnings.to_pandas().to_string(index=False)}"
                )

            errors = anomalies.filter(pl.col("severity") == str(Severity.ERROR))
            if errors.height > 0:
                # Convert to pandas for string rendering (polars DataFrame.to_string() is not supported)
                logger.error(
                    f"\tValidation Error Report for {self.__class__.__name__}:\n{errors.to_pandas().to_string(index=False)}"
                )

    def _log_anomaly_records(self) -> None:
        """Log failed record DataFrames to CSV files for each anomaly type.

        Iterates the internal anomaly records structure and writes non-empty DataFrames
        to CSV files using the IOService. A timestamped directory is used to avoid collisions.

        Returns:
            None
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # 1. Loop through classname
        for classname, severity_dict in self._anomaly_records.items():
            # 2. Loop through severity
            for severity, validator_dict in severity_dict.items():
                # 3. Loop through validator_name
                for validator_name, anomaly_type_dict in validator_dict.items():
                    # 4. Loop through anomaly_type and the DataFrame
                    for anomaly_type, df in anomaly_type_dict.items():

                        # Only log if there are actual error records
                        if not df.is_empty():
                            # Create a more descriptive/unique filename
                            log_name = f"{classname}_{severity}_{validator_name}_{anomaly_type}"

                            log_filepath = self._get_log_filepath(
                                anomaly_type=log_name, timestamp=timestamp
                            )
                            logger.debug(
                                f"Logging {df.height} records for {log_name} to {log_filepath.name}"
                            )
                            IOService.write(data=df, filepath=log_filepath)

    def _add_anomalies(
        self, severity: str, classname: str, anomaly_type: str, column: str, count: int
    ) -> None:

        self._is_valid = self._is_valid and severity != str(Severity.ERROR)
        self._num_errors += count if severity == str(Severity.ERROR) else 0
        self._num_warnings += count if severity == str(Severity.WARNING) else 0
        anomaly = {
            "severity": severity,
            "classname": classname,
            "validator": self.__class__.__name__,
            "anomaly_type": anomaly_type,
            "column": column,
            "count": count,
        }
        self._anomalies.append(anomaly)

    def _add_anomaly_records(
        self, classname: str, severity: str, anomaly_type: str, records: pl.DataFrame
    ) -> None:
        """Add failed records for a specific anomaly_type and update failure counts.

        Args:
            classname (str): The name of the class (e.g., file) being validated.
            anomaly_type (str): The anomaly_type for failure.
            records (pl.DataFrame): The DataFrame containing the failed records.

        Returns:
            None
        """
        # 1. Add this check: no point running if there are no error records
        if records.is_empty():
            return

        validator_name = self.__class__.__name__

        existing_records = self._anomaly_records[classname][severity][validator_name][anomaly_type]

        # Concatenate polars DataFrames
        self._anomaly_records[classname][severity][validator_name][anomaly_type] = pl.concat(
            [existing_records, records]
        )

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
            / f"dfset.{timestamp}"
            / "validation_anomaly_records"
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

    def _validate(self, df: pl.DataFrame, classname: str) -> None:
        """Validate that all required columns are present.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        for col in self._required_columns:
            if col not in df.columns:

                self._add_anomalies(
                    classname=classname,
                    severity=str(Severity.ERROR),
                    anomaly_type="MissingColumn",
                    column=col,
                    count=1,
                )
                msg = f"{classname} {self.__class__.__name__} validation found column'{col}' missing in the dfset."
                logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class ColumnTypeValidator(Validator):
    """Validator checking that columns match expected dtypes.

    Args:
        column_types (Dict[str, pl.DataType | str]): Mapping from column name to expected polars dtype
            or legacy dtype string (e.g. "float64", "Int64", "datetime64[ns]").
    """

    def __init__(self, column_types: Dict[str, object]) -> None:
        super().__init__()
        # Normalize expected dtype values to either Polars DataType or canonical string.
        self._column_types = {
            col: self._normalize_expected_dtype(dt) for col, dt in column_types.items()
        }

    @staticmethod
    def _normalize_expected_dtype(dt: object) -> object:
        """Normalize an expected dtype specification to either a Polars DataType or canonical string.

        Accepts:
          - pl.DataType instances -> returned unchanged
          - strings like 'float64', 'Int64', 'datetime64[ns]', 'string' -> mapped to pl types
          - other dtype objects with a 'name' attribute -> their name is used as fallback
        """
        if isinstance(dt, pl.DataType):
            return dt
        if isinstance(dt, str):
            s = dt.strip().lower()
            if s in ("string", "str", "utf8", "utf-8"):
                return pl.Utf8
            if s in ("float64", "float", "double"):
                return pl.Float64
            if s in ("int64", "int", "i64"):
                return pl.Int64
            if s in ("int32", "i32"):
                return pl.Int32
            if s in ("bool", "boolean"):
                return pl.Boolean
            if s in ("datetime64[ns]", "datetime", "datetime64", "dt64[ns]", "Datetime", "Date"):
                return pl.Datetime("ns")
            # fallback to the raw string so we still have something to compare
            return s
        # try to extract a name attribute (numpy/pandas dtype)
        try:
            name = getattr(dt, "name", None)
            if isinstance(name, str):
                return ColumnTypeValidator._normalize_expected_dtype(name)
        except Exception:
            pass
        # unknown -> return as-is (comparison later will use string forms)
        return dt

    def _validate(self, df: pl.DataFrame, classname: str) -> None:
        """Validate that columns have expected df types.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        schema = df.schema
        for col in df.columns:
            actual_dtype = schema.get(col)
            expected = self._column_types.get(col)
            if expected is None:
                continue

            # If expected is a Polars DataType, do direct compare.
            mismatch = False
            if isinstance(expected, pl.DataType):
                mismatch = actual_dtype != expected
            else:
                # Compare canonical string representations (case-insensitive)
                actual_str = str(actual_dtype).lower() if actual_dtype is not None else "none"
                expected_str = str(expected).lower()
                mismatch = actual_str != expected_str

            if mismatch:
                self._add_anomalies(
                    classname=classname,
                    severity=str(Severity.ERROR),
                    anomaly_type="InvalidColumnType",
                    column=col,
                    count=1,
                )
                msg = (
                    f"{classname} {self.__class__.__name__} validation found column '{col}' "
                    f"to have an invalid type '{actual_dtype}'; expected '{expected}'."
                )
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

    def _validate(self, df: pl.DataFrame, classname: str) -> None:
        """Validate that specified columns contain no negative values.

        Args:
            df (pl.DataFrame): The DataFrame to validate.
            classname (str): The name of the class being validated (for logging purposes).

        Returns:
            None
        """
        for col in self._columns:
            if col not in df.columns:
                continue

            # Prefer expression-based count to avoid materializing filters unnecessarily.
            try:
                cnt_expr_df = df.select((pl.col(col) < 0).sum().alias("cnt"))
                # select returns a DataFrame; get scalar safely
                cnt = int(cnt_expr_df.row(0)[0]) if cnt_expr_df.height > 0 else 0
            except Exception as e:
                # If expression fails (e.g., type mismatch), try a safe coercion path and log.
                logger.debug(
                    f"NonNegativeValidator: expression count failed for column '{col}': {e}"
                )
                try:
                    # attempt to coerce to float for numeric comparison; avoid changing original df
                    cnt = int(
                        df.with_columns(pl.col(col).cast(pl.Float64).alias("__tmp"))
                        .select((pl.col("__tmp") < 0).sum().alias("cnt"))
                        .row(0)[0]
                    )
                except Exception:
                    # Last resort: eager filter (may be expensive)
                    try:
                        cnt = int(df.filter(pl.col(col) < 0).height)
                    except Exception:
                        cnt = 0

            if cnt > 0:
                # Materialize only when there are negatives
                negative_values = df.filter(pl.col(col) < 0)

                anomaly_type = "NegativeValue"
                self._add_anomalies(
                    classname=classname,
                    severity=str(Severity.WARNING),
                    anomaly_type=anomaly_type,
                    column=col,
                    count=cnt,
                )
                self._add_anomaly_records(
                    classname=classname,
                    severity=str(Severity.WARNING),
                    anomaly_type=anomaly_type,
                    records=negative_values,
                )
                msg = (
                    f"{classname} {self.__class__.__name__} validation found "
                    f"{cnt} rows with negative values in the '{col}' column."
                )
                logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class RangeValidator(Validator):
    """Validator ensuring a numeric column's values fall within an inclusive range.

    Args:
        column (str): Column name to validate.
        min_value (float): Inclusive minimum allowed value.
        max_value (float): Inclusive maximum allowed value.
    """

    def __init__(self, column: str, min_value: float, max_value: float) -> None:
        super().__init__()

        self._column = column
        self._min_value = min_value
        self._max_value = max_value

    def _validate(self, df: pl.DataFrame, classname: str) -> None:

        below_min = df.filter(pl.col(self._column) < self._min_value)
        above_max = df.filter(pl.col(self._column) > self._max_value)
        out_of_range = pl.concat([below_min, above_max])
        if not out_of_range.is_empty():
            anomaly_type = "OutOfRangeValue"
            self._add_anomalies(
                classname=classname,
                severity=str(Severity.ERROR),
                anomaly_type=anomaly_type,
                column=self._column,
                count=out_of_range.height,
            )
            self._add_anomaly_records(
                classname=classname,
                severity=str(Severity.ERROR),
                anomaly_type=anomaly_type,
                records=out_of_range,
            )
            msg = f"{classname} {self.__class__.__name__} validation found {out_of_range.height} rows with out of range values in the '{self._column}' column."
            logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class DensifyValidator(Validator):
    """Validator ensuring the DataFrame has the expected number of rows after densification.

    Args:
        categories (int): Number of unique categories expected.
        weeks (int): Number of unique weeks expected.
        stores (int): Number of unique stores expected.
    """

    def __init__(self, categories: int = 28, weeks: int = 365, stores: int = 93) -> None:
        super().__init__()
        self._categories = categories
        self._weeks = weeks
        self._stores = stores

    def _validate(self, df: pl.DataFrame, classname: str) -> None:
        expected_rows = self._categories * self._weeks * self._stores
        actual_rows = df.height
        if actual_rows != expected_rows:
            anomaly_type = "IncorrectRowCount"
            self._add_anomalies(
                classname=classname,
                severity=str(Severity.ERROR),
                anomaly_type=anomaly_type,
                column="row_count",
                count=1,
            )
            msg = f"{classname} {self.__class__.__name__} validation found incorrect row count: expected {expected_rows}, got {actual_rows}."
            logger.debug(msg)


# ------------------------------------------------------------------------------------------------ #
class ValidationBuilder:
    """Builder for constructing Validation objects with various validators."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> ValidationBuilder:
        self._validation = Validation()
        return self

    def with_missing_column_validator(self, required_columns: List[str]) -> ValidationBuilder:
        name = "ColumnValidator"
        self._validation.add_validator(name, MissingColumnValidator(list(required_columns)))
        return self

    def with_column_type_validator(self, column_types: Dict[str, pl.DataType]) -> ValidationBuilder:
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

    def with_densify_validator(self) -> ValidationBuilder:
        name = "DensifyValidator"
        self._validation.add_validator(name, DensifyValidator())
        return self

    def build(self) -> Validation:
        validation = self._validation
        self.reset()
        return validation
