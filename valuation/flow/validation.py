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
# Modified   : Tuesday October 21st 2025 12:23:43 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for data validation results."""
from __future__ import annotations

from typing import Dict, List

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger
import pandas as pd

from valuation.infra.file.io import IOService

# ------------------------------------------------------------------------------------------------ #
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 0)  # prevents wrapping
pd.set_option("display.colheader_justify", "left")


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
    failed_records: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)

    # Internal list for non-record-specific messages (e.g., "Missing column X")
    _messages: List[str] = field(default_factory=list)

    def add_failed_records(self, classname: str, reason: str, records: pd.DataFrame) -> None:
        """
        Adds records that failed validation under a specific reason.
        """
        # Ensure nested dict structure exists
        if classname not in self.failed_records:
            self.failed_records[classname] = {}

        if reason in self.failed_records[classname]:
            self.failed_records[classname][reason] = pd.concat(
                [self.failed_records[classname][reason], records], ignore_index=True
            )
        else:
            self.failed_records[classname][reason] = records

        self.is_valid = False
        self.num_failures += len(records)

    def log_failed_records(self) -> None:
        """
        Logs the failed records for each reason using print statements.
        This is a simple way to output the failures; in a real application,
        you might want to use a logging framework or write to a file.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for classname, reasons in self.failed_records.items():
            for reason, df in reasons.items():
                log_filepath = self._get_log_filepath(
                    classname=classname, reason=reason, timestamp=timestamp
                )
                logger.info(
                    f"Logging failed {len(df)} records for reason: {reason} to {log_filepath.name}"
                )
                IOService.write(data=df, filepath=log_filepath)

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
            all_messages.append(f"\n\n--- Record-Level Validation Summary ---")
            for reason, df in self.failed_records.items():
                count = len(df)
                all_messages.append(f"{count} records failed validation due to: {reason}")
            all_messages.append("--------------------------------------")

        # 3. Add final summary
        if not self.is_valid:
            all_messages.append(f"\nTotal Failures Recorded: {self.num_failures}")

        return "\n".join(all_messages)

    def _get_log_filepath(self, classname: str, reason: str, timestamp: str | None = None) -> Path:
        """Generates a log file path for failed records and ensures parent directories exist.

        Args:
            reason (str): The reason for failure used to name the file.

            timestamp (str | None): Timestamp string to include in the directory name. If None, the current timestamp is used.

        Returns:
            pathlib.Path: Path to the log file with parent directories created.
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        safe_reason = reason.replace(" ", "_").replace("'", "")
        file_path = (
            Path("logs")
            / f"dataset.{timestamp}"
            / f"{classname}"
            / "validation_failed_records"
            / f"{safe_reason}.csv"
        )

        # Sanitize any double dots in the path string representation
        file_path = Path(str(file_path).replace("..", "."))

        # Ensure parent directories exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        return file_path
