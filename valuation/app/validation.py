#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/workflow/validation.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 08:29:37 pm                                              #
# Modified   : Thursday October 16th 2025 08:30:22 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Module for data validation results."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import pandas as pd


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
