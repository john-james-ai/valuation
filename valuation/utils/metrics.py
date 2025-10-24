#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/utils/metrics.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 01:38:47 am                                                #
# Modified   : Friday October 24th 2025 01:40:43 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #


def compute_smape(y_true, y_pred):
    """
    Computes the Symmetric Mean Absolute Percentage Error (SMAPE).

    y_true: Actual values (numpy array or list)
    y_pred: Forecast/Predicted values (numpy array or list)
    """
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculate the numerator: |y_true - y_pred|
    numerator = np.abs(y_true - y_pred)

    # Calculate the denominator: (|y_true| + |y_pred|) / 2.
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Calculate the individual SMAPE terms (prevents division by zero when both are 0)
    # np.divide handles division by zero by inserting 'inf' or 'nan', which are then set to 0.
    smape_terms = np.divide(
        numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0
    )

    # Take the mean of all terms and multiply by 100 for percentage
    return 100 * np.mean(smape_terms)


def compute_wape(y_true, y_pred, as_percent=True):
    """
    Computes the Weighted Absolute Percentage Error (WAPE).

    y_true: Actual values (numpy array or list)
    y_pred: Forecast/Predicted values (numpy array or list)
    as_percent: Boolean to return the result as a percentage (default: True)
    """
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Sum of absolute errors
    sum_abs_errors = np.sum(np.abs(y_true - y_pred))

    # Sum of absolute actual values
    sum_abs_actuals = np.sum(np.abs(y_true))

    # Prevent division by zero if total actual sales are zero
    if sum_abs_actuals == 0:
        return 0.0

    # Calculate WAPE
    wape = sum_abs_errors / sum_abs_actuals

    if as_percent:
        return 100 * wape
    else:
        return wape
