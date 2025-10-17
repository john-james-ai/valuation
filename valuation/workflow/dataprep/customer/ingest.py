#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/customer/ingest.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday October 12th 2025 11:51:12 pm                                                #
# Modified   : Thursday October 16th 2025 07:29:02 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Union

import pandas as pd

from valuation.dataprep.base import Task, TaskConfig, Validation


# ------------------------------------------------------------------------------------------------ #
class IngestCustomerDataTask(Task):
    """Ingests a raw customer data file.

    Args:
        config (TaskConfig): Configuration for the ingestion process.
    """

    def __init__(self, config: TaskConfig) -> None:
        super().__init__(config=config)

    def _execute(self, data: Union[pd.DataFrame, Any]) -> pd.DataFrame:

        return data

    def _validate_result(self, data: pd.DataFrame) -> Validation:
        validation = Validation()
        COLUMNS = ["Week", "Store"]
        validation = self._validate_columns(
            validation=validation, data=data, required_columns=COLUMNS
        )
        return validation
