#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config.py                                                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Friday October 10th 2025 06:31:17 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path
from typing import Any, Dict

from loguru import logger

from valuation.utils.io import IOService

# ------------------------------------------------------------------------------------------------ #
# --- 1. Directories and Filepaths ---
PROJ_ROOT = Path(__file__).resolve().parents[1]

# Define directories
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REFERENCES_DIR = PROJ_ROOT / "references"

# Models directory
MODELS_DIR = PROJ_ROOT / "models"

# Logs directory and files
LOGS_DIR = PROJ_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
LOGS_DATASET = LOGS_DIR / "dataset.log"
LOGS_MODELING = LOGS_DIR / "modeling.log"
LOGS_VALUATION = LOGS_DIR / "valuation.log"

# Reference Filepaths
WEEK_DECODE_TABLE_FILEPATH = REFERENCES_DIR / "week_decode_table.csv"

# Configuration file and keys
CONFIG_FILEPATH = PROJ_ROOT / "config.yaml"
CONFIG_CATEGORY_FILEPATH = "category_filenames"

# Filenames
SALES_DATA_FILENAME = "sales.csv"
SAME_STORE_SALES_DATA_FILENAME = "same_store_sales.csv"
CATEGORY_DATA_FILENAME = "category.csv"
STORE_DATA_FILENAME = "store.csv"
TRAIN_DATA_FILENAME = "train.csv"
VALIDATION_DATA_FILENAME = "validation.csv"
TEST_DATA_FILENAME = "test.csv"


# Module Names
DATASET_MODULE = "valuation.dataset"
MODELING_MODULE = "valuation.modeling"
VALUATION_MODULE = "valuation.valuation"


# ------------------------------------------------------------------------------------------------ #
class ConfigReader:
    """Reads configuration settings from a YAML file."""

    def __init__(self, io: IOService = IOService()) -> None:
        self.config = io.read(filepath=CONFIG_FILEPATH)

    def read(self, key: str, default=None) -> Dict[str, Any]:
        return self.config.get(key, default)


# ------------------------------------------------------------------------------------------------ #
# Configure logging
try:
    logger.remove()  # Remove all previously added handlers

    # -------------------------------------------------------------------------------------------- #
    #                                LOG FILE SINKS CONFIGURATION                                  #
    # -------------------------------------------------------------------------------------------- #
    # 1. Configure a specific sink for ETL logs
    logger.add(
        LOGS_DATASET,
        level="DEBUG",
        filter=lambda record: record["name"] == DATASET_MODULE,
        rotation="1 week",
    )

    # 2. Configure another specific sink for Modeling logs
    logger.add(
        LOGS_MODELING,
        level="DEBUG",
        filter=lambda record: record["name"] == MODELING_MODULE,
        rotation="1 week",
    )

    # 3. Configure another specific sink for Valuation logs
    logger.add(
        LOGS_VALUATION,
        level="DEBUG",
        filter=lambda record: record["name"] == VALUATION_MODULE,
        rotation="1 week",
    )


except ModuleNotFoundError:
    pass
