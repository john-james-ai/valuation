#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/config/filepaths.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday October 8th 2025 02:52:13 pm                                              #
# Modified   : Saturday October 11th 2025 09:27:23 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

# ------------------------------------------------------------------------------------------------ #
# --- 1. Directories and Filepaths ---
PROJ_ROOT = Path(__file__).resolve().parents[2]

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

# Configuration file and keys
CONFIG_FILEPATH = PROJ_ROOT / "config.yaml"
CONFIG_CATEGORY_FILEPATH = "category_filenames"

# DATA FILEPATHS
# External data files
WEEK_DECODE_TABLE_FILEPATH = EXTERNAL_DATA_DIR / "week_decode_table.csv"
FINANCIALS_BENCHMARKS = EXTERNAL_DATA_DIR / "financials.yaml"
# Processed data files
DATASET_PROFILE_FILEPATH = PROCESSED_DATA_DIR / "profile.csv"
SALES_DATA_FILEPATH = PROCESSED_DATA_DIR / "sales.csv"
CATEGORY_DATA_FILEPATH = PROCESSED_DATA_DIR / "category.csv"
STORE_DATA_FILEPATH = PROCESSED_DATA_DIR / "store.csv"
TRAIN_DATA_FILEPATH = PROCESSED_DATA_DIR / "train.csv"
VALIDATION_DATA_FILEPATH = PROCESSED_DATA_DIR / "validation.csv"
TEST_DATA_FILEPATH = PROCESSED_DATA_DIR / "test.csv"
