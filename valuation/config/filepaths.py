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
# Modified   : Tuesday October 14th 2025 09:53:22 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

# ------------------------------------------------------------------------------------------------ #
# --- 1. Directories and Filepaths ---
PROJ_ROOT = Path(__file__).resolve().parents[2]

# DIRECTORIES
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INGEST_DATA_DIR = DATA_DIR / "ingest"
CLEAN_DATA_DIR = DATA_DIR / "clean"
AGGREGATE_DATA_DIR = DATA_DIR / "aggregate"
ENRICH_DATA_DIR = DATA_DIR / "enrich"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
REFERENCES_DIR = PROJ_ROOT / "references"

# MODELS
MODELS_DIR = PROJ_ROOT / "models"

# LOG FILES
LOGS_DIR = PROJ_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
LOGS_DATASET = LOGS_DIR / "dataset.log"
LOGS_MODELING = LOGS_DIR / "modeling.log"
LOGS_VALUATION = LOGS_DIR / "valuation.log"

# Configuration filepath
CONFIG_FILEPATH = PROJ_ROOT / "config.yaml"

# EXTERNAL FILES
WEEK_DECODE_TABLE_FILEPATH = EXTERNAL_DATA_DIR / "week_decode_table.csv"
FINANCIALS_BENCHMARKS = EXTERNAL_DATA_DIR / "financials.yaml"

# DATA FILES
# Raw Stage
FILEPATH_CUSTOMER_RAW = RAW_DATA_DIR / "ccount_stata.zip"
FILEPATH_STORE_DEMO_RAW = RAW_DATA_DIR / "demo_stata.zip"
# Ingest Stage
FILEPATH_SALES_INGEST = INGEST_DATA_DIR / "sales_ingest.parquet"
FILEPATH_CUSTOMER_INGEST = INGEST_DATA_DIR / "customer_ingest.csv"
FILEPATH_STORE_DEMO_INGEST = INGEST_DATA_DIR / "store_demo_ingest.csv"
FILEPATH_SALES_PROFILE = INGEST_DATA_DIR / "sales_profile.csv"
FILEPATH_CUSTOMER_PROFILE = INGEST_DATA_DIR / "customer_profile.csv"
FILEPATH_STORE_DEMO_PROFILE = INGEST_DATA_DIR / "store_demo_profile.csv"

# Clean Stage
FILEPATH_SALES_CLEAN = CLEAN_DATA_DIR / "sales_clean.csv"
FILEPATH_CUSTOMER_CLEAN = CLEAN_DATA_DIR / "customer_clean.csv"
FILEPATH_STORE_DEMO_CLEAN = CLEAN_DATA_DIR / "store_demo_clean.csv"

# Aggregate Stage
FILEPATH_SALES_AGGREGATE_YEAR = AGGREGATE_DATA_DIR / "sales_aggregate_year.csv"  # Year
FILEPATH_SALES_AGGREGATE_SCW = (
    AGGREGATE_DATA_DIR / "sales_aggregate_scw.csv"
)  # Store Category Week
FILEPATH_SALES_AGGREGATE_SY = AGGREGATE_DATA_DIR / "sales_aggregate_sy.csv"  # Store Year
FILEPATH_SALES_AGGREGATE_CY = AGGREGATE_DATA_DIR / "sales_aggregate_cy.csv"  # Category Year


FILEPATH_CUSTOMER_AGGREGATE_YEAR = AGGREGATE_DATA_DIR / "customer_aggregate_year.csv"  # Year
FILEPATH_CUSTOMER_AGGREGATE_SCW = (
    AGGREGATE_DATA_DIR / "customer_aggregate_scw.csv"
)  # Store Category Week
FILEPATH_CUSTOMER_AGGREGATE_SY = AGGREGATE_DATA_DIR / "customer_aggregate_sy.csv"  # Store Year
FILEPATH_CUSTOMER_AGGREGATE_CY = AGGREGATE_DATA_DIR / "customer_aggregate_cy.csv"  # Category Year

FILEPATH_STORE_DEMO_AGGREGATE_YEAR = AGGREGATE_DATA_DIR / "store_demo_aggregate_year.csv"  # Year
FILEPATH_STORE_DEMO_AGGREGATE_SCW = (
    AGGREGATE_DATA_DIR / "store_demo_aggregate_scw.csv"
)  # Store Category Week
FILEPATH_STORE_DEMO_AGGREGATE_SY = AGGREGATE_DATA_DIR / "store_demo_aggregate_sy.csv"  # Store Year
FILEPATH_STORE_DEMO_AGGREGATE_CY = (
    AGGREGATE_DATA_DIR / "store_demo_aggregate_cy.csv"
)  # Category Year

# Enrich Stage
FILEPATH_SALES_ENRICH = ENRICH_DATA_DIR / "sales_enrich.csv"
