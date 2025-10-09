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
# Modified   : Thursday October 9th 2025 03:07:16 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from pathlib import Path

from loguru import logger

from valuation.io import IOService

# ------------------------------------------------------------------------------------------------ #
# --- 1. Directories and Filepaths ---
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LOGS_DIR = PROJ_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure logs directory exists
LOGS_DATASET = LOGS_DIR / "dataset.log"
LOGS_MODELING = LOGS_DIR / "modeling.log"
LOGS_VALUATION = LOGS_DIR / "valuation.log"

CONFIG_FILEPATH = PROJ_ROOT / "config.yaml"
CONFIG_CATEGORY_FILEPATH = "category_filenames"


# ------------------------------------------------------------------------------------------------ #
class ConfigReader:
    """Reads configuration settings from a YAML file."""

    def __init__(self, io: IOService = IOService()):
        self.config = io.read(filepath=CONFIG_FILEPATH)

    def read(self, key: str, default=None):
        return self.config.get(key, default)


# ------------------------------------------------------------------------------------------------ #
# Configure logging
try:
    from tqdm import tqdm

    logger.remove(0)

    # 1. Configure a specific sink for ETL logs
    # This sink will only accept logs from any module inside "modules.etl"
    logger.add(
        LOGS_DATASET,
        level="DEBUG",
        filter=lambda record: record["name"] is not None
        and record["name"].startswith("valuation.dataset"),
        rotation="1 week",
    )

    # 2. Configure another specific sink for Modeling logs
    logger.add(
        LOGS_MODELING,
        level="DEBUG",
        filter=lambda record: record["name"] is not None
        and record["name"].startswith("valuation.modeling"),
        rotation="1 week",
    )

    # 3. Configure another specific sink for Valuation logs
    logger.add(
        LOGS_VALUATION,
        level="DEBUG",
        filter=lambda record: record["name"] is not None
        and record["name"].startswith("valuation.valuation"),
        rotation="1 week",
    )

    # 4. Configure a sink for console output that works well with tqdm
    # https://github.com/Delgan/loguru/issues/135
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
