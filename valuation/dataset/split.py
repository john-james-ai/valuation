#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataset/split.py                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 10th 2025 06:45:03 pm                                                #
# Modified   : Saturday October 11th 2025 01:39:09 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Data Splitter Module"""
from pathlib import Path

from loguru import logger
from pydantic.dataclasses import dataclass

from valuation.config.data_prep import DataPrepBaseConfig, DataPrepCoreConfig
from valuation.dataset.base import DataPrep
from valuation.utils.data import DataFramePartitioner
from valuation.utils.io import IOService
from valuation.utils.print import Printer

# ------------------------------------------------------------------------------------------------ #


@dataclass
class PathsConfig:
    """Holds all file paths for the splitting process."""

    input_filepath: Path
    train_filepath: Path
    validation_filepath: Path
    test_filepath: Path


@dataclass
class SplitterConfig(DataPrepBaseConfig):
    """Holds all parameters for the data splitter."""

    core_config: DataPrepCoreConfig
    paths: PathsConfig
    val_col: str
    train_size: float
    val_size: float
    shuffle: bool = False
    random_state: int = None


# ------------------------------------------------------------------------------------------------ #
class DatasetSplitter(DataPrep):
    """Splits datasets into training, validation, and test sets.

    Args:
        splitter (DataFramePartitioner, optional): An instance of DataFramePartitioner for splitting
            the dataset. Defaults to DataFramePartitioner.
        printer (Printer, optional): An instance of Printer for printing information. Defaults to
            Printer.
        io (IOService, optional): An instance of IOService for input/output operations. Defaults to
            IOService.

    """

    def __init__(
        self,
        splitter: type[DataFramePartitioner] = DataFramePartitioner,
        printer: Printer = Printer,
        io: IOService = IOService,
    ) -> None:
        super().__init__(io)
        self._printer = printer
        self._splitter = splitter()
        self._splits = None

    @property
    def info(self) -> None:
        """Prints information about the dataset splits."""
        if self._splits:
            self._printer.print_dict(title="Dataset Split", data=self._splits["meta"])
        else:
            print("No dataset has been split. Run the prepare method to split a dataset.")

    def prepare(self, config: SplitterConfig) -> None:
        """Prepares the dataset by splitting it into training, validation, and test sets.
        Args:
            config (SplitterConfig): Configuration parameters for the splitter.
        """
        if self._use_cache(config=config):
            return

        df = self.load(filepath=config.paths.input_filepath)
        self._splits = self._splitter.split_by_proportion_of_values(
            df=df,
            val_col=config.val_col,
            train_size=config.train_size,
            val_size=config.val_size,
        )
        try:
            self.save(df=self._splits["data"]["train"], filepath=config.paths.train_filepath)
            self.save(
                df=self._splits["data"]["validation"], filepath=config.paths.validation_filepath
            )
            self.save(df=self._splits["data"]["test"], filepath=config.paths.test_filepath)
        except Exception as e:
            logger.error(f"Error saving split datasets: {e}")
            raise

    def _use_cache(self, config: SplitterConfig) -> bool:
        """Determines whether to use cached data based on file existence and force flag.

        Args:
            config (DataPrepSISOConfig): Configuration object containing core settings.
        Returns:
            bool: True if cached data should be used, False otherwise.

        """
        all_files_exist = (
            self.exists(filepath=config.paths.train_filepath)
            and self.exists(filepath=config.paths.validation_filepath)
            and self.exists(filepath=config.paths.test_filepath)
        )

        if config.core_config.force or not all_files_exist:
            self.delete(filepath=config.paths.train_filepath)
            self.delete(filepath=config.paths.validation_filepath)
            self.delete(filepath=config.paths.test_filepath)
            use_cache = False
        else:
            use_cache = all_files_exist and not config.core_config.force

        if use_cache:
            logger.info(
                f"{config.core_config.task_name} - Output file already exists. Using cached data."
            )
        else:
            logger.info(f"{config.core_config.task_name}  - Starting")
        return use_cache
