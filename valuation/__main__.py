#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mercor Dominick's Fine Foods Acquisition Analysis                                   #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/__main__.py                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mercor-dominicks-acquisition-analysis              #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 9th 2025 11:01:16 pm                                               #
# Modified   : Thursday October 9th 2025 11:12:20 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Main module for the Valuation package."""
import typer

from valuation.config import CONFIG_CATEGORY_FILEPATH, ConfigReader
from valuation.dataset import SalesDataProcessor

# ------------------------------------------------------------------------------------------------ #
app = typer.Typer()


@app.command()
def main(
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        case_sensitive=False,
        help="Whether to force reprocessing if the file already exists.",
    )
):
    """Processes raw sales data into a cleaned and aggregated dataset."""

    # Obtain categories and filenames from config
    config_reader = ConfigReader()
    category_filenames = config_reader.read(CONFIG_CATEGORY_FILEPATH)

    # ----------------------------------------------
    # Instantiate the processor
    processor = SalesDataProcessor()

    # Run the processor pipeline
    processor.process(category_filenames=category_filenames, force=force)


if __name__ == "__main__":
    app()
