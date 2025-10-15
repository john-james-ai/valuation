#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/dataprep/pipeline.py                                                     #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday October 14th 2025 10:53:05 pm                                               #
# Modified   : Tuesday October 14th 2025 11:01:10 pm                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
class DataPrepPipeline:
    """A class to manage the data preparation pipeline."""

    def __init__(self):
        """Initializes the DataPrepPipeline."""
        self.tasks = []

    def add_task(self, task):
        """Adds a task to the pipeline.

        Args:
            task: An instance of a data preparation task.
        """
        self.tasks.append(task)

    def run(self, force: bool = False):
        """Runs all tasks in the pipeline.

        Args:
            force (bool): If True, forces re-execution of all tasks.
        """
        df = None
        for task in self.tasks:
            df = task.run(data=df, force=force)
