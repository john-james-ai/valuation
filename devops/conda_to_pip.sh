#!/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/conda_to_pip.sh                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 23rd 2025 06:09:23 pm                                              #
# Modified   : Thursday October 23rd 2025 06:28:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
PACKAGE_NAME=$1

if [ -z "$PACKAGE_NAME" ]; then
    echo "Error: No package name provided."
    echo "Usage: $0 <package-name>"
    exit 1  # Exit should be inside the 'then' block
fi         

echo "Reinstalling package '$PACKAGE_NAME' from pip."
# Uninstall from conda first
echo "...uninstalling '$PACKAGE_NAME' from conda."
conda uninstall -y $PACKAGE_NAME

# Uninstall using pip
echo "...uninstalling '$PACKAGE_NAME' from pip."
pip uninstall -y $PACKAGE_NAME

# Reinstall using pip
if pip install $PACKAGE_NAME --upgrade --force-reinstall then
    echo "Pip package '$PACKAGE_NAME' reinstalled successfully. ✅"
else
    echo "Error: Failed to reinstall pip package '$PACKAGE_NAME'. ❌"
    exit 1
fi

# Housekeeping
echo "Cleaning conda cache."
conda clean -a -y


echo "Exporting Conda environment top level dependencies to environment.yml"
conda env export --from-history > environment.yml