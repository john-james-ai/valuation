#!/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/rebuild_env.sh                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 04:05:21 pm                                              #
# Modified   : Saturday October 25th 2025 01:51:24 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# --- START Conda Initialization for Non-Interactive Scripts ---
# Find the base conda directory
CONDA_BASE=$(conda info --base)
# Source the conda setup script to enable commands like 'conda activate'
source "$CONDA_BASE/etc/profile.d/conda.sh"
# --- END Conda Initialization ---

# Define variables
ENV_FILE="environment.yml"

# Extract the environment name from the .yml file (assumes 'name: my_env' is the second line)
ENV_NAME=$(grep "name:" $ENV_FILE | awk '{print $2}')

# Check if the environment name was found
if [ -z "$ENV_NAME" ]; then
    echo "Error: Could not find environment name in $ENV_FILE. Ensure the file starts with 'name: your_env_name'."
    exit 1
fi

echo "--- Conda Environment Clean Reinstallation Script ---"
echo "Environment Name: $ENV_NAME"
echo "Environment File: $ENV_FILE"

# 1. Deactivate the environment if it's currently active
# Note: Since we sourced conda.sh, 'conda info' should work reliably.
CURRENT_ENV=$(conda info --envs | grep '^\*' | awk '{print $1}')
if [ "$CURRENT_ENV" == "$ENV_NAME" ]; then
    echo "Deactivating $ENV_NAME..."
    conda deactivate
fi

# 2. Remove the existing environment (if it exists)
echo "Removing existing environment '$ENV_NAME'..."
conda env remove --name $ENV_NAME --yes

# 3. Clean up the conda cache
# This step removes downloaded packages and index caches, ensuring a fresh download and install.
echo "Cleaning all Conda caches to ensure a fresh download..."
conda clean --all --yes

# 4. Create the new environment from the YAML file
echo "Creating new environment '$ENV_NAME' from $ENV_FILE..."
conda env create --file $ENV_FILE

# 5. Final check
if [ $? -eq 0 ]; then
    echo "✅ Successfully reinstalled environment '$ENV_NAME'."
    echo "Run 'conda activate $ENV_NAME' to use it."
else
    echo "❌ Failed to create environment '$ENV_NAME'."
fi