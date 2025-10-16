#!/bin/bash
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation of Dominick's Fine Foods, Inc. 1997-2003                                  #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /devops/rebuild_env.sh                                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday October 16th 2025 04:05:21 pm                                              #
# Modified   : Thursday October 16th 2025 05:03:42 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# ----------------- CONDA INITIALIZATION -----------------
# Find the base conda directory
CONDA_BASE=$(conda info --base)
# Source the conda setup script
source "$CONDA_BASE/etc/profile.d/conda.sh"
# --------------------------------------------------------
# This script rebuilds the conda environment from scratch.
echo "Rebuilding conda environment 'valuation' with necessary packages"
echo "This may take a few minutes..."
echo "Please wait..."
echo ""
conda deactivate
# Remove the existing environment if it exists
echo "Removing old environment..."
conda env remove -n valuation -y
echo "Old environment removed. Cleaning up..."
conda clean --all -y
echo "Cleanup complete."
echo ""
# Create new environment
echo "Creating new environment..."
conda create -n valuation python=3.12.11 -y
echo "Environment created."
echo ""
# Activate the environment and install packages
conda activate valuation
echo "Environment valuation activated."
echo ""
# Install packages from conda-forge
echo "Installing packages from conda-forge..."
conda install -c conda-forge pandas pyarrow pytest pytest-cov ipykernel tqdm typer pip loguru isort flake8 black isort -y
echo "Packages from conda-forge installed."
echo ""
# Install Dask from PyPI using pip
echo "Installing Dask from PyPI using pip..."
pip install "dask[complete]"
echo "Pip packages installed."
echo ""