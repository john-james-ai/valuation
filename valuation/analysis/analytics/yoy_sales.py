#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/analysis/analytics/yoy_sales.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday October 25th 2025 10:54:03 am                                              #
# Modified   : Saturday October 25th 2025 10:55:27 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Literal, Optional

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns

from valuation.analysis.analytics.base import Analytics

# ------------------------------------------------------------------------------------------------ #


class YoYSalesAnalytics(Analytics):
    """Year-on-Year sales analytics with support for company, store, and category levels."""

    @staticmethod
    def analyze(
        df: pl.DataFrame | pl.LazyFrame,
        level: Literal["company", "store", "category"] = "company",
        identifier: Optional[str] = None,
        **kwargs,
    ) -> pl.DataFrame | pl.LazyFrame:
        """
        Analyze Year-on-Year sales growth.

        Args:
            df: Polars DataFrame or LazyFrame with columns: year, revenue, store, category
            level: Analysis level - "company", "store", or "category"
            identifier: Specific store or category name to filter (required for store/category level)
            **kwargs: Additional arguments (e.g., sort_by="yoy_growth")

        Returns:
            DataFrame with YoY analysis including growth rates and changes
        """
        is_lazy = isinstance(df, pl.LazyFrame)

        # Filter by identifier if specified
        if identifier is not None:
            if level == "store":
                df = df.filter(pl.col("store") == identifier)
            elif level == "category":
                df = df.filter(pl.col("category") == identifier)

        # Group by appropriate columns
        if level == "company":
            group_cols = ["year"]
        elif level == "store":
            group_cols = ["year", "store"]
        elif level == "category":
            group_cols = ["year", "category"]
        else:
            raise ValueError(f"Invalid level: {level}. Must be 'company', 'store', or 'category'")

        # Aggregate revenue by year and level
        result = (
            df.group_by(group_cols).agg(pl.col("revenue").sum().alias("total_revenue")).sort("year")
        )

        # Calculate YoY metrics
        partition_cols = [col for col in group_cols if col != "year"]

        if partition_cols:
            result = result.with_columns(
                [pl.col("total_revenue").shift(1).over(partition_cols).alias("prev_year_revenue")]
            )
        else:
            result = result.with_columns(
                [pl.col("total_revenue").shift(1).alias("prev_year_revenue")]
            )

        # Calculate YoY change and growth rate
        result = result.with_columns(
            [
                (pl.col("total_revenue") - pl.col("prev_year_revenue")).alias("yoy_change"),
                (
                    (pl.col("total_revenue") - pl.col("prev_year_revenue"))
                    / pl.col("prev_year_revenue")
                    * 100
                )
                .round(2)
                .alias("yoy_growth_pct"),
            ]
        )

        # Apply any sorting from kwargs
        if "sort_by" in kwargs:
            result = result.sort(kwargs["sort_by"], descending=kwargs.get("descending", False))

        # Return in same format as input
        if is_lazy and isinstance(result, pl.DataFrame):
            return result.lazy()
        elif not is_lazy and isinstance(result, pl.LazyFrame):
            return result.collect()

        return result

    @staticmethod
    def visualize(
        df: pl.DataFrame | pl.LazyFrame,
        level: Literal["company", "store", "category"] = "company",
        identifier: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        Visualize Year-on-Year sales trends.

        Args:
            df: Polars DataFrame or LazyFrame with YoY analysis results
            level: Analysis level used in analyze()
            identifier: Specific identifier for title
            **kwargs: Additional plotting arguments (figsize, title, etc.)
        """
        # Ensure we have a DataFrame
        if isinstance(df, pl.LazyFrame):
            df = df.collect()

        # Set up the plot style
        sns.set_style("whitegrid")
        figsize = kwargs.get("figsize", (14, 6))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Determine title suffix
        title_suffix = ""
        if identifier:
            title_suffix = f" - {identifier}"
        elif level == "company":
            title_suffix = " - Company Wide"

        # Plot 1: Revenue trend
        if level in ["store", "category"] and identifier is None:
            # Multiple entities - show top 5
            entities = df.select(pl.col(level)).unique().to_series().to_list()[:5]
            for entity in entities:
                entity_data = df.filter(pl.col(level) == entity)
                ax1.plot(
                    entity_data["year"],
                    entity_data["total_revenue"],
                    marker="o",
                    label=entity,
                    linewidth=2,
                )
            ax1.legend()
        else:
            # Single entity or company level
            ax1.plot(
                df["year"],
                df["total_revenue"],
                marker="o",
                color="steelblue",
                linewidth=2,
                markersize=8,
            )

        ax1.set_xlabel("Year", fontsize=12)
        ax1.set_ylabel("Total Revenue", fontsize=12)
        ax1.set_title(f"Revenue Trend{title_suffix}", fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)

        # Plot 2: YoY Growth Rate
        growth_data = df.filter(pl.col("yoy_growth_pct").is_not_null())

        if level in ["store", "category"] and identifier is None:
            # Multiple entities
            for entity in entities:
                entity_data = growth_data.filter(pl.col(level) == entity)
                ax2.plot(
                    entity_data["year"],
                    entity_data["yoy_growth_pct"],
                    marker="s",
                    label=entity,
                    linewidth=2,
                )
            ax2.legend()
        else:
            # Single entity or company level
            colors = ["green" if x > 0 else "red" for x in growth_data["yoy_growth_pct"]]
            ax2.bar(
                growth_data["year"],
                growth_data["yoy_growth_pct"],
                color=colors,
                alpha=0.7,
                edgecolor="black",
            )

        ax2.axhline(y=0, color="black", linestyle="--", linewidth=1)
        ax2.set_xlabel("Year", fontsize=12)
        ax2.set_ylabel("YoY Growth (%)", fontsize=12)
        ax2.set_title(f"YoY Growth Rate{title_suffix}", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.show()
