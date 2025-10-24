#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Valuation - Discounted Cash Flow Method                                             #
# Version    : 0.1.0                                                                               #
# Python     : 3.12.11                                                                             #
# Filename   : /valuation/flow/modeling/model_selection/selector.py                                #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/valuation                                          #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday October 24th 2025 07:26:11 am                                                #
# Modified   : Friday October 24th 2025 09:43:53 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Evaluates model performance across hierarchy levels."""
from typing import Tuple

from hierarchicalforecast.core import HierarchicalReconciliation
from hierarchicalforecast.methods import MinTrace
from hierarchicalforecast.utils import aggregate
from lightgbm import LGBMRegressor  # The ML model
from loguru import logger
from mlforecast import MLForecast
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from valuation.asset.identity.model import ModelPassport
from valuation.asset.model.mlforecast import MLForecastModel
from valuation.flow.modeling.model_selection.base import ModelParams
from valuation.flow.modeling.model_selection.cv import CrossValidationHP
from valuation.flow.modeling.model_selection.light_gbm import LightGBMHP
from valuation.flow.modeling.model_selection.mlforecast import MLForecastHP
from valuation.flow.modeling.model_selection.performance import PerformanceMetrics
from valuation.infra.store.model import ModelStore
from valuation.utils.metrics import compute_smape, compute_wape
import wandb


# ------------------------------------------------------------------------------------------------ #
class ModelSelector:
    """Evaluates model performance across hierarchy levels."""

    def __init__(
        self,
        model_name: str,
        lgbm_params: LightGBMHP,
        mlforecast_params: MLForecastHP,
        cv_params: CrossValidationHP,
        model_store: type[ModelStore] = ModelStore,
    ) -> None:
        self.model_name = model_name

        self._lgbm_params = lgbm_params
        self._mlforecast_params = mlforecast_params
        self._cv_params = cv_params

        self._model_store = model_store()

        self._params: ModelParams | None = None
        self._passport: ModelPassport | None = None
        self._model_asset: MLForecastModel | None = None

        self._mlforecast_model = None
        self._lgbm_model = None
        self._cross_validation = None

        self._cv_df_base = None
        self._cv_df_reconciled = None
        self._cv_df_eval = None

        self._y_pred_aggregated = None
        self._y_pred_reconciled = None
        self._y_true_aggregated = None

        self._S_df = None
        self._tags = None

        self._spec = [["store"], ["category"], ["store", "category"]]

        self._model_cols = []

        self._run = wandb.init(
            project="valuation",
            entity="john-james-ai",
            name=f"ModelSelection-{self.model_name}",
            reinit=True,
        )
        self._run.config.update(
            {
                "model_name": self.model_name,
                "lgbm_params": self._lgbm_params.as_dict(),
                "mlforecast_params": self._mlforecast_params.as_dict(),
                "cv_params": self._cv_params.as_dict(),
            }
        )

        self._evaluation = None

    # -------------------------------------------------------------------------------------------- #
    def cross_validate(self, train_df: pd.DataFrame) -> None:
        """Performs cross-validation."""
        self._initialize()

        self._cv_df_base = self._mlforecast_model.cross_validation(train_df, **self._cv_params.as_dict()).fit(df=train_df)  # type: ignore

        _, self._S_df, self._tags = self._create_summing_matrix(train_df)

        self._y_pred_aggregated = self._aggregate_forecasts()

        self._cv_df_reconciled = self._reconcile_forecasts()

        self._evaluation = self.evaluate()
        self.diagnose(train_df)

        self._finalize()

    # -------------------------------------------------------------------------------------------- #
    def _finalize(self) -> None:
        """Finalizes the WandB run."""
        self._create_and_save_model()
        if self._evaluation is not None:
            self._run.log(self._evaluation.as_dict())
        else:
            raise ValueError("No evaluation results to log. Run evaluate() first.")
        self._run.finish()
        logger.info(self._params)
        logger.info(self._evaluation)

    def _create_and_save_model(self) -> None:
        """Saves the model to the model store."""
        self._passport = ModelPassport.create(
            name=self.model_name,
            description="MLForecast model for hierarchical time series forecasting.",
        )
        self._params = ModelParams(
            light_gbm=self._lgbm_params,
            mlforecast=self._mlforecast_params,
            cross_validation=self._cv_params,
        )

        self._model_asset = MLForecastModel(
            passport=self._passport,
            model=self._mlforecast_model,
            performance=self._evaluation,
            params=self._params,
        )

        self._model_store.add(model=self._model_asset, overwrite=True)

    def evaluate(
        self,
    ) -> PerformanceMetrics:

        if self._cv_df_base is None or self._cv_df_reconciled is None or self._cv_df_eval is None:
            raise ValueError(
                "Cross-validation dataframes are not set. Please run cross_validate() first."
            )
        # 1. Get y_true at ALL hierarchy levels
        y_true_base = self._cv_df_base[["unique_id", "ds", "cutoff", "y"]].copy()
        y_true_base["store"] = y_true_base["unique_id"].apply(lambda s: s.split("_")[0])
        y_true_base["category"] = y_true_base["unique_id"].apply(lambda s: s.split("_")[1])
        y_true_base_for_agg = y_true_base.drop(columns=["unique_id"])

        # Aggregate y_true
        self._y_true_aggregated, _, _ = aggregate(df=y_true_base_for_agg, spec=self._spec)

        # 3. Classify hierarchy levels properly
        def classify_level(uid):
            if "_" in uid:
                return "bottom"  # store_category (e.g., "100_beer")
            elif "/" in uid:
                return "store_category"  # aggregated store/category (e.g., "100/beer")
            else:
                # Check if it's a store (numeric) or category (text)
                try:
                    int(uid)
                    return "store"  # Just store (e.g., "100")
                except:
                    return "category"  # Just category (e.g., "beer")

        self._cv_df_eval["level"] = self._cv_df_eval["unique_id"].apply(classify_level)

        # 4. Get model columns
        model_cols = [
            col
            for col in self._cv_df_reconciled.columns
            if col not in ["unique_id", "ds", "cutoff"]
        ]

        print(f"Found model columns: {model_cols}")
        print(f"\nDataset overview:")
        print(f"  Total forecasts: {len(self._cv_df_eval):,}")
        print(f"  Unique series: {self._cv_df_eval['unique_id'].nunique():,}")
        print(f"  Date range: {self._cv_df_eval['ds'].min()} to {self._cv_df_eval['ds'].max()}")
        print(f"\nActual values summary:")
        print(self._cv_df_eval["y"].describe())

        # 5. Overall Performance
        print("\n" + "=" * 80)
        print("OVERALL PERFORMANCE (All Hierarchy Levels)")
        print("=" * 80)

        performance_results = []
        for model_col in model_cols:
            mask = self._cv_df_eval[[model_col, "y"]].notna().all(axis=1)
            y_true = self._cv_df_eval.loc[mask, "y"]
            y_pred = self._cv_df_eval.loc[mask, model_col]

            performance = {
                "model": model_col.replace("LGBMRegressor/", ""),  # Shorter names
                "MSE": mean_squared_error(y_true, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
                "MAE": mean_absolute_error(y_true, y_pred),
                "MAPE": mean_absolute_percentage_error(y_true, y_pred) * 100,  # As percentage
                "SMAPE": compute_smape(y_true, y_pred),
                "WAPE": compute_wape(y_true, y_pred),
                "y_bar": y_true.mean(),
                "n": len(y_true),
            }

            performance_results.append(performance)
            if model_col == self.model_name:
                evaluation = PerformanceMetrics(**performance)

        overall_perf = pd.DataFrame(performance_results).set_index("model")
        print(overall_perf.to_string())

        # 6. Performance by Hierarchy Level
        print("\n" + "=" * 80)
        print("PERFORMANCE BY HIERARCHY LEVEL")
        print("=" * 80)

        levels_order = ["bottom", "store_category", "store", "category"]
        level_results = []

        for level in levels_order:
            level_data = self._cv_df_eval[self._cv_df_eval["level"] == level]

            if len(level_data) == 0:
                continue

            print(f"\n{level.upper()} Level:")
            print(f"  Unique series: {level_data['unique_id'].nunique():,}")
            print(f"  Total forecasts: {len(level_data):,}")
            print(f"  Actual mean: {level_data['y'].mean():.2f}, std: {level_data['y'].std():.2f}")

            for model_col in model_cols:
                mask = level_data[[model_col, "y"]].notna().all(axis=1)
                y_true = level_data.loc[mask, "y"]
                y_pred = level_data.loc[mask, model_col]

                if len(y_true) > 0:

                    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
                    mae_val = mean_absolute_error(y_true, y_pred)
                    mape_val = mean_absolute_percentage_error(y_true, y_pred) * 100
                    smape_val = compute_smape(y_true, y_pred)
                    wape_val = compute_wape(y_true, y_pred)

                    performance = {
                        "Level": level,
                        "Model": model_col.replace("LGBMRegressor/", ""),
                        "RMSE": rmse_val,
                        "MAE": mae_val,
                        "MAPE%": mape_val,
                        "Mean_Actual": y_true.mean(),
                        "n": len(y_true),
                    }

                    level_results.append(performance)

                    # Normalized error (MAE as % of mean)
                    normalized_mae = (mae_val / y_true.mean() * 100) if y_true.mean() > 0 else 0

                    print(
                        f"    {model_col.replace('LGBMRegressor/', '')[:30]:30s} -> "
                        f"RMSE: {rmse_val:>10.2f}, MAE: {mae_val:>10.2f}, "
                        f"MAPE: {mape_val:>6.2f}%, Norm_MAE: {normalized_mae:>6.2f}%"
                    )

        # 7. Comparison Table
        print("\n" + "=" * 80)
        print("RECONCILIATION IMPACT (Comparing Base vs Reconciled)")
        print("=" * 80)

        level_perf_df = pd.DataFrame(level_results)
        if len(level_perf_df) > 0:
            comparison = level_perf_df.pivot_table(
                index="Level", columns="Model", values=["RMSE", "MAE", "MAPE%"]
            )
            print(comparison.to_string())

        # 8. Sample predictions vs y_true
        print("\n" + "=" * 80)
        print("SAMPLE PREDICTIONS (First 10 bottom-level forecasts)")
        print("=" * 80)

        sample = (
            self._cv_df_eval[self._cv_df_eval["level"] == "bottom"]
            .head(10)[["unique_id", "ds", "y"] + model_cols]
            .round(2)
        )
        print(sample.to_string())

        return evaluation

    def diagnose(
        self,
        train_df: pd.DataFrame,
    ) -> None:
        """Generates diagnostic plots for model evaluation."""
        if self._cv_df_eval is None or self._cv_df_base is None:
            raise ValueError(
                "Cross-validation dataframes are not set. Please run cross_validate() first."
            )
        print("=" * 80)
        print("PERFORMANCE DIAGNOSTICS")
        print("=" * 80)

        # 1. Check imputation impact
        print("\n1. IMPUTATION ANALYSIS")
        print("-" * 80)

        # Check what percentage of data was originally missing
        # This assumes train_df is your final densified/imputed dataset
        if "train_df" in locals():
            # Check series lengths - all should be equal after densification
            series_lengths = train_df.groupby("unique_id").size()
            print(f"Series lengths after densification:")
            print(f"  Min: {series_lengths.min()}, Max: {series_lengths.max()}")
            print(f"  All equal: {len(series_lengths.unique()) == 1}")

            # If you tracked NaNs before imputation, report it
            # Otherwise, skip this section
            print("\n⚠️  Note: Imputation percentage not tracked.")
            print("   If performance is poor, imputation may be the cause.")

        # 2. Check training data quality
        print("\n2. TRAINING DATA QUALITY")
        print("-" * 80)

        if "train_df" in locals():
            print(f"Training data shape: {train_df.shape}")
            print(f"Unique series: {train_df['unique_id'].nunique()}")
            print(f"Date range: {train_df['ds'].min()} to {train_df['ds'].max()}")

            # Check for zero/near-zero values
            zero_pct = (train_df["y"] == 0).sum() / len(train_df) * 100
            near_zero_pct = (train_df["y"] < 1).sum() / len(train_df) * 100
            print(f"\nZero values: {zero_pct:.1f}%")
            print(f"Near-zero (<1): {near_zero_pct:.1f}%")

            # Revenue distribution
            print(f"\nRevenue distribution:")
            print(train_df["y"].describe())

            # Check for negative values
            if (train_df["y"] < 0).any():
                print(f"⚠️  WARNING: {(train_df['y'] < 0).sum()} negative values found!")

        # 3. Check predictions quality
        print("\n3. PREDICTION QUALITY ANALYSIS")
        print("-" * 80)

        if "cv_df_eval" in locals() and "model_cols" in locals():
            # Check for extreme predictions
            for model_col in self._model_cols:
                preds = self._cv_df_eval[model_col].dropna()
                y_true = self._cv_df_eval["y"].dropna()

                print(f"\n{model_col}:")
                print(f"  Predictions range: [{preds.min():.2f}, {preds.max():.2f}]")
                print(f"  Actuals range: [{y_true.min():.2f}, {y_true.max():.2f}]")
                print(
                    f"  Predictions mean: {preds.mean():.2f} vs Actuals mean: {y_true.mean():.2f}"
                )

                # Check for negative predictions
                neg_preds = (preds < 0).sum()
                if neg_preds > 0:
                    print(
                        f"  ⚠️  {neg_preds} negative predictions ({neg_preds/len(preds)*100:.1f}%)"
                    )

                # Check for extreme predictions
                extreme_high = (preds > y_true.quantile(0.99) * 2).sum()
                if extreme_high > 0:
                    print(f"  ⚠️  {extreme_high} extremely high predictions")

        # 4. Residual analysis
        print("\n4. RESIDUAL ANALYSIS")
        print("-" * 80)

        if "cv_df_eval" in locals() and len(self._model_cols) > 0:
            model_col = self._model_cols[0]  # Analyze first model

            mask = self._cv_df_eval[[model_col, "y"]].notna().all(axis=1)
            y_true = self._cv_df_eval.loc[mask, "y"]
            preds = self._cv_df_eval.loc[mask, model_col]
            residuals = y_true - preds

            print(f"Analyzing {model_col}:")
            print(f"  Mean residual: {residuals.mean():.2f}")
            print(f"  Median residual: {residuals.median():.2f}")
            print(f"  Residual std: {residuals.std():.2f}")
            print(f"  Mean absolute error: {np.abs(residuals).mean():.2f}")

            # Check for systematic bias
            if abs(residuals.mean()) > y_true.std() * 0.1:  # type: ignore
                print(f"  ⚠️  WARNING: Systematic bias detected!")
                if residuals.mean() > 0:  # type: ignore
                    print(f"     Model is UNDER-predicting on average")
                else:
                    print(f"     Model is OVER-predicting on average")

        # 5. Check differences transformation
        print("\n5. DIFFERENCING IMPACT")
        print("-" * 80)

        if "TARGET_TRANSFORMS" in locals() and len(self._mlforecast_params.target_transforms) > 0:
            print(f"Target transforms applied: {self._mlforecast_params.target_transforms}")
            print("⚠️  Differencing can cause issues if:")
            print("   - Series are short")
            print("   - Series have many imputed values")
            print("   - Series are already stationary")
            print(
                "\n💡 RECOMMENDATION: Try removing Differences([1]) and see if performance improves"
            )

        # 6. Check for data leakage
        print("\n6. DATA LEAKAGE CHECK")
        print("-" * 80)

        if "cv_df_base" in locals():
            # Check if CV is working correctly
            cv_windows = self._cv_df_base.groupby(["unique_id", "cutoff"]).size()
            print(f"CV windows per series: {cv_windows.groupby(level=0).size().value_counts()}")

            # Check cutoff dates
            print(f"\nCutoff dates: {sorted(self._cv_df_base['cutoff'].unique())}")

        # 7. Specific problematic series
        print("\n7. WORST PERFORMING SERIES")
        print("-" * 80)

        if "cv_df_eval" in locals() and "self._model_cols" in locals():
            model_col = self._model_cols[0]

            # Calculate MAE per series
            series_mae = (
                self._cv_df_eval.groupby("unique_id")
                .apply(
                    lambda x: (
                        mean_absolute_error(x["y"].dropna(), x[model_col].dropna())
                        if len(x["y"].dropna()) > 0
                        else np.nan
                    )
                )
                .sort_values(ascending=False)
            )

            print("Top 10 worst series (by MAE):")
            for uid, mae in series_mae.head(10).items():  # type: ignore
                series_data = self._cv_df_eval[self._cv_df_eval["unique_id"] == uid]
                actual_mean = series_data["y"].mean()
                pred_mean = series_data[model_col].mean()
                print(
                    f"  {uid:30s} MAE: {mae:>10.2f}, Actual mean: {actual_mean:>10.2f}, Pred mean: {pred_mean:>10.2f}"
                )

        print("\n" + "=" * 80)
        print("RECOMMENDATIONS")
        print("=" * 80)
        print(
            """
        Based on diagnostics above, try these fixes:

        1. REMOVE DIFFERENCING if series are short or heavily imputed:
        TARGET_TRANSFORMS = []  # Remove Differences([1])

        2. REDUCE LAGS if many series are short:
        LAGS = [1, 2, 4, 8, 13]  # Remove 26, 52

        3. FILTER OUT heavily imputed series (>30% imputed)

        4. CHECK if forward/backward fill is creating unrealistic patterns

        5. INCREASE min_child_samples for more regularization:
        min_child_samples=50 or 100

        6. ADD more regularization:
        reg_alpha=1.0, reg_lambda=1.0
        """
        )

    # -------------------------------------------------------------------------------------------- #
    def _initialize(self) -> None:
        """Initializes the MLForecast model with LightGBM as the base learner."""
        self._lgbm_model = self._create_lgbm_model()
        self._mlforecast_model = self._create_mlforecast_model()

    # -------------------------------------------------------------------------------------------- #
    def _create_lgbm_model(self) -> LGBMRegressor:
        """Creates the MLForecast model with LightGBM as the base learner."""
        return LGBMRegressor(**self._lgbm_params.as_dict())

    # -------------------------------------------------------------------------------------------- #
    def _create_mlforecast_model(self) -> MLForecast:
        """Creates the MLForecast model with LightGBM as the base learner."""
        lgbm_model = self._create_lgbm_model()
        return MLForecast(models={lgbm_model}, **self._mlforecast_params.as_dict())

    # -------------------------------------------------------------------------------------------- #
    def _create_summing_matrix(
        self, train_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        # 1. Start with the core data (unique_id, ds, y)
        hierarchy_df = train_df[["unique_id", "ds", "y"]].drop_duplicates()

        # 2. Create the grouping columns
        hierarchy_df["store"] = hierarchy_df["unique_id"].apply(lambda s: s.split("_")[0])
        hierarchy_df["category"] = hierarchy_df["unique_id"].apply(lambda s: s.split("_")[1])

        # 3. Drop the 'unique_id' column from the input DF before aggregation
        #    The aggregate function knows to use the combination of columns in 'spec'
        #    to uniquely identify the time series levels.
        hierarchy_df_clean = hierarchy_df.drop(columns=["unique_id"])  # 👈 ADD THIS LINE

        # Pass the cleaned DataFrame to the aggregate function
        return aggregate(df=hierarchy_df_clean, spec=self._spec)

    def _aggregate_forecasts(self) -> pd.DataFrame:
        if self._spec is None or self._spec is None or self._cv_df_base is None:
            raise ValueError("spec or cv_df_base not set. Cannot aggregate forecasts.")

        # Clean and Prepare Y_hat_df_base
        if self._cv_df_base is not None:
            Y_hat_df_base = self._cv_df_base.drop(columns=["cutoff", "y"], errors="ignore")
        else:
            raise ValueError("cv_df_base is not set. Cannot aggregate forecasts.")

        # 1. Add the hierarchy columns to the base forecasts
        Y_hat_df_base_clean = Y_hat_df_base.copy()
        Y_hat_df_base_clean["store"] = Y_hat_df_base_clean["unique_id"].apply(
            lambda s: s.split("_")[0]
        )
        Y_hat_df_base_clean["category"] = Y_hat_df_base_clean["unique_id"].apply(
            lambda s: s.split("_")[1]
        )

        # 2. Identify the forecast column(s) - typically 'LGBMRegressor' or similar model name
        forecast_col = "LGBMRegressor"  # Adjust if your column has a different name

        # 3. Rename forecast column to 'y' temporarily for aggregation
        Y_hat_df_base_for_agg = Y_hat_df_base_clean.copy()
        Y_hat_df_base_for_agg = Y_hat_df_base_for_agg.rename(columns={forecast_col: "y"})
        Y_hat_df_base_for_agg = Y_hat_df_base_for_agg.drop(columns=["unique_id"])

        # 4. Aggregate to create forecasts at all hierarchy levels
        Y_hat_aggregated, _, _ = aggregate(df=Y_hat_df_base_for_agg, spec=self._spec)

        # 5. Rename back to original forecast column name
        Y_hat_aggregated = Y_hat_aggregated.rename(columns={"y": forecast_col})

        print("Forecasts aggregated successfully across all hierarchy levels.")
        print(f"Base forecasts shape: {Y_hat_df_base.shape}")
        print(f"Aggregated forecasts shape: {Y_hat_aggregated.shape}")
        print(f"Unique IDs in aggregated forecasts: {Y_hat_aggregated['unique_id'].nunique()}")

        return Y_hat_aggregated

    def _reconcile_forecasts(self) -> pd.DataFrame:
        reconcilers = [MinTrace(method="ols")]
        hrec = HierarchicalReconciliation(reconcilers=reconcilers)

        if (
            self._cv_df_base is None
            or self._y_pred_aggregated is None
            or self._S_df is None
            or self._tags is None
        ):
            raise ValueError(
                "cv_df_base, y_pred_aggregated, S_df, or tags not set. Cannot reconcile forecasts."
            )

        # Prepare aggregated y_true
        Y_df_base = self._cv_df_base[["unique_id", "ds", "y"]].copy()
        Y_df_base["store"] = Y_df_base["unique_id"].apply(lambda s: s.split("_")[0])
        Y_df_base["category"] = Y_df_base["unique_id"].apply(lambda s: s.split("_")[1])
        Y_df_base_for_agg = Y_df_base.drop(columns=["unique_id"])
        Y_df_y_true, _, _ = aggregate(df=Y_df_base_for_agg, spec=self._spec)

        # Reconcile (this adjusts the aggregated forecasts for coherence)
        cv_df_reconciled = hrec.reconcile(
            Y_hat_df=self._y_pred_aggregated,  # Now has all hierarchy levels
            Y_df=Y_df_y_true,
            S_df=self._S_df,
            tags=self._tags,
        )
        self._model_cols = [
            col for col in cv_df_reconciled.columns if col not in ["unique_id", "ds", "cutoff"]
        ]

        return cv_df_reconciled
